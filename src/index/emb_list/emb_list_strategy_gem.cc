// Copyright (C) 2019-2023 Zilliz. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under the License.

#include <algorithm>
#include <cstring>
#include <limits>
#include <vector>

#include "knowhere/comp/index_param.h"
#include "knowhere/index/emb_list_strategy.h"
#include "knowhere/index/index_node.h"
#include "knowhere/log.h"
#include "knowhere/utils.h"

#if defined(NOT_COMPILE_FOR_SWIG) && !defined(KNOWHERE_WITH_LIGHT)
#include "knowhere/comp/time_recorder.h"
#include "knowhere/prometheus_client.h"
#endif

namespace knowhere {

// GEM hands whole emb_lists to a base index that indexes sets itself (SVS_GEM). It performs no
// encoding of its own: the base index searches document sets directly and returns doc ids, which
// this strategy reranks exactly, as TokenANN/MUVERA do, via CalcDistByStorageIds.
class GemEmbListStrategy : public EmbListStrategy {
 public:
    [[nodiscard]] std::string
    Type() const override {
        return meta::EMB_LIST_STRATEGY_GEM;
    }

    expected<EmbListBuildShape>
    GetBuildShape(const DataSetPtr dataset, size_t num_docs, const BaseConfig& config) const override {
        return EmbListBuildShape{.rows = dataset->GetRows(), .dim = dataset->GetDim()};
    }

    expected<std::optional<DataSetPtr>>
    PrepareDataForBuild(const DataSetPtr dataset, const EmbListOffset& doc_offset, const BaseConfig& config) override {
        emb_list_offset_ = std::make_shared<EmbListOffset>(doc_offset.offset);
        // Unchanged: the base SVS_GEM node reads tokens and EMB_LIST_OFFSET straight off this dataset.
        return std::optional<DataSetPtr>(dataset);
    }

    [[nodiscard]] bool
    NeedsBaseIndexIDMap() const override {
        return false;
    }

    [[nodiscard]] bool
    NeedsRawVectorStorage() const override {
        return true;
    }

    expected<DataSetPtr>
    Search(const DataSetPtr query_dataset, const EmbListOffset& query_offset, const IndexNode* index,
           std::unique_ptr<Config> cfg, const BitsetView& bitset, milvus::OpContext* op_context) const override {
        if (!emb_list_offset_) {
            return expected<DataSetPtr>::Err(Status::emb_list_inner_error, "gem strategy not initialized");
        }

        auto& config = static_cast<BaseConfig&>(*cfg);
        auto k = config.k.value();
        auto metric_or = ParseEmbListMetric(config);
        if (!metric_or.has_value()) {
            return expected<DataSetPtr>::Err(metric_or.error(), metric_or.what());
        }
        auto& mi = metric_or.value();

        auto num_query_docs = query_offset.num_el();
        auto num_docs = emb_list_offset_->num_el();

        auto retrieval_ann_ratio = config.retrieval_ann_ratio.value();
        if (retrieval_ann_ratio <= 0.0f) {
            auto err_msg = "retrieval_ann_ratio could not be less than or equal to 0";
            LOG_KNOWHERE_WARNING_ << err_msg;
            return expected<DataSetPtr>::Err(Status::emb_list_inner_error, err_msg);
        }
        int32_t ann_k = std::min(std::max(static_cast<int32_t>(k * retrieval_ann_ratio), 1),
                                 static_cast<int32_t>(num_docs));

#if defined(NOT_COMPILE_FOR_SWIG) && !defined(KNOWHERE_WITH_LIGHT)
        knowhere_search_emb_list_retrieval_ann_ratio.Observe(retrieval_ann_ratio);
        TimeRecorder rc("GEM Search - 1st round ann search");
#endif

        // The base node reads query tokens and query EMB_LIST_OFFSET straight off query_dataset,
        // which already carries both, since IndexNode::SearchEmbList forwards them unchanged.
        config.k = ann_k;
        config.metric_type = mi.sub_metric_type;
        auto ann_result = index->Search(query_dataset, std::move(cfg), bitset, op_context);
        if (!ann_result.has_value()) {
            if (ann_result.error() == Status::cancelled) {
                return expected<DataSetPtr>::Err(Status::cancelled, ann_result.what());
            }
            LOG_KNOWHERE_ERROR_ << "GEM ANN search failed: " << ann_result.what();
            return expected<DataSetPtr>::Err(Status::emb_list_inner_error, "ANN search failed");
        }
        const auto* ann_ids = ann_result.value()->GetIds();

        // Base search returns public doc ids (NeedsBaseIndexIDMap() is false): map back to compact
        // ids before using emb_list_offset_ and CalcDistByStorageIds.
        auto to_in_doc_id = [&](int64_t out_id) -> expected<int64_t> {
            if (out_id < 0) {
                return -1;
            }
            const auto in_id = index->MapOutToIn(out_id);
            if (in_id < 0) {
                return expected<int64_t>::Err(Status::invalid_args, "doc id maps outside compact storage");
            }
            return in_id;
        };

#if defined(NOT_COMPILE_FOR_SWIG) && !defined(KNOWHERE_WITH_LIGHT)
        auto time = rc.ElapseFromBegin("done");
        time *= 0.001;
        knowhere_search_emb_list_1st_ann_latency.Observe(time);
        TimeRecorder rc2("GEM Search - 2nd round bf and agg");
#endif

        auto dim = query_dataset->GetDim();
        const auto* query_data = static_cast<const float*>(query_dataset->GetTensor());

        auto ids = std::make_unique<int64_t[]>(num_query_docs * k);
        auto dists = std::make_unique<float[]>(num_query_docs * k);

        size_t total_doc_vecs = 0;
        size_t total_distance_computations = 0;
        std::vector<int64_t> candidate_docs;
        candidate_docs.reserve(ann_k);
        for (size_t q = 0; q < num_query_docs; ++q) {
            size_t q_vec_start = query_offset.offset[q];
            size_t q_vec_end = query_offset.offset[q + 1];
            size_t nq = q_vec_end - q_vec_start;

            candidate_docs.clear();
            for (int32_t i = 0; i < ann_k; ++i) {
                auto doc_id_or = to_in_doc_id(ann_ids[q * ann_k + i]);
                if (!doc_id_or.has_value()) {
                    return expected<DataSetPtr>::Err(doc_id_or.error(), doc_id_or.what());
                }
                int64_t doc_id = doc_id_or.value();
                if (doc_id >= 0 && static_cast<size_t>(doc_id) < num_docs) {
                    candidate_docs.push_back(doc_id);
                }
            }

            auto bf_query_dataset = GenDataSet(nq, dim, query_data + q_vec_start * dim);
            auto status = RerankByCalcDistByStorageIds(candidate_docs, bf_query_dataset, nq, k, mi.larger_is_closer,
                                                       mi.is_cosine, emb_list_offset_, index, bitset, op_context,
                                                       mi.agg_func, ids.get() + q * k, dists.get() + q * k,
                                                       total_doc_vecs, total_distance_computations);
            if (status == Status::cancelled) {
                return expected<DataSetPtr>::Err(Status::cancelled, "rerank cancelled by the caller");
            }
            if (status != Status::success) {
                return expected<DataSetPtr>::Err(Status::emb_list_inner_error, "rerank distance computation error");
            }
        }

#if defined(NOT_COMPILE_FOR_SWIG) && !defined(KNOWHERE_WITH_LIGHT)
        auto time2 = rc2.ElapseFromBegin("done");
        time2 *= 0.001;
        knowhere_search_emb_list_2nd_bf_agg_latency.Observe(time2);
#endif

        return GenResultDataSet(static_cast<int64_t>(num_query_docs), static_cast<int64_t>(k), std::move(ids),
                                std::move(dists));
    }

    Status
    Serialize(std::shared_ptr<uint8_t[]>& data, int64_t& size) const override {
        // Blob format: [magic][version][offsets]
        constexpr int32_t kMagic = 0x47454D42;  // "GEMB"
        constexpr int32_t kVersion = 1;
        size_t offset_size = EmbListOffsetByteSize(emb_list_offset_);
        size = static_cast<int64_t>(2 * sizeof(int32_t) + offset_size);
        data = std::shared_ptr<uint8_t[]>(new uint8_t[size]);
        uint8_t* ptr = data.get();

        std::memcpy(ptr, &kMagic, sizeof(int32_t));
        ptr += sizeof(int32_t);
        std::memcpy(ptr, &kVersion, sizeof(int32_t));
        ptr += sizeof(int32_t);
        SerializeEmbListOffsetToBytes(emb_list_offset_, ptr);
        return Status::success;
    }

    Status
    Deserialize(const uint8_t* data, int64_t size, const BaseConfig& config) override {
        constexpr int32_t kMagic = 0x47454D42;  // "GEMB"
        constexpr int64_t kHeaderSize = 2 * sizeof(int32_t);
        if (size < kHeaderSize) {
            LOG_KNOWHERE_ERROR_ << "GEM: blob too small: " << size << " < " << kHeaderSize;
            return Status::emb_list_inner_error;
        }

        const uint8_t* ptr = data;
        int32_t magic = 0;
        std::memcpy(&magic, ptr, sizeof(int32_t));
        ptr += sizeof(int32_t);
        if (magic != kMagic) {
            LOG_KNOWHERE_ERROR_ << "GEM: invalid magic number 0x" << std::hex << magic;
            return Status::emb_list_inner_error;
        }

        int32_t version = 0;
        std::memcpy(&version, ptr, sizeof(int32_t));
        ptr += sizeof(int32_t);
        if (version > 1) {
            LOG_KNOWHERE_ERROR_ << "GEM: unsupported serialization version " << version << " (max supported: 1)";
            return Status::emb_list_inner_error;
        }

        DeserializeEmbListOffsetFromBytes(ptr, emb_list_offset_);
        return Status::success;
    }

    [[nodiscard]] int64_t
    GetDocCount() const override {
        return emb_list_offset_ ? emb_list_offset_->num_el() : 0;
    }

    [[nodiscard]] std::shared_ptr<EmbListOffset>
    GetEmbListOffset() const override {
        return emb_list_offset_;
    }

 private:
    std::shared_ptr<EmbListOffset> emb_list_offset_;
};

EmbListStrategyPtr
CreateGemEmbListStrategy() {
    return std::make_unique<GemEmbListStrategy>();
}

}  // namespace knowhere
