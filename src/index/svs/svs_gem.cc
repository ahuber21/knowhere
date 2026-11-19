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

#if defined(KNOWHERE_WITH_SVS) && defined(FAISS_SVS_HAS_GEM)

#include <limits>
#include <optional>

#include "common/metric.h"
#include "faiss/cppcontrib/knowhere/impl/CountSizeIOWriter.h"
#include "faiss/index_io.h"
#include "faiss/svs/IndexSVSGem.h"
#include "io/memory_io.h"
#include "knowhere/bitsetview_idselector.h"
#include "knowhere/comp/index_param.h"
#include "knowhere/comp/task.h"
#include "knowhere/context.h"
#include "knowhere/emb_list_utils.h"
#include "knowhere/feature.h"
#include "knowhere/index/index_factory.h"
#include "knowhere/log.h"
#include "knowhere/utils.h"
#include "index/svs/svs_config.h"

namespace knowhere {

namespace {
// 0: NorthWest, 1: CheapestRow, matching faiss::IndexSVSGem::transport_start.
std::optional<int>
str_to_gem_transport_start(const std::string& s) {
    if (s == "northwest") {
        return 0;
    }
    if (s == "cheapestrow") {
        return 1;
    }
    return std::nullopt;
}
}  // namespace

// SVS_GEM indexes multi-vector documents (sets of tokens) directly: the base index holds the sets
// itself, so it only ever sees the "gem" emb_list strategy and fp32 data.
class SvsGemIndexNode : public IndexNode {
 public:
    SvsGemIndexNode(const int32_t version, const Object& object) : IndexNode(version), index_(nullptr) {
    }

    Status
    Train(const DataSetPtr dataset, std::shared_ptr<Config> cfg, bool use_knowhere_build_pool) override {
        return Status::not_implemented;
    }

    Status
    Add(const DataSetPtr dataset, std::shared_ptr<Config> cfg, bool use_knowhere_build_pool) override {
        LOG_KNOWHERE_ERROR_ << "SVS_GEM does not support incremental add; a segment gets GEM only once it seals.";
        return Status::not_implemented;
    }

    Status
    Build(const DataSetPtr dataset, std::shared_ptr<Config> cfg, bool use_knowhere_build_pool,
          milvus::OpContext* op_context) override {
        const SvsGemConfig& g_cfg = static_cast<const SvsGemConfig&>(*cfg);

        if (g_cfg.emb_list_strategy.value_or("") != meta::EMB_LIST_STRATEGY_GEM) {
            LOG_KNOWHERE_ERROR_ << "SVS_GEM requires emb_list_strategy=gem";
            return Status::invalid_args;
        }

        const size_t* lims = dataset->Get<const size_t*>(meta::EMB_LIST_OFFSET);
        if (lims == nullptr) {
            LOG_KNOWHERE_ERROR_ << "SVS_GEM requires emb_list data (missing EMB_LIST_OFFSET)";
            return Status::invalid_args;
        }
        size_t num_docs = 0;
        try {
            num_docs = GetEmbListCount(dataset);
        } catch (const std::exception& e) {
            LOG_KNOWHERE_ERROR_ << "SVS_GEM: " << e.what();
            return Status::invalid_args;
        }

        auto transport_start = str_to_gem_transport_start(g_cfg.gem_transport_start.value());
        if (!transport_start.has_value()) {
            LOG_KNOWHERE_ERROR_ << "unknown gem_transport_start: " << g_cfg.gem_transport_start.value();
            return Status::invalid_args;
        }

        const auto dim = dataset->GetDim();
        const auto* tokens = static_cast<const float*>(dataset->GetTensor());

        try {
            auto idx = std::make_unique<faiss::IndexSVSGem>(dim);
            idx->num_centroids = g_cfg.gem_num_centroids.value();
            idx->graph_max_degree = g_cfg.svs_graph_max_degree.value();
            idx->construction_window_size = g_cfg.svs_construction_window_size.value();
            idx->alpha = g_cfg.svs_alpha.value();
            idx->transport_start = transport_start.value();
            idx->build_router = g_cfg.gem_build_router.value();
            idx->route_inserts = g_cfg.gem_route_inserts.value();
            idx->num_threads = GetBuildThreadPoolSize();
            idx->build(static_cast<faiss::idx_t>(num_docs), tokens, lims);
            index_ = std::move(idx);
        } catch (const std::exception& e) {
            LOG_KNOWHERE_ERROR_ << "SVS GEM build error: " << e.what();
            return Status::faiss_inner_error;
        }

        return Status::success;
    }

    expected<DataSetPtr>
    Search(const DataSetPtr dataset, std::unique_ptr<Config> cfg, const BitsetView& bitset,
           milvus::OpContext* op_context) const override {
        if (!index_) {
            LOG_KNOWHERE_WARNING_ << "search on empty index";
            return expected<DataSetPtr>::Err(Status::empty_index, "index not loaded");
        }

        const SvsGemConfig& g_cfg = static_cast<const SvsGemConfig&>(*cfg);
        auto k = g_cfg.k.value();

        const size_t* q_lims = dataset->Get<const size_t*>(meta::EMB_LIST_OFFSET);
        if (q_lims == nullptr) {
            LOG_KNOWHERE_ERROR_ << "SVS_GEM search requires emb_list query offsets";
            return expected<DataSetPtr>::Err(Status::emb_list_inner_error, "missing query emb_list offset");
        }
        size_t num_queries = 0;
        try {
            num_queries = GetEmbListCount(dataset);
        } catch (const std::exception& e) {
            return expected<DataSetPtr>::Err(Status::emb_list_inner_error, e.what());
        }

        const auto* tokens = static_cast<const float*>(dataset->GetTensor());

        auto ids = std::make_unique<int64_t[]>(k * num_queries);
        auto distances = std::make_unique<float[]>(k * num_queries);

        faiss::SearchParametersSVSGem sp;
        sp.search_window_size = std::max(static_cast<size_t>(g_cfg.svs_search_window_size.value()),
                                         static_cast<size_t>(k));
        sp.router_nprobe = static_cast<size_t>(g_cfg.gem_router_nprobe.value());
        std::unique_ptr<BitsetViewIDSelector> bw_idselector;
        if (!bitset.empty() || bitset.num_bits() != 0) {
            bw_idselector = std::make_unique<BitsetViewIDSelector>(bitset);
            sp.sel = bw_idselector.get();
        }

        try {
            knowhere::checkCancellation(op_context);
            index_->search_sets(static_cast<faiss::idx_t>(num_queries), tokens, q_lims, k, distances.get(), ids.get(),
                                &sp);
        } catch (const std::exception& e) {
            LOG_KNOWHERE_WARNING_ << "error inner faiss: " << e.what();
            return expected<DataSetPtr>::Err(Status::faiss_inner_error, e.what());
        }

        // search_sets reports qChamfer distance (smaller is closer); MAX_SIM_IP expects larger is
        // closer, and a missing result (-1 label) sorts last under that convention.
        for (int64_t i = 0; i < k * static_cast<int64_t>(num_queries); ++i) {
            distances[i] = (ids[i] < 0) ? -std::numeric_limits<float>::infinity() : -distances[i];
        }

        auto res = GenResultDataSet(static_cast<int64_t>(num_queries), k, std::move(ids), std::move(distances));
        MapSearchResultIdsToOutIds(res);
        return res;
    }

    expected<DataSetPtr>
    GetVectorByIds(const DataSetPtr dataset, milvus::OpContext* op_context) const override {
        return expected<DataSetPtr>::Err(Status::not_implemented, "SVS_GEM does not support vector retrieval");
    }

    bool
    HasRawData(const std::string& metric_type) const override {
        return false;
    }

    static bool
    StaticHasRawData(const knowhere::BaseConfig& config, const IndexVersion& version) {
        return false;
    }

    expected<DataSetPtr>
    GetIndexMeta(std::unique_ptr<Config>) const override {
        return expected<DataSetPtr>::Err(Status::not_implemented, "GetIndexMeta not implemented");
    }

    Status
    Serialize(BinarySet& binset) const override {
        if (!index_) {
            LOG_KNOWHERE_ERROR_ << "Can not serialize empty index.";
            return Status::empty_index;
        }

        try {
            MemoryIOWriter writer;
            faiss::write_index(index_.get(), &writer);
            std::shared_ptr<uint8_t[]> data(writer.data());
            binset.Append(Type(), data, writer.tellg());
            return Status::success;
        } catch (const std::exception& e) {
            LOG_KNOWHERE_WARNING_ << "error inner faiss: " << e.what();
            return Status::faiss_inner_error;
        }
    }

    Status
    Deserialize(const BinarySet& binset, std::shared_ptr<Config>) override {
        auto binary = binset.GetByNames({Type()});
        if (binary == nullptr) {
            LOG_KNOWHERE_ERROR_ << "Invalid binary set.";
            return Status::invalid_binary_set;
        }

        MemoryIOReader reader(binary->data.get(), binary->size);
        try {
            auto index = std::unique_ptr<faiss::Index>(faiss::read_index(&reader));
            if (!dynamic_cast<faiss::IndexSVSGem*>(index.get())) {
                return Status::invalid_binary_set;
            }
            index_.reset(static_cast<faiss::IndexSVSGem*>(index.release()));
        } catch (const std::exception& e) {
            LOG_KNOWHERE_WARNING_ << "error inner faiss: " << e.what();
            return Status::faiss_inner_error;
        }

        return Status::success;
    }

    Status
    DeserializeFromFile(const std::string& filename, std::shared_ptr<Config> cfg) override {
        // No mmap path: GEM keeps the id sets, graph and centroids in memory, not the token store.
        try {
            auto index = std::unique_ptr<faiss::Index>(faiss::read_index(filename.data()));
            if (!dynamic_cast<faiss::IndexSVSGem*>(index.get())) {
                return Status::invalid_binary_set;
            }
            index_.reset(static_cast<faiss::IndexSVSGem*>(index.release()));
        } catch (const std::exception& e) {
            LOG_KNOWHERE_WARNING_ << "error inner faiss: " << e.what();
            return Status::faiss_inner_error;
        }

        return Status::success;
    }

    static std::unique_ptr<BaseConfig>
    StaticCreateConfig() {
        return std::make_unique<SvsGemConfig>();
    }

    std::unique_ptr<BaseConfig>
    CreateConfig() const override {
        return StaticCreateConfig();
    }

    static Status
    StaticConfigCheck(const Config& cfg, PARAM_TYPE paramType, std::string& msg) {
        if (paramType == PARAM_TYPE::TRAIN) {
            auto& gem_cfg = static_cast<const BaseConfig&>(cfg);
            auto strategy = gem_cfg.emb_list_strategy.value_or("");
            if (strategy != meta::EMB_LIST_STRATEGY_GEM) {
                msg = "SVS_GEM requires emb_list_strategy=gem, got '" + strategy + "'";
                return Status::invalid_args;
            }
            if (!get_el_metric_type(gem_cfg.metric_type.value()).has_value()) {
                msg = "SVS_GEM requires an emb_list metric type (e.g. MAX_SIM_IP), got '" +
                      gem_cfg.metric_type.value() + "'";
                return Status::invalid_metric_type;
            }
        }
        return Status::success;
    }

    int64_t
    Dim() const override {
        if (!index_) {
            return -1;
        }
        return index_->d;
    }

    int64_t
    Size() const override {
        if (!index_) {
            return 0;
        }

        faiss::cppcontrib::knowhere::CountSizeIOWriter writer;
        faiss::write_index(index_.get(), &writer);
        return writer.total_size;
    }

    int64_t
    Count() const override {
        if (!index_) {
            return -1;
        }
        return index_->ntotal;
    }

    std::string
    Type() const override {
        return knowhere::IndexEnum::INDEX_SVS_GEM;
    }

 protected:
    std::unique_ptr<faiss::IndexSVSGem> index_;
};

KNOWHERE_SIMPLE_REGISTER_GLOBAL(SVS_GEM, SvsGemIndexNode, fp32, knowhere::feature::EMB_LIST | knowhere::feature::FLOAT32)

}  // namespace knowhere

#endif  // defined(KNOWHERE_WITH_SVS) && defined(FAISS_SVS_HAS_GEM)
