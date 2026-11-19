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

#include <cstdint>
#include <limits>
#include <random>
#include <vector>

#include "catch2/catch_test_macros.hpp"
#include "knowhere/bitsetview.h"
#include "knowhere/comp/index_param.h"
#include "knowhere/dataset.h"
#include "knowhere/index/index_factory.h"
#include "knowhere/utils.h"
#include "utils.h"

namespace {

// Builds a synthetic emb_list dataset of `num_docs` documents, each with a random token count in
// [min_len, max_len], dim `dim`. Reusing the same tensor and offsets as a query dataset makes every
// query doc identical to one indexed doc, so top-1 should name that doc for most queries.
knowhere::DataSetPtr
GenGemEmbListDataSet(int64_t num_docs, int64_t dim, int min_len, int max_len, uint64_t seed) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> len_distrib(min_len, max_len);
    std::uniform_real_distribution<float> val_distrib(-1.0f, 1.0f);

    auto offsets = std::make_unique<size_t[]>(num_docs + 1);
    offsets[0] = 0;
    for (int64_t i = 0; i < num_docs; ++i) {
        offsets[i + 1] = offsets[i] + static_cast<size_t>(len_distrib(rng));
    }
    const size_t total_tokens = offsets[num_docs];

    auto* tokens = new float[total_tokens * dim];
    for (size_t i = 0; i < total_tokens * static_cast<size_t>(dim); ++i) {
        tokens[i] = val_distrib(rng);
    }

    auto ds = knowhere::GenDataSet(static_cast<int64_t>(total_tokens), dim, tokens);
    ds->SetIsOwner(true);
    const size_t* offsets_const = offsets.release();
    ds->Set(knowhere::meta::EMB_LIST_OFFSET, offsets_const);
    ds->Set(knowhere::meta::EMB_LIST_COUNT, num_docs);
    return ds;
}

}  // namespace

TEST_CASE("Test SVS GEM Build and Search", "[svs][gem]") {
    const int64_t num_docs = 500;
    const int64_t dim = 32;
    const int32_t topk = 10;

    auto version = GenTestEmbListVersionList();

    auto train_ds = GenGemEmbListDataSet(num_docs, dim, /*min_len=*/4, /*max_len=*/12, /*seed=*/42);
    // Query with the indexed docs' own tokens: each query doc should name itself as top-1.
    auto query_ds = GenGemEmbListDataSet(num_docs, dim, /*min_len=*/4, /*max_len=*/12, /*seed=*/42);

    knowhere::Json conf;
    conf[knowhere::meta::DIM] = dim;
    conf[knowhere::meta::METRIC_TYPE] = knowhere::metric::MAX_SIM_IP;
    conf[knowhere::meta::TOPK] = topk;
    conf[knowhere::indexparam::RETRIEVAL_ANN_RATIO] = 10.0f;
    conf["emb_list_strategy"] = "gem";
    conf["gem_num_centroids"] = 64;

    auto idx = knowhere::IndexFactory::Instance().Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_SVS_GEM, version);
    REQUIRE(idx.has_value());
    auto index = idx.value();

    REQUIRE(index.Type() == knowhere::IndexEnum::INDEX_SVS_GEM);
    REQUIRE(index.Build(train_ds, conf) == knowhere::Status::success);
    REQUIRE(index.Size() > 0);
    REQUIRE(index.Count() == num_docs);

    auto check_results = [&](const knowhere::DataSetPtr& result) {
        const auto* ids = result->GetIds();
        const auto* dists = result->GetDistance();
        int64_t self_match = 0;
        for (int64_t q = 0; q < num_docs; ++q) {
            float prev = std::numeric_limits<float>::max();
            for (int32_t j = 0; j < topk; ++j) {
                auto id = ids[q * topk + j];
                REQUIRE((id == -1 || (id >= 0 && id < num_docs)));
                if (id != -1) {
                    REQUIRE(dists[q * topk + j] <= prev);
                    prev = dists[q * topk + j];
                }
            }
            if (ids[q * topk] == q) {
                self_match++;
            }
        }
        // ANN retrieval is approximate; require a strong majority of queries to self-match.
        REQUIRE(self_match >= num_docs * 9 / 10);
    };

    SECTION("Build and search") {
        auto res = index.Search(query_ds, conf, nullptr);
        REQUIRE(res.has_value());
        check_results(res.value());
    }

    SECTION("Serialize/Deserialize round trip") {
        auto res1 = index.Search(query_ds, conf, nullptr);
        REQUIRE(res1.has_value());

        knowhere::BinarySet bs;
        REQUIRE(index.Serialize(bs) == knowhere::Status::success);

        auto idx2 =
            knowhere::IndexFactory::Instance().Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_SVS_GEM, version);
        REQUIRE(idx2.has_value());
        auto index2 = idx2.value();
        REQUIRE(index2.Deserialize(bs, conf) == knowhere::Status::success);

        auto res2 = index2.Search(query_ds, conf, nullptr);
        REQUIRE(res2.has_value());

        const auto rows = res1.value()->GetRows();
        const auto* ids1 = res1.value()->GetIds();
        const auto* ids2 = res2.value()->GetIds();
        const auto* dists1 = res1.value()->GetDistance();
        const auto* dists2 = res2.value()->GetDistance();
        for (int64_t i = 0; i < rows * topk; ++i) {
            REQUIRE(ids1[i] == ids2[i]);
            REQUIRE(dists1[i] == dists2[i]);
        }
    }

    SECTION("Bitset filtering excludes docs") {
        // Bit i set means doc i is excluded: filter out every even-numbered doc.
        std::vector<uint8_t> bitset_data((num_docs + 7) / 8, 0);
        for (int64_t i = 0; i < num_docs; i += 2) {
            bitset_data[i / 8] |= static_cast<uint8_t>(1u << (i % 8));
        }
        knowhere::BitsetView bitset(bitset_data.data(), num_docs);

        auto res = index.Search(query_ds, conf, bitset);
        REQUIRE(res.has_value());
        const auto* ids = res.value()->GetIds();
        for (int64_t q = 0; q < num_docs; ++q) {
            for (int32_t j = 0; j < topk; ++j) {
                auto id = ids[q * topk + j];
                // No filtered (even) doc id may appear in the results.
                REQUIRE((id == -1 || id % 2 != 0));
            }
        }
    }
}

#endif  // defined(KNOWHERE_WITH_SVS) && defined(FAISS_SVS_HAS_GEM)
