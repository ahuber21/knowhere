/*
 * Portions Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

/*
 * Portions Copyright 2025 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <faiss/Index.h>
#include <faiss/svs/IndexSVSFaissUtils.h>

#include <svs/runtime/api_defs.h>
#include <svs/runtime/gem_index.h>

#include <iostream>
#include <memory>

namespace faiss {

struct SearchParametersSVSGem : public SearchParameters {
    size_t search_window_size = 1000;
    size_t router_nprobe = 4;
};

// Indexes whole token sets, so the single-vector add()/search() of Index throw;
// build() and search_sets() are the entry points.
struct IndexSVSGem : Index {
    // build parameters
    size_t num_centroids = 0; // 0: auto
    size_t graph_max_degree = 24;
    size_t construction_window_size = 80;
    float alpha = 1.2f;
    size_t prune_to = 0;             // 0: R - 4
    size_t max_candidate_pool_size = 0; // 0: 3L
    int transport_start = 0;         // 0 NorthWest, 1 CheapestRow
    bool build_router = true;
    bool route_inserts = false;
    uint32_t seed = 42;
    size_t num_threads = 0; // 0: runtime default
    // Borrowed: must outlive build(); never serialized. When set, build()
    // runs on it and num_threads is ignored.
    svs_runtime::ThreadPoolInterface* build_thread_pool = nullptr;

    // default search parameters
    size_t search_window_size = 1000;
    size_t router_nprobe = 4;

    IndexSVSGem();
    explicit IndexSVSGem(idx_t d); // metric_type = METRIC_INNER_PRODUCT
    ~IndexSVSGem() override;

    // offsets has n_docs + 1 entries, offsets[0] == 0; tokens are d floats
    // each.
    void build(idx_t n_docs, const float* tokens, const size_t* offsets);

    // distances: qChamfer, smaller is closer. Missing results get label -1.
    // params->sel filters doc ids.
    void search_sets(
            idx_t n_queries,
            const float* query_tokens,
            const size_t* query_offsets,
            idx_t k,
            float* distances,
            idx_t* labels,
            const SearchParameters* params = nullptr) const;

    size_t trained_num_centroids() const; // K
    size_t memory_usage() const;

    // Throws: a Gem index is built once from whole emb_lists, not grown
    // vector by vector. Use build().
    void add(idx_t n, const float* x) override;

    // Throws: use search_sets(), which operates on token sets.
    void search(
            idx_t n,
            const float* x,
            idx_t k,
            float* distances,
            idx_t* labels,
            const SearchParameters* params = nullptr) const override;

    void reset() override;

    void serialize_impl(std::ostream& out) const;
    void deserialize_impl(std::istream& in);

   private:
    struct Deleter {
        void operator()(svs_runtime::GemIndex* p) const noexcept;
    };
    std::unique_ptr<svs_runtime::GemIndex, Deleter> impl;
};

} // namespace faiss
