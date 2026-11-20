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

#include <faiss/svs/IndexSVSFaissUtils.h>
#include <faiss/svs/IndexSVSGem.h>

#include <faiss/impl/FaissAssert.h>

#include <svs/runtime/api_defs.h>
#include <svs/runtime/gem_index.h>

namespace faiss {
namespace {
svs_runtime::GemTransportStart to_svs_gem_transport_start(int start) {
    switch (start) {
        case 0:
            return svs_runtime::GemTransportStart::NorthWest;
        case 1:
            return svs_runtime::GemTransportStart::CheapestRow;
        default:
            FAISS_THROW_FMT(
                    "IndexSVSGem: invalid transport_start=%d (0: NorthWest, "
                    "1: CheapestRow)",
                    start);
    }
}
} // namespace

IndexSVSGem::IndexSVSGem() = default;

IndexSVSGem::IndexSVSGem(idx_t d) : Index(d, METRIC_INNER_PRODUCT) {}

IndexSVSGem::~IndexSVSGem() = default;

void IndexSVSGem::Deleter::operator()(svs_runtime::GemIndex* p) const noexcept {
    if (p) {
        auto status = svs_runtime::GemIndex::destroy(p);
        FAISS_ASSERT(status.ok());
    }
}

void IndexSVSGem::build(idx_t n_docs, const float* tokens, const size_t* offsets) {
    FAISS_THROW_IF_MSG(impl, "IndexSVSGem::build: index already built");
    FAISS_THROW_IF_NOT_MSG(
            n_docs >= 0, "IndexSVSGem::build: n_docs must be >= 0");

    auto build_params = svs_runtime::GemBuildParameters{
            .num_centroids = num_centroids,
            .graph_max_degree = graph_max_degree,
            .construction_window_size = construction_window_size,
            .alpha = alpha,
            .prune_to = prune_to,
            .max_candidate_pool_size = max_candidate_pool_size,
            .start = to_svs_gem_transport_start(transport_start),
            .build_router = build_router,
            .route_inserts = route_inserts,
            .seed = seed,
            .num_threads = num_threads,
    };
    auto default_search = svs_runtime::GemSearchParameters{
            .search_window_size = search_window_size,
            .router_nprobe = router_nprobe,
    };

    svs_runtime::GemIndex* raw = nullptr;
    auto status = build_thread_pool != nullptr
            ? svs_runtime::GemIndex::build(
                      &raw,
                      static_cast<size_t>(d),
                      static_cast<size_t>(n_docs),
                      tokens,
                      offsets,
                      *build_thread_pool,
                      build_params,
                      default_search)
            : svs_runtime::GemIndex::build(
                      &raw,
                      static_cast<size_t>(d),
                      static_cast<size_t>(n_docs),
                      tokens,
                      offsets,
                      build_params,
                      default_search);
    if (!status.ok()) {
        FAISS_THROW_MSG(status.message());
    }
    impl.reset(raw);
    ntotal = n_docs;
    is_trained = true;
}

void IndexSVSGem::search_sets(
        idx_t n_queries,
        const float* query_tokens,
        const size_t* query_offsets,
        idx_t k,
        float* distances,
        idx_t* labels,
        const SearchParameters* params) const {
    FAISS_THROW_IF_NOT_MSG(impl, "IndexSVSGem::search_sets: index not built");
    FAISS_THROW_IF_NOT(k > 0);

    auto sp = svs_runtime::GemSearchParameters{
            .search_window_size = search_window_size,
            .router_nprobe = router_nprobe,
    };
    if (auto gem_params = dynamic_cast<const SearchParametersSVSGem*>(params)) {
        sp.search_window_size = gem_params->search_window_size;
        sp.router_nprobe = gem_params->router_nprobe;
    }

    auto id_filter = make_faiss_id_filter(params);
    auto status = impl->search(
            static_cast<size_t>(n_queries),
            query_tokens,
            query_offsets,
            static_cast<size_t>(k),
            distances,
            convert_output_buffer<size_t>(
                    labels, static_cast<size_t>(n_queries * k)),
            &sp,
            id_filter.get());
    if (!status.ok()) {
        FAISS_THROW_MSG(status.message());
    }
}

size_t IndexSVSGem::trained_num_centroids() const {
    FAISS_THROW_IF_NOT_MSG(
            impl, "IndexSVSGem::trained_num_centroids: index not built");
    return impl->num_centroids();
}

size_t IndexSVSGem::memory_usage() const {
    return impl ? impl->get_memory_usage() : 0;
}

void IndexSVSGem::add(idx_t /* n */, const float* /* x */) {
    FAISS_THROW_MSG("IndexSVSGem::add: not supported, use build() instead");
}

void IndexSVSGem::search(
        idx_t /* n */,
        const float* /* x */,
        idx_t /* k */,
        float* /* distances */,
        idx_t* /* labels */,
        const SearchParameters* /* params */) const {
    FAISS_THROW_MSG(
            "IndexSVSGem::search: not supported, use search_sets() instead");
}

void IndexSVSGem::reset() {
    impl.reset();
    is_trained = false;
    ntotal = 0;
}

void IndexSVSGem::serialize_impl(std::ostream& out) const {
    FAISS_THROW_IF_NOT_MSG(
            impl, "IndexSVSGem::serialize_impl: index not built");
    auto status = impl->save(out);
    if (!status.ok()) {
        FAISS_THROW_MSG(status.message());
    }
}

void IndexSVSGem::deserialize_impl(std::istream& in) {
    FAISS_THROW_IF_MSG(
            impl, "IndexSVSGem::deserialize_impl: index already loaded");
    svs_runtime::GemIndex* raw = nullptr;
    auto status = svs_runtime::GemIndex::load(&raw, in);
    if (!status.ok()) {
        FAISS_THROW_MSG(status.message());
    }
    impl.reset(raw);
}

} // namespace faiss
