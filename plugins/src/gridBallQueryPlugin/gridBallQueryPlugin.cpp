//
// Created by nrsl on 23-5-2.
//
#include "gridBallQueryPlugin.h"

void QueryByPointHashingBatchLauncher(const uint32_t num_batch, const uint32_t num_src,
                                      const uint32_t num_qry, const uint32_t num_nei,
                                      const uint32_t num_hash, const float search_radius, const float3 (*voxels)[3],
                                      const uint32_t *hash_tables, const uint32_t *coord2query,
                                      const float3 *queries, const float3 *sources,
                                      uint32_t *queried_ids, uint32_t *num_queried,
                                      cudaStream_t stream = nullptr);

namespace nvinfer1::plugin {
int32_t GridBallQueryPlugin::initialize() noexcept { return 0; }

void GridBallQueryPlugin::terminate() noexcept {}

int32_t GridBallQueryPlugin::enqueue(cudaStream_t stream) noexcept {
    // queried_ids.zero_();
    // num_queried.zero_();

    cudaMemsetAsync(out.indices.ptr, 0x00, out.indices.bytes, stream);
    cudaMemsetAsync(out.num_valid.ptr, 0x00, out.num_valid.bytes, stream);

    QueryByPointHashingBatchLauncher(
            def.num_batch, def.num_source, def.num_query, attr.num_neighbor, def.num_hash,
            attr.radius,
            (float3 (*)[3]) in.voxel_infos.ptr,
            (uint32_t *) in.hash_tables.ptr,
            (uint32_t *) in.hash2subset.ptr,
            (float3 *) in.queries.ptr,
            (float3 *) in.sources.ptr,
            (uint32_t *) out.indices.ptr,
            (uint32_t *) out.num_valid.ptr,
            stream
    );
    return 0;
}
}