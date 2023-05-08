//
// Created by nrsl on 23-5-1.
//

#include "havSampling.h"
#include "havSamplingReturnHashPlugin.h"

namespace nvinfer1::plugin {
int32_t HAVSamplingReturnHashPlugin::initialize() noexcept { return 0; }

void HAVSamplingReturnHashPlugin::terminate() noexcept {}

int32_t HAVSamplingReturnHashPlugin::enqueue(cudaStream_t stream) noexcept {
    static float flt_max = std::numeric_limits<float>::max();
    auto init_voxel = float3{attr.init_voxel[0], attr.init_voxel[1], attr.init_voxel[2]};

    cudaMemsetAsync(out.indices.ptr, 0x00, out.indices.bytes, stream);
    cudaMemsetAsync(ws.batch_masks.ptr, 0x00, ws.batch_masks.bytes, stream);
    cudaMemsetAsync(ws.num_sampled.ptr, 0x00, ws.num_sampled.bytes, stream);
    cuMemsetD32Async((CUdeviceptr) out.dist_tables.ptr, *(uint32_t *) &flt_max, out.dist_tables.elems, stream);

    HAVSamplingBatchLauncher(
            def.num_batch, def.num_source, def.num_sample, def.num_hash,
            init_voxel, attr.tolerance, attr.max_iteration,
            (float3 *) in.sources.ptr,
            ws.batch_masks.ptr,
            ws.num_sampled.ptr,
            (float3 (*)[3]) out.voxel_infos.ptr,
            out.hash_tables.ptr,
            out.dist_tables.ptr,
            ws.point_slots.ptr,
            ws.point_dists.ptr,
            ws.point_masks.ptr,
            out.indices.ptr,
            true,
            stream
    );

    // print(in.sources.ptr, {def.num_batch, def.num_source, 3}, "sources");
    // print(ws.point_dists.ptr, {def.num_batch, def.num_source, 1}, "point_dists");
    // print(out.indices.ptr, {def.num_batch, def.num_sample}, "indices");
    // print(out.voxel_infos.ptr, {def.num_batch, 9}, "voxel_size");
    return 0;

}
}