#include <cmath>
#include <cinttypes>

#include <cuda.h>
#include <cuda_runtime_api.h>


inline auto get_table_size(int64_t nums, int64_t min = 2048) {
    auto table_size = nums * 2 > min ? nums * 2 : min;
    table_size = 2 << ((int64_t) ceilf((logf(static_cast<float>(table_size)) / logf(2.0f))) - 1);
    return table_size;
}

void HAVSamplingBatchLauncher(const int num_batch, const int num_src,
                              const int num_trg, const int num_hash,
                              const float3 init_voxel, const float tolerance, const int max_iterations,
                              const float3 *sources, uint32_t *batch_masks,
                              uint32_t *num_sampled, float3 (*voxel_infos)[3],
                              uint32_t *hash_tables, float *dist_tables,
                              uint32_t *point_slots, float *point_dists, uint8_t *point_masks,
                              uint32_t *sampled_ids,
                              const bool return_hash2subset = false,
                              cudaStream_t stream = nullptr);