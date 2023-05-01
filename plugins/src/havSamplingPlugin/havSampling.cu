#include "hash.cuh"

#include <cfloat>
#include <cstdio>
#include <cassert>

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cub/device/device_scan.cuh>

// #define DEBUG
#ifdef DEBUG
#define dbg(STR, ...) printf(#__VA_ARGS__ ": " STR "\n" ,__VA_ARGS__)
#else
#define dbg(...)
#endif

__global__
void InitVoxels(const float3 init_voxel, float3 (*__restrict__ voxel_infos)[3]) {
    voxel_infos[threadIdx.x][0] = {.0f, .0f, .0f};
    voxel_infos[threadIdx.x][1] = init_voxel;
    voxel_infos[threadIdx.x][2] = init_voxel + init_voxel;
}

__global__
void InitHashTables(const uint32_t num_hash,
                    const uint32_t *__restrict__ batch_masks,
                    uint32_t *__restrict__ hash_tables) {
    if (batch_masks[blockIdx.y]) {
        return;
    }
    hash_tables[num_hash * blockIdx.y + blockDim.x * blockIdx.x + threadIdx.x] = kEmpty;
}

__global__
void CountNonEmptyVoxel(const uint32_t num_src, const uint32_t num_hash,
                        const uint32_t *__restrict__ batch_masks,
                        const float3 *__restrict__ sources, const float3 (*__restrict__ voxel_infos)[3],
                        uint32_t *__restrict__ hash_tables, uint32_t *__restrict__ num_sampled) {
    const uint32_t bid = blockIdx.y;
    const uint32_t pid = blockDim.x * blockIdx.x + threadIdx.x;
    if (batch_masks[bid] || pid >= num_src) {
        return;
    }

    const auto voxel = voxel_infos[bid][1];
    const auto point = sources[num_src * bid + pid];
    const auto table = hash_tables + num_hash * bid;

    const uint32_t hash_key = coord_hash_32((int) roundf(point.x / voxel.x),
                                            (int) roundf(point.y / voxel.y),
                                            (int) roundf(point.z / voxel.z));

    const uint32_t kHashMax = num_hash - 1;
    uint32_t hash_slot = hash_key & kHashMax;

    while (true) {
        const uint32_t old = atomicCAS(table + hash_slot, kEmpty, hash_key);
        if (old == hash_key) {
            return;
        }
        if (old == kEmpty) {
            atomicAdd(num_sampled + bid, 1);
            return;
        }
        hash_slot = (hash_slot + 1) & kHashMax;
    }
}

__global__
void UpdateVoxelsSizeIfNotConverge(const uint32_t num_batch, const uint32_t num_trg,
                                   const uint32_t lower_bound, const uint32_t upper_bound,
                                   uint32_t *__restrict__ batch_masks,
                                   float3 (*__restrict__ voxel_infos)[3],
                                   uint32_t *__restrict__ num_sampled) {
    uint32_t bid = threadIdx.x;
    if (batch_masks[bid])
        return;

    const auto num = num_sampled[bid];
    if (lower_bound <= num and num <= upper_bound) {   // fall into tolerance.
        batch_masks[bid] = 1;
        atomicAdd(&batch_masks[num_batch], 1);
        dbg("%d", num);
        dbg("%f %f %f", voxel_infos[bid][1].x, voxel_infos[bid][1].y, voxel_infos[bid][1].z);
    } else {  // has not converged yet.
        if (num > num_trg) {
            voxel_infos[bid][0] = voxel_infos[bid][1];
        }
        if (num < num_trg) {
            voxel_infos[bid][2] = voxel_infos[bid][1];
        }
        // update current voxel by the average of left and right voxels.
        voxel_infos[bid][1] = (voxel_infos[bid][0] + voxel_infos[bid][2]) / 2.0f;
        num_sampled[bid] = 0;
    }
}

__global__
void FindMiniDistToCenterForEachNonEmptyVoxels(const uint32_t num_src, const uint32_t num_hash,
                                               const float3 *__restrict__ sources,
                                               const float3 (*__restrict__ voxel_infos)[3],
                                               uint32_t *__restrict__ hash_tables,
                                               float *__restrict__ dist_tables,
                                               uint32_t *__restrict__ point_slots,
                                               float *__restrict__ point_dists) {
    const uint32_t bid = blockIdx.y;
    const uint32_t pid = blockDim.x * blockIdx.x + threadIdx.x;
    if (pid >= num_src) {
        return;
    }

    const auto pid_global = num_src * bid + pid;
    const auto point = sources[pid_global];
    const auto voxel = voxel_infos[bid][1];

    const auto coord_x = roundf(point.x / voxel.x);
    const auto coord_y = roundf(point.y / voxel.y);
    const auto coord_z = roundf(point.z / voxel.z);
    const auto d1 = point.x - coord_x * voxel.x;
    const auto d2 = point.y - coord_y * voxel.y;
    const auto d3 = point.z - coord_z * voxel.z;
    const auto noise = (float) pid * FLT_MIN;  // to ensure all point distances are different.
    const auto dist = d1 * d1 + d2 * d2 + d3 * d3 + noise;
    point_dists[pid_global] = dist;

    const auto dist_table = dist_tables + num_hash * bid;
    const auto hash_table = hash_tables + num_hash * bid;
    const uint32_t hash_key = coord_hash_32((int) coord_x, (int) coord_y, (int) coord_z);

    const uint32_t kHashMax = num_hash - 1;
    uint32_t hash_slot = hash_key & kHashMax;
    while (true) {
        const uint32_t old = atomicCAS(hash_table + hash_slot, kEmpty, hash_key);
        assert(old != kEmpty); // should never meet.
        if (old == hash_key) {
            atomicMin(dist_table + hash_slot, dist);
            point_slots[pid_global] = hash_slot;
            return;
        }
        hash_slot = (hash_slot + 1) & kHashMax;
    }
}

__global__
void MaskSourceWithMinimumDistanceToCenter(const uint32_t num_src, const uint32_t num_hash,
                                           const uint32_t *__restrict__ point_slots,
                                           const float *__restrict__ point_dists,
                                           const float *__restrict__ dist_tables,
                                           uint8_t *__restrict__ point_masks) {
    const uint32_t bid = blockIdx.y;
    const uint32_t pid = blockDim.x * blockIdx.x + threadIdx.x;
    if (pid >= num_src) {
        return;
    }

    const uint32_t pid_global = num_src * bid + pid;
    const auto min_dist = dist_tables[num_hash * bid + point_slots[pid_global]];
    point_masks[pid_global] = min_dist == point_dists[pid_global];
}

inline
void ExclusivePrefixSum(const uint32_t num_batch, const uint32_t num_src, const uint32_t num_hash,
                        void *temp_mem, uint8_t *point_masks, uint32_t *point_masks_sum, cudaStream_t stream) {
    size_t temp_mem_size = num_batch * num_hash;  // must be higher than expected.

    for (int bid = 0; bid < num_batch; ++bid) {
        cub::DeviceScan::ExclusiveSum(
                temp_mem, temp_mem_size,
                point_masks + bid * num_src,
                point_masks_sum + bid * num_src,
                num_src, stream
        );
    }
}

__global__
void MaskOutSubsetIndices(const uint32_t num_src, const uint32_t num_trg,
                          const uint8_t *__restrict__ point_masks,
                          const uint32_t *__restrict__ point_masks_sum,
                          uint32_t *__restrict__ sampled_ids) {
    const uint32_t bid = blockIdx.y;
    const uint32_t pid = blockDim.x * blockIdx.x + threadIdx.x;
    if (pid >= num_src) {
        return;
    }

    const auto pid_global = num_src * bid + pid;
    const auto mask_sum = point_masks_sum[pid_global];
    if (point_masks[pid_global] and mask_sum < num_trg) {
        sampled_ids[num_trg * bid + mask_sum] = pid;
    }
}

__global__
void MaskOutSubsetIndices(const uint32_t num_src, const uint32_t num_trg, const uint32_t num_hash,
                          const uint32_t *__restrict__ point_slots,
                          const uint8_t *__restrict__ point_masks,
                          const uint32_t *__restrict__ point_masks_sum,
                          uint32_t *__restrict__ sampled_ids, uint32_t *__restrict__ hash2subset) {
    const uint32_t bid = blockIdx.y;
    const uint32_t pid = blockDim.x * blockIdx.x + threadIdx.x;

    if (pid >= num_src) {
        return;
    }
    const auto pid_global = num_src * bid + pid;
    const auto mask_sum = point_masks_sum[pid_global];
    if (point_masks[pid_global] and mask_sum < num_trg) {
        sampled_ids[num_trg * bid + mask_sum] = pid;
        hash2subset[num_hash * bid + point_slots[pid_global]] = mask_sum;
    }
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
                              cudaStream_t stream = nullptr) {
    // static float flt_max = std::numeric_limits<float>::max();
    // cudaMemsetAsync(batch_masks, 0x00, (num_batch+1)*sizeof(*batch_masks), stream);
    // cudaMemsetAsync(num_sampled, 0x00, (num_batch)*sizeof(*num_sampled), stream);
    // cuMemsetD32Async((CUdeviceptr) dist_tables, *(uint32_t *) &flt_max, num_batch*num_hash, stream);
    InitVoxels<<<1, num_batch, 0, stream>>>(init_voxel, voxel_infos);

    const auto src_grid = BLOCKS2D(num_src, num_batch);
    const auto table_grid = BLOCKS2D(num_hash, num_batch);
    const auto block = THREADS();

    const auto lower_bound = uint32_t((float) num_trg * (1.0f + 0.0f));
    const auto upper_bound = uint32_t((float) num_trg * (1.0f + tolerance));

    uint32_t num_complete = 0;
    uint32_t cur_iteration = 1;
    while (max_iterations >= cur_iteration++ and num_complete != num_batch) {
        InitHashTables<<<table_grid, block, 0, stream>>>(
                num_hash, batch_masks, hash_tables
        );
        CountNonEmptyVoxel<<<src_grid, block, 0, stream>>>(
                num_src, num_hash, batch_masks, sources, voxel_infos, hash_tables, num_sampled
        );

        if (max_iterations >= cur_iteration) {  // voxels should not be updated in last iteration.
            UpdateVoxelsSizeIfNotConverge<<<1, num_batch, 0, stream>>>(
                    num_batch, num_trg, lower_bound, upper_bound, batch_masks, voxel_infos, num_sampled
            );
        }
        cudaMemcpyAsync(
                &num_complete, &batch_masks[num_batch],
                sizeof(num_complete), cudaMemcpyDeviceToHost, stream
        );
        cudaStreamSynchronize(stream);
    }
    FindMiniDistToCenterForEachNonEmptyVoxels<<< src_grid, block, 0, stream>>>(
            num_src, num_hash, sources, voxel_infos, hash_tables, dist_tables, point_slots, point_dists
    );
    MaskSourceWithMinimumDistanceToCenter<<<src_grid, block, 0, stream>>>(
            num_src, num_hash, point_slots, point_dists, dist_tables, point_masks
    );

    auto *temp_mem = (void *) dist_tables;  // reuse dist_tables as temporary memories.
    auto *point_masks_sum = (uint32_t *) point_dists;  // reuse point_dists as point_masks_sum
    ExclusivePrefixSum(
            num_batch, num_src, num_hash, temp_mem, point_masks, point_masks_sum, stream
    );
    if (return_hash2subset) {
        auto *hash2subset = (uint32_t *) dist_tables;  // reuse dist_tables as hash2subset.
        // cudaMemsetAsync(hash2subset, 0x00, num_batch * num_hash * sizeof(uint32_t), stream);
        MaskOutSubsetIndices<<<src_grid, block, 0, stream>>>(
                num_src, num_trg, num_hash, point_slots, point_masks, point_masks_sum, sampled_ids, hash2subset
        );
    } else {
        MaskOutSubsetIndices<<<src_grid, block, 0, stream>>>(
                num_src, num_trg, point_masks, point_masks_sum, sampled_ids
        );
    }
}