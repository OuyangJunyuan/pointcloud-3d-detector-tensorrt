#include <cfloat>
#include <cstdio>

#include <cuda.h>
#include <havSampling.h>


/**
 * @note shared version
 */
__global__ void voxel_update_kernel(uint32_t it, const uint32_t batch_size, const uint32_t sample_num,
                                    const float threshold, const uint32_t *__restrict__ sampled_num,
                                    uint8_t *__restrict__ batch_mask,
                                    Voxel *__restrict__ voxels, float3 init_voxel) {
    uint32_t batch_id = threadIdx.x;
    if (batch_id >= batch_size || batch_mask[batch_id])
        return;
    if (it == 1) {
        voxels[batch_id].c = init_voxel;
        voxels[batch_id].l = float3{0, 0, 0};
        voxels[batch_id].r = init_voxel + init_voxel;
        //        printf("%.2f,%.2f,%.2f\n", voxels[batch_id].c.x, voxels[batch_id].c.y, voxels[batch_id].c.z);
    } else {
        uint32_t num = sampled_num[batch_id];
        //        printf("%u\n", num);
        float upper_bound = sample_num * (1.0f + threshold), lower_bound = sample_num * 1.0f;
        if (upper_bound >= num and num >= lower_bound) {
            batch_mask[batch_id] = 1;
            return;
        }
        if (num > sample_num)
            voxels[batch_id].l = voxels[batch_id].c;
        if (num < sample_num)
            voxels[batch_id].r = voxels[batch_id].c;
        voxels[batch_id].c = (voxels[batch_id].l + voxels[batch_id].r) / 2;
        //        printf("%.2f,%.2f,%.2f\n", voxels[batch_id].c.x, voxels[batch_id].c.y, voxels[batch_id].c.z);
    }
}

/**
 * @note batch version
 */
__global__ void valid_voxel_kernel(const uint32_t B, const uint32_t N, const uint32_t T, const uint32_t MAX,
                                   const uint8_t *__restrict__ mask, const float3 *__restrict__ xyz,
                                   const Voxel *__restrict__ voxel,
                                   uint32_t *__restrict__ table, uint32_t *__restrict__ count) {
    uint32_t batch_id = blockIdx.y;
    uint32_t pts_id = blockDim.x * blockIdx.x + threadIdx.x;
    if (batch_id >= B || mask[batch_id] || pts_id >= N)
        return;
    auto table_this_batch = table + batch_id * T;
    auto pt_this_batch = batch_id * N + pts_id;
    auto pt = xyz[pt_this_batch], v = voxel[batch_id].c;

    uint32_t key = coord_hash_32(roundf(pt.x / v.x), roundf(pt.y / v.y), roundf(pt.z / v.z));
    uint32_t slot = key & MAX;
#ifdef DEBUG
    __shared__ int tid;
    int cnt = 0;
    tid = -1;
#endif
    while (true) {
        uint32_t prev = atomicCAS(table_this_batch + slot, kEmpty, key);
        if (prev == key) {
            return;
        }
        if (prev == kEmpty) {
            atomicAdd(count + batch_id, 1);
            return;
        }
        slot = (slot + 1) & MAX;
#ifdef DEBUG
        cnt++;
        if (cnt > 1000 and tid == -1)
        {
            atomicCAS(&tid, -1, pts_id);
        }
        if (tid == pts_id)
        {
            printf("%d,%d,%d,%x\n", MAX, pts_id, slot, prev);
        }
#endif
    }
}

/**
 * note batch version
 */
__global__ void find_mini_dist_for_valid_voxels_batch(const uint32_t N, const uint32_t T,
                                                      const float3 *__restrict__ xyz, const Voxel *__restrict__ voxel,
                                                      uint32_t *__restrict__ key_table,
                                                      float *__restrict__ dist_table, uint32_t *__restrict__ pts_slot,
                                                      float *__restrict__ pts_dist) {
    uint32_t batch_id = blockIdx.y;
    uint32_t pts_id = blockDim.x * blockIdx.x + threadIdx.x;
    if (pts_id >= N)
        return;
    const uint32_t MAX = T - 1;
    auto pt_this_batch = batch_id * N + pts_id;
    auto key_this_batch = key_table + batch_id * T;

    auto pt = xyz[pt_this_batch], v = voxel[batch_id].c;
    auto coord_x = roundf(pt.x / v.x);
    auto coord_y = roundf(pt.y / v.y);
    auto coord_z = roundf(pt.z / v.z);
    auto d1 = pt.x - coord_x * v.x;
    auto d2 = pt.y - coord_y * v.y;
    auto d3 = pt.z - coord_z * v.z;

    pts_dist[pt_this_batch] = d1 * d1 + d2 * d2 + d3 * d3;
    uint32_t key = coord_hash_32(coord_x, coord_y, coord_z);
    uint32_t slot = key & MAX;

    while (true) {
        uint32_t prev = atomicCAS(key_this_batch + slot, kEmpty, key);
        if (prev == key or prev == kEmpty) {
            atomicMin(dist_table + batch_id * T + slot, pts_dist[pt_this_batch]);
            pts_slot[pt_this_batch] = slot;
            return;
        }
        slot = (slot + 1) & MAX;
    }
}

