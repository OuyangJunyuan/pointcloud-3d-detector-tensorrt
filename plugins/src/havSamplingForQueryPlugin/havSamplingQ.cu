#include <cfloat>

#include <cuda.h>
#include <havSamplingQ.h>


/**
 * @note shared version
 */
__global__ void voxel_update_kernel(uint32_t it, const uint32_t batch_size, const uint32_t sample_num,
                                    const float threshold, const uint32_t *__restrict__ sampled_num,
                                    uint8_t *__restrict__ batch_mask,
                                    Voxel *__restrict__ voxels, float3 init_voxel);

/**
 * @note batch version
 */
__global__ void valid_voxel_kernel(const uint32_t N, const uint32_t T,
                                   const uint8_t *__restrict__ mask, const float3 *__restrict__ xyz,
                                   const Voxel *__restrict__ voxel,
                                   uint32_t *__restrict__ table, uint32_t *__restrict__ count) {
    uint32_t batch_id = blockIdx.y;
    uint32_t pts_id = blockDim.x * blockIdx.x + threadIdx.x;
    if (mask[batch_id] || pts_id >= N)
        return;

    const uint32_t MAX = T - 1;  // 0x00..ff..ff

    auto table_this_batch = table + T * batch_id;
    auto pt_this_batch = N * batch_id + pts_id;
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
__global__ void unique_mini_dist_kernel(const uint32_t B, const uint32_t N, const uint32_t T, const uint32_t MAX,
                                        const float3 *__restrict__ xyz, const Voxel *__restrict__ voxel,
                                        uint32_t *__restrict__ key_table,
                                        float *__restrict__ dist_table, uint32_t *__restrict__ pts_slot,
                                        float *__restrict__ pts_dist) {
    uint32_t batch_id = blockIdx.y;
    uint32_t pts_id = blockDim.x * blockIdx.x + threadIdx.x;
    if (batch_id >= B || pts_id >= N)
        return;

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
            //            printf("%f,%f\n", pts_dist[pt_this_batch], dist_table[batch_id * T + slot]);
            atomicMin(dist_table + batch_id * T + slot, pts_dist[pt_this_batch]);
            pts_slot[pt_this_batch] = slot;
            return;
        }
        slot = (slot + 1) & MAX;
    }
}

/**
 * @note batch version
 */
__global__ void set_mask_kernel(const uint32_t B, const uint32_t N, const uint32_t S, const uint32_t T,
                                const uint32_t *__restrict__ pts_slot, const float *__restrict__ pts_dist,
                                float *__restrict__ table,
                                uint32_t *__restrict__ ind, uint32_t *__restrict__ count) {
    uint32_t batch_id = blockIdx.y;
    uint32_t pts_id = blockDim.x * blockIdx.x + threadIdx.x;
    if (batch_id >= B || pts_id >= N)
        return;

    uint32_t pt_this_batch = batch_id * N + pts_id;
    auto *min_dist = table + batch_id * T + pts_slot[pt_this_batch];
    //    printf("%f,%f\n", pts_dist[pt_this_batch], table[batch_id * T + pts_slot[pt_this_batch]]);
    if (*min_dist == pts_dist[pt_this_batch]) {
        auto cnt = atomicAdd(count + batch_id, 1);
        if (cnt < S) {
            //            printf("%d,%d\n", count[batch_id], S);
            atomicExch(min_dist, MAXFLOAT);
            ind[batch_id * S + cnt] = pts_id;
        }
    }
}


/**
 * @note batch version
 */
__global__
void mask_input_if_with_min_dist_batch(const uint32_t N, const uint32_t T,
                                       const uint32_t *__restrict__ pts_slot,
                                       const float *__restrict__ pts_dist,
                                       float *__restrict__ dist_table,
                                       uint8_t *__restrict__ mask) {
    uint32_t pt_idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (pt_idx >= N)
        return;
    uint32_t batch_idx = blockIdx.y;
    uint32_t pt_idx_global = N * batch_idx + pt_idx;

    auto *min_dist_in_table = dist_table + T * batch_idx + pts_slot[pt_idx_global];
    if (*min_dist_in_table == pts_dist[pt_idx_global]) {  // this point with minimum distance to its voxel center.
        // remove the min_distance recorded in dist_table[pt_slot]
        // to guarantee only one point is masked in one voxel.
        // this can be carefully deleted as this rarely happens with nature data.
        float prev = atomicExch(min_dist_in_table, FLT_MAX);
        if (prev != FLT_MAX) {  // the first point with min distance in this voxel
            mask[pt_idx_global] = 1;
        }
    }
}

__global__
void mask_out_to_output_and_table_batch(const uint32_t N, const uint32_t M, const uint32_t T,
                                        const uint8_t *__restrict__ mask,
                                        const uint32_t *__restrict__ mask_sum,
                                        const uint32_t *__restrict__ pts_slot,
                                        uint32_t *__restrict__ ind,
                                        uint32_t *__restrict__ ind_table) {
    uint32_t pt_idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (pt_idx >= N) {
        return;
    }
    auto bt_idx = blockIdx.y;
    auto pt_idx_batch = N * bt_idx + pt_idx;
    auto output_ind = mask_sum[pt_idx_batch];
    if (mask[pt_idx_batch]) {
        if (output_ind < M) {
            ind[M * bt_idx + output_ind] = pt_idx;
        }
        // every active element in slot_table corresponding an index of output.
        ind_table[T * bt_idx + pts_slot[pt_idx_batch]] = output_ind;
    }
}

