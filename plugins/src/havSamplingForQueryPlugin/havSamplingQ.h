#include <cuda.h>
#include <common/hash.h>

__global__
void voxel_update_kernel(uint32_t it, uint32_t batch_size, uint32_t sample_num,
                         float threshold, const uint32_t *__restrict__ sampled_num,
                         uint8_t *__restrict__ batch_mask,
                         Voxel *__restrict__ voxels, float3 init_voxel);

__global__
void valid_voxel_kernel(uint32_t N, uint32_t T,
                        const uint8_t *__restrict__ mask, const float3 *__restrict__ xyz,
                        const Voxel *__restrict__ voxel,
                        uint32_t *__restrict__ table, uint32_t *__restrict__ count);


__global__
void find_mini_dist_for_valid_voxels_batch(const uint32_t N, const uint32_t T,
                                           const float3 *__restrict__ xyz, const Voxel *__restrict__ voxel,
                                           uint32_t *__restrict__ key_table, float *__restrict__ dist_table,
                                           uint32_t *__restrict__ pts_slot, float *__restrict__ pts_dist);

__global__
void mask_input_if_with_min_dist_batch(const uint32_t N, const uint32_t T,
                                       const uint32_t *__restrict__ pts_slot,
                                       const float *__restrict__ pts_dist,
                                       float *__restrict__ dist_table,
                                       uint8_t *__restrict__ mask);

__global__
void mask_out_to_output_and_table_batch(const uint32_t N, const uint32_t M, const uint32_t T,
                                        const uint8_t *__restrict__ mask,
                                        const uint32_t *__restrict__ mask_sum,
                                        const uint32_t *__restrict__ pts_slot,
                                        uint32_t *__restrict__ ind,
                                        uint32_t *__restrict__ ind_table);