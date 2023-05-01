//
// Created by nrsl on 23-5-1.
//

#include "common/plugin.h"
#include "common/print.h"

#define DIVUP(m, n) ((m) / (n) + ((m) % (n) > 0))
#define TOTAL_THREADS 1024
#define THREADS_PER_BLOCK 256

__global__
void ball_query_cnt_kernel(int b, int n, int m, float radius, int nsample, const float *__restrict__ new_xyz,
                           const float *__restrict__ xyz, int *__restrict__ idx_cnt, int *__restrict__ idx) {
    // new_xyz: (B, M, 3)
    // xyz: (B, N, 3)
    // output:
    //      idx_cnt: (B, M)
    //      idx: (B, M, nsample)
    int bs_idx = blockIdx.y;
    int pt_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (bs_idx >= b || pt_idx >= m)
        return;

    new_xyz += bs_idx * m * 3 + pt_idx * 3;
    xyz += bs_idx * n * 3;
    idx += bs_idx * m * nsample + pt_idx * nsample;
    idx_cnt += bs_idx * m + pt_idx;

    float radius2 = radius * radius;
    float new_x = new_xyz[0];
    float new_y = new_xyz[1];
    float new_z = new_xyz[2];

    int cnt = 0;
    for (int k = 0; k < n; ++k) {
        float x = xyz[k * 3 + 0];
        float y = xyz[k * 3 + 1];
        float z = xyz[k * 3 + 2];
        float d2 = (new_x - x) * (new_x - x) + (new_y - y) * (new_y - y) + (new_z - z) * (new_z - z);
        if (d2 < radius2) {
            idx[cnt] = k;
            ++cnt;
            if (cnt >= nsample)
                break;
        }
    }
    idx_cnt[0] = cnt;
    for (int l = 0; cnt < nsample; ++l, ++cnt) {
        idx[cnt] = idx[l];
    }
}


void BallQueryLauncher(int batchSize, int pointSize, int querySize, float radius, int neighborSize,
                       const float *const xyzPtr, const float *const queryPtr,
                       int32_t *const cntPtr, int32_t *const indPtr,
                       cudaStream_t stream) {
    PLUGIN_CUASSERT(cudaMemsetAsync(cntPtr, 0x00, batchSize * querySize * sizeof(int), stream));
    PLUGIN_CUASSERT(cudaMemsetAsync(indPtr, 0x00, batchSize * querySize * neighborSize * sizeof(int), stream));
    dim3 blocks(DIVUP(querySize, THREADS_PER_BLOCK), batchSize);
    dim3 threads(THREADS_PER_BLOCK);
    ball_query_cnt_kernel<<<blocks, threads, 0, stream>>>(
            batchSize, pointSize, querySize,
            radius, neighborSize,
            queryPtr, xyzPtr, cntPtr, indPtr
    );
}