/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */


#include "havSamplingPlugin.h"

#include <cfloat>
#include <cmath>
#include <cstring>
#include <iostream>

#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/reduce.h>

#include "havSampling.h"
#include "common/print.h"
#include "common/plugin.h"
#include "common/common.h"

#define cuda_assert(...)  assert(__VA_ARGS==cudaSuccess)
namespace nvinfer1::plugin {

int32_t HAVSamplingPlugin::enqueue(cudaStream_t stream) noexcept {
    static float float_max = FLT_MAX;
    // cuMemsetD32Async((CUdeviceptr) (ws.dist_table), *(unsigned int *) (&float_max), ws.elem.dist_table, stream);

//     // cuMemsetD2D32(,)
//     // Initialize workspace memory
//     uint32_t sampledIndSize = batchSize * mNbSamples * sizeof(uint32_t);
//     PLUGIN_CUASSERT(cudaMemsetAsync(batchMaskPtr, 0, batchMaskSize, stream));
//     PLUGIN_CUASSERT(cudaMemsetAsync(sampledIndPtr, 0, sampledIndSize, stream));
//     thrust::device_ptr<float> dev_dist_table_ptr(distTablePtr);
//     thrust::fill(
//             thrust::cuda::par.on(stream), dev_dist_table_ptr, dev_dist_table_ptr + batchSize * mTableSize, FLT_MAX);
//     // compute potential voxel size.
//     int iter = 0;
//     float3 voxel = {mVoxelX, mVoxelY, mVoxelZ};
//     thrust::device_ptr<uint8_t> dev_batch_mask_ptr(batchMaskPtr);
//     auto max = mTableSize - 1;
//     auto blocks = BLOCKS2D(mNbPoints, batchSize), threads = THREADS();
//     while (++iter <= mMaxIters
//            and (voxel_update_kernel<<<1, batchSize, 0, stream>>>(
//             iter, batchSize, mNbSamples, mTolerance, sampledNumPtr, batchMaskPtr, voxelPtr, voxel),
//             thrust::reduce(thrust::cuda::par.on(stream), dev_batch_mask_ptr, dev_batch_mask_ptr + batchSize,
//                            (int) 0, thrust::plus<bool>())
//             != batchSize)) {
//         PLUGIN_CUASSERT(cudaMemsetAsync(sampledNumPtr, 0, nbSampledSize, stream));
//         PLUGIN_CUASSERT(cudaMemsetAsync(hashTablePtr, 0xff, hashTableSize, stream));
// #ifdef DEBUG
//         cudaStreamSynchronize(stream);
//         printf("%d begin\n", iter);
//         printf("%d points\n", mNbPoints);
//         print((float *) voxelPtr, {batchSize, 9}, "HAVSampling::voxelPtr");
//         print((uint32_t *) hashTablePtr, {batchSize, (int) mTableSize}, "HAVSampling::hashTablePtr");
// #endif
//         valid_voxel_kernel<<<blocks, threads, 0, stream>>>(batchSize, mNbPoints, mTableSize, max, batchMaskPtr,
//                 pointCloudPtr, voxelPtr, hashTablePtr, sampledNumPtr);
// #ifdef DEBUG
//         cudaStreamSynchronize(stream);
//         printf("%d end\n", iter);
//         print(sampledNumPtr, {batchSize}, "HAVSampling::sampledNumPtr");
// #endif
//     }
//     PLUGIN_CUASSERT(cudaMemsetAsync(sampledNumPtr, 0, nbSampledSize, stream));
//     unique_mini_dist_kernel<<<blocks, threads, 0, stream>>>(batchSize, mNbPoints, mTableSize, max, pointCloudPtr,
//             voxelPtr, hashTablePtr, distTablePtr, pointSlotPtr,
//             pointDistPtr);
//
//     set_mask_kernel<<<blocks, threads, 0, stream>>>(batchSize, mNbPoints, mNbSamples, mTableSize, pointSlotPtr,
//             pointDistPtr, distTablePtr,
//             static_cast<uint32_t *>(sampledIndPtr), sampledNumPtr);
// #ifdef DEBUG
//     cudaStreamSynchronize(stream);
//     print(sampledIndPtr, {batchSize, mNbSamples}, "HAVSampling::SampledInd");
// #endif
    return 0;
}
} // namespace nvinfer1
