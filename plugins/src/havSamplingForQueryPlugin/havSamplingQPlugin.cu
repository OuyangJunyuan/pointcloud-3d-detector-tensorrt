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


#include "havSamplingQPlugin.h"

#include <cmath>
#include <cstring>
#include <iostream>

#include <cub/cub.cuh>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>

#include "havSamplingQ.h"
#include "common/print.h"
#include "common/plugin.h"
#include "common/common.h"


namespace nvinfer1 {
namespace plugin {

using namespace nvinfer1;
using nvinfer1::plugin::HAVSamplingPluginQ;
using nvinfer1::plugin::HAVSamplingPluginQCreator;

namespace {
char const *const kHAV_SAMPLING_PLUGIN_VERSION{"1"};
char const *const kHAV_SAMPLING_PLUGIN_NAME{"HAVSamplingQ"};
size_t constexpr kSERIALIZATION_SIZE{4 * sizeof(float) + 2 * sizeof(int32_t)};
} // namespace

// Static class fields initialization
PluginFieldCollection HAVSamplingPluginQCreator::mFC{};
std::vector<PluginField> HAVSamplingPluginQCreator::mPluginAttributes;

HAVSamplingPluginQ::HAVSamplingPluginQ(
        int32_t nbSamples, float voxelX, float voxelY, float voxelZ, float tolerance, int32_t maxIters)
        : mNbSamples(nbSamples), mMaxIters(maxIters), mVoxelX(voxelX), mVoxelY(voxelY), mVoxelZ(voxelZ),
          mTolerance(tolerance) {
}

HAVSamplingPluginQ::HAVSamplingPluginQ(void const *data, size_t length) {
    PLUGIN_ASSERT(data != nullptr);
    uint8_t const *d = reinterpret_cast<uint8_t const *>(data);
    auto const *a = d;
    mNbSamples = readFromBuffer<int32_t>(d);
    mVoxelX = readFromBuffer<float>(d);
    mVoxelY = readFromBuffer<float>(d);
    mVoxelZ = readFromBuffer<float>(d);
    mTolerance = readFromBuffer<float>(d);
    mMaxIters = readFromBuffer<int32_t>(d);
    PLUGIN_ASSERT(d == a + length);
}

nvinfer1::IPluginV2DynamicExt *HAVSamplingPluginQ::clone() const noexcept {
    try {
        auto *plugin = new HAVSamplingPluginQ(mNbSamples, mVoxelX, mVoxelY, mVoxelZ, mTolerance, mMaxIters);
        plugin->setPluginNamespace(mNamespace.c_str());
        return plugin;
    }
    catch (std::exception const &e) {
        caughtError(e);
    }
    return nullptr;
}

void HAVSamplingPluginQ::configurePlugin(nvinfer1::DynamicPluginTensorDesc const *in, int32_t nbInputs,
                                         nvinfer1::DynamicPluginTensorDesc const *out, int32_t nbOutputs) noexcept {

    try {
        PLUGIN_VALIDATE(in != nullptr);
        PLUGIN_VALIDATE(out != nullptr);
        PLUGIN_VALIDATE(nbInputs == 1);
        PLUGIN_VALIDATE(nbOutputs == 4);
    }
    catch (std::exception const &e) {
        caughtError(e);
    }
}

nvinfer1::DimsExprs HAVSamplingPluginQ::getOutputDimensions(int32_t outputIndex, nvinfer1::DimsExprs const *inputs,
                                                            int32_t nbInputs,
                                                            nvinfer1::IExprBuilder &exprBuilder) noexcept {
    try {
        PLUGIN_VALIDATE(outputIndex >= 0 && outputIndex < this->getNbOutputs());
        auto batchSize = inputs[0].d[0];
        nvinfer1::DimsExprs dim{};
        switch (outputIndex) {
            case 0: {  // indices_of_input int32[B,M]
                dim.nbDims = 2;
                dim.d[0] = batchSize;
                dim.d[1] = exprBuilder.constant(mNbSamples);
                return dim;
            }
            case 1: {  // adaptive_voxels float32[B,3]
                dim.nbDims = 2;
                dim.d[0] = batchSize;
                dim.d[1] = exprBuilder.constant(sizeof(Voxel) / sizeof(float));
                return dim;
            }
            case 2:   // voxel_hash_table int32[B,T]
            case 3: {  // subset_ind_table int32[B,T];
                auto srcPointNum = inputs[0].d[1]->getConstantValue();
                auto hashTableSize = get_table_size(srcPointNum, 2048);
                dim.nbDims = 2;
                dim.d[0] = batchSize;
                dim.d[1] = exprBuilder.constant(hashTableSize);
                return dim;
            }
            default: {
                assert(0);
            }
        }
    }
    catch (std::exception const &e) {
        caughtError(e);
    }
    return nvinfer1::DimsExprs{};
}

nvinfer1::DataType HAVSamplingPluginQ::getOutputDataType(
        int32_t index, nvinfer1::DataType const *inputTypes, int32_t nbInputs) const noexcept {
    try {
        PLUGIN_VALIDATE(inputTypes != nullptr);
        PLUGIN_VALIDATE(nbInputs != 1);

        switch (index) {
            case 0: {  // indices_of_input int32[B,M]
                return nvinfer1::DataType::kINT32;
            }
            case 1: {   // adaptive_voxels float32[B,3]
                return nvinfer1::DataType::kFLOAT;
            }
            case 2:  // voxel_hash_table int32[B,T]
            case 3: {  // subset_ind_table int32[B,T]
                return nvinfer1::DataType::kINT32;
            }
            default: {
                assert(0);
            }
        }
    }
    catch (std::exception const &e) {
        caughtError(e);
    }
    return nvinfer1::DataType{};
}

bool HAVSamplingPluginQ::supportsFormatCombination(
        int32_t pos, nvinfer1::PluginTensorDesc const *inOut, int32_t nbInputs, int32_t nbOutputs) noexcept {
    try {
        PLUGIN_VALIDATE(inOut != nullptr);
        PLUGIN_VALIDATE(nbInputs == 1);
        PLUGIN_VALIDATE(nbOutputs == 4);
        PluginTensorDesc const &in = inOut[pos];
        switch (pos) {
            case 0: {  // input: source_points float32[B,N,3]
                return (in.type == nvinfer1::DataType::kFLOAT) && (in.format == TensorFormat::kLINEAR);
            }
            case 1: {  // output: indices_of_input int32[B,M]
                return (in.type == nvinfer1::DataType::kINT32) && (in.format == TensorFormat::kLINEAR);
            }
            case 2: {  // output: adaptive_voxels float32[B,3*3]
                return (in.type == nvinfer1::DataType::kFLOAT) && (in.format == TensorFormat::kLINEAR);
            }
            case 3:    // output: voxel_hash_table int32[B,T]
            case 4: {  // output: subset_ind_table int32[B,T]
                return (in.type == nvinfer1::DataType::kINT32) && (in.format == TensorFormat::kLINEAR);
            }
            default: {
                return false;
            }
        }
    }
    catch (std::exception const &e) {
        caughtError(e);
    }
    return false;
}


size_t HAVSamplingPluginQ::getWorkspaceSize(nvinfer1::PluginTensorDesc const *inputs, int32_t nbInputs,
                                            nvinfer1::PluginTensorDesc const *outputs,
                                            int32_t nbOutputs) const noexcept {
    try {
        int32_t batchSize = inputs[0].dims.d[0];
        int32_t srcPointNum = inputs[0].dims.d[1];
        int32_t hashTableSize = get_table_size(srcPointNum, 2048);

        size_t batchMaskSize = batchSize * sizeof(uint8_t);
        size_t distTableSize = batchSize * hashTableSize * sizeof(float);
        size_t pointSlotSize = batchSize * srcPointNum * sizeof(uint32_t);
        size_t pointDistSize = batchSize * srcPointNum * sizeof(float);
        size_t sampledMaskSize = batchSize * srcPointNum * sizeof(uint8_t);
        size_t sampledMaskSumSize = batchSize * srcPointNum * sizeof(uint32_t);
        size_t nbSampledSize = batchSize * sizeof(uint32_t);

        size_t workspaces[7];
        workspaces[0] = batchMaskSize;
        workspaces[1] = distTableSize;
        workspaces[2] = pointSlotSize;
        workspaces[3] = pointDistSize;
        workspaces[4] = sampledMaskSize;
        workspaces[5] = sampledMaskSumSize;
        workspaces[6] = nbSampledSize;
        return calculateTotalWorkspaceSize(workspaces, 7);
    }
    catch (std::exception const &e) {
        caughtError(e);
    }
    return 0U;
}

int32_t HAVSamplingPluginQ::getNbOutputs() const noexcept {
    return 4;
}

int32_t HAVSamplingPluginQ::enqueue(nvinfer1::PluginTensorDesc const *inputDesc,
                                    nvinfer1::PluginTensorDesc const *outputDesc, void const *const *inputs,
                                    void *const *outputs, void *workspace,
                                    cudaStream_t stream) noexcept {
    try {
        int32_t batchSize = inputDesc[0].dims.d[0];
        int32_t srcPointNum = inputDesc[0].dims.d[1];
        int32_t tableSize = get_table_size(srcPointNum, 2048);

        // TRT-input
        const auto *pointCloudPtr = static_cast<const float3 *>(inputs[0]);
        // TRT-output
        auto *sampledIndPtr = static_cast<uint32_t *>(outputs[0]);
        auto *voxelPtr = static_cast<Voxel *>(outputs[1]);
        auto *hashTablePtr = static_cast<uint32_t *>(outputs[2]);
        auto *subsetIndTablePtr = static_cast<uint32_t *>(outputs[3]);

        // Temporary
        size_t batchMaskSize = batchSize * sizeof(uint8_t);
        size_t distTableSize = batchSize * tableSize * sizeof(float);
        size_t hashTableSize = batchSize * tableSize * sizeof(uint32_t);
        size_t pointSlotSize = batchSize * srcPointNum * sizeof(uint32_t);
        size_t pointDistSize = batchSize * srcPointNum * sizeof(float);
        size_t sampledMaskSize = batchSize * srcPointNum * sizeof(uint8_t);
        size_t sampledMaskSumSize = batchSize * srcPointNum * sizeof(uint32_t);
        size_t nbSampledSize = batchSize * sizeof(uint32_t);

        auto *batchMaskPtr
                = static_cast<uint8_t *>(workspace);
        auto *distTablePtr
                = reinterpret_cast<float *>(nextWorkspacePtr(reinterpret_cast<int8_t *>(batchMaskPtr), batchMaskSize));
        auto *pointSlotPtr
                = reinterpret_cast<uint32_t *>(nextWorkspacePtr(reinterpret_cast<int8_t *>(distTablePtr),
                                                                distTableSize));
        auto *pointDistPtr
                = reinterpret_cast<float *>(nextWorkspacePtr(reinterpret_cast<int8_t *>(pointSlotPtr), pointSlotSize));
        auto *sampledMaskPtr
                = reinterpret_cast<uint8_t *>( nextWorkspacePtr(reinterpret_cast<int8_t *>(pointDistPtr),
                                                                pointDistSize));
        auto *sampledMaskSumPtr
                = reinterpret_cast<uint32_t *>( nextWorkspacePtr(reinterpret_cast<int8_t *>(sampledMaskPtr),
                                                                 sampledMaskSize));
        auto *sampledNumPtr
                = reinterpret_cast<uint32_t *>(nextWorkspacePtr(reinterpret_cast<int8_t *>(sampledMaskSumPtr),
                                                                sampledMaskSumSize));
        // Initialize workspace memory
        uint32_t sampledIndSize = batchSize * mNbSamples * sizeof(uint32_t);
        PLUGIN_CUASSERT(cudaMemsetAsync(batchMaskPtr, 0, batchMaskSize, stream));
        PLUGIN_CUASSERT(cudaMemsetAsync(sampledMaskPtr, 0, sampledMaskSize, stream));
        PLUGIN_CUASSERT(cudaMemsetAsync(sampledIndPtr, 0, sampledIndSize, stream));
        thrust::device_ptr<float> dev_dist_table_ptr(distTablePtr);
        thrust::fill(thrust::cuda::par.on(stream),
                     dev_dist_table_ptr, dev_dist_table_ptr + batchSize * tableSize, FLT_MAX);

        // compute potential voxel size.
        int iter = 0;
        float3 init_voxel = {mVoxelX, mVoxelY, mVoxelZ};
        thrust::device_ptr<uint8_t> dev_batch_mask_ptr(batchMaskPtr);

        auto blocks = BLOCKS2D(srcPointNum, batchSize), threads = THREADS();
        while (++iter <= mMaxIters and
               (voxel_update_kernel<<<1, batchSize, 0, stream>>>(
                       iter, batchSize, mNbSamples, mTolerance, sampledNumPtr, batchMaskPtr, voxelPtr, init_voxel
               ), thrust::reduce(thrust::cuda::par.on(stream), dev_batch_mask_ptr, dev_batch_mask_ptr + batchSize,
                                 (int) 0, thrust::plus<uint8_t>()
               ) != batchSize)) {
            PLUGIN_CUASSERT(cudaMemsetAsync(sampledNumPtr, 0, nbSampledSize, stream));
            PLUGIN_CUASSERT(cudaMemsetAsync(hashTablePtr, 0xff, hashTableSize, stream));

            valid_voxel_kernel<<<blocks, threads, 0, stream>>>(srcPointNum, tableSize, batchMaskPtr,
                                                               pointCloudPtr, voxelPtr, hashTablePtr, sampledNumPtr);
            // fprintf(stderr, "==========\niter: %d\n", iter);
            // print((uint8_t *) batchMaskPtr, {batchSize}, "mask");
            // print((float *) voxelPtr, {batchSize, 3 * 3}, "voxel");
            // print(sampledNumPtr, {batchSize}, "nums");
        }
        find_mini_dist_for_valid_voxels_batch<<<blocks, threads, 0, stream>>>(srcPointNum, tableSize, pointCloudPtr,
                                                                              voxelPtr, hashTablePtr, distTablePtr,
                                                                              pointSlotPtr,
                                                                              pointDistPtr);

        // print(pointSlotPtr, {batchSize, srcPointNum}, "PointSlot");
        // print(pointDistPtr, {batchSize, srcPointNum}, "PointDist");
        // print(hashTablePtr, {batchSize, tableSize}, "HashTable");
        // print(distTablePtr, {batchSize, tableSize}, "DistTable");

        mask_input_if_with_min_dist_batch<<< blocks, threads, 0, stream>>>(
                srcPointNum, tableSize,
                pointSlotPtr,
                pointDistPtr,
                distTablePtr,
                sampledMaskPtr
        );
        // print(sampledMaskPtr, {batchSize, srcPointNum}, "SampledMask");
        // print(distTablePtr, {batchSize, tableSize}, "DistTable");

        auto *mask_ptr = sampledMaskPtr;
        auto *mask_sum_ptr = sampledMaskSumPtr;
        for (int b = 0; b < batchSize; ++b, mask_ptr += srcPointNum, mask_sum_ptr += srcPointNum) {
            thrust::exclusive_scan(thrust::cuda::par.on(stream), mask_ptr, mask_ptr + srcPointNum, mask_sum_ptr);
        }
        // print(sampledMaskSumPtr, {batchSize, srcPointNum}, "MaskPrefixSum");

        mask_out_to_output_and_table_batch<<<blocks, threads, 0, stream>>>(
                srcPointNum, mNbSamples, tableSize,
                sampledMaskPtr,
                sampledMaskSumPtr,
                pointSlotPtr,
                sampledIndPtr,
                subsetIndTablePtr);
        // print(sampledIndPtr, {batchSize, mNbSamples}, "SampledInd");
        // print(subsetIndTablePtr, {batchSize, tableSize}, "SubsetIndTable");

        return 0;
    }
    catch (std::exception const &e) {
        caughtError(e);
    }
    return -1;
}

char const *HAVSamplingPluginQ::getPluginType() const noexcept {
    return kHAV_SAMPLING_PLUGIN_NAME;
}

char const *HAVSamplingPluginQ::getPluginVersion() const noexcept {
    return kHAV_SAMPLING_PLUGIN_VERSION;
}


int32_t HAVSamplingPluginQ::initialize() noexcept {
    return 0;
}

void HAVSamplingPluginQ::terminate() noexcept {}

size_t HAVSamplingPluginQ::getSerializationSize() const noexcept {
    return kSERIALIZATION_SIZE;
}

void HAVSamplingPluginQ::serialize(void *buffer) const noexcept {

    PLUGIN_ASSERT(buffer != nullptr);
    uint8_t *d = reinterpret_cast<uint8_t *>(buffer);
    auto *a = d;
    writeToBuffer<int32_t>(d, mNbSamples);
    writeToBuffer<float>(d, mVoxelX);
    writeToBuffer<float>(d, mVoxelY);
    writeToBuffer<float>(d, mVoxelZ);
    writeToBuffer<float>(d, mTolerance);
    writeToBuffer<int32_t>(d, mMaxIters);
    PLUGIN_ASSERT(d == a + getSerializationSize());
}

void HAVSamplingPluginQ::destroy() noexcept {
    delete this;
}

void HAVSamplingPluginQ::setPluginNamespace(char const *libNamespace) noexcept {
    try {
        PLUGIN_VALIDATE(libNamespace != nullptr);
        mNamespace = libNamespace;
    }
    catch (std::exception const &e) {
        caughtError(e);
    }
}

char const *HAVSamplingPluginQ::getPluginNamespace() const noexcept {
    return mNamespace.c_str();
}

HAVSamplingPluginQCreator::HAVSamplingPluginQCreator() {
    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(PluginField("sample_num", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("voxel_size", nullptr, PluginFieldType::kFLOAT32, 1));
    mPluginAttributes.emplace_back(PluginField("tolerance", nullptr, PluginFieldType::kFLOAT32, 1));
    mPluginAttributes.emplace_back(PluginField("max_iter", nullptr, PluginFieldType::kINT32, 1));
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

char const *HAVSamplingPluginQCreator::getPluginName() const noexcept {
    return kHAV_SAMPLING_PLUGIN_NAME;
}

char const *HAVSamplingPluginQCreator::getPluginVersion() const noexcept {
    return kHAV_SAMPLING_PLUGIN_VERSION;
}

PluginFieldCollection const *HAVSamplingPluginQCreator::getFieldNames() noexcept {
    return &mFC;
}

IPluginV2 *HAVSamplingPluginQCreator::createPlugin(char const *name, PluginFieldCollection const *fc) noexcept {
    try {
        PLUGIN_VALIDATE(fc != nullptr);
        PluginField const *fields = fc->fields;
        int32_t nbFields = fc->nbFields;

        int32_t nbSamples = 0;
        float voxelSize[6]{};
        float tolerance = 0;
        int32_t maxIters = 0;

        for (int32_t i = 0; i < nbFields; ++i) {
            char const *attrName = fields[i].name;
            if (!strcmp(attrName, "sample_num")) {
                int32_t const *d = static_cast<int32_t const *>(fields[i].data);
                nbSamples = d[0];
            } else if (!strcmp(attrName, "voxel_size")) {
                float const *d = static_cast<float const *>(fields[i].data);
                voxelSize[0] = d[0];
                voxelSize[1] = d[1];
                voxelSize[2] = d[2];
            } else if (!strcmp(attrName, "tolerance")) {
                float const *d = static_cast<float const *>(fields[i].data);
                tolerance = d[0];
            } else if (!strcmp(attrName, "max_iter")) {
                int32_t const *d = static_cast<int32_t const *>(fields[i].data);
                maxIters = d[0];
            }
        }
        IPluginV2 *plugin
                = new HAVSamplingPluginQ(nbSamples, voxelSize[0], voxelSize[1], voxelSize[2], tolerance, maxIters);
        return plugin;
    }
    catch (std::exception const &e) {
        caughtError(e);
    }
    return nullptr;
}

IPluginV2 *HAVSamplingPluginQCreator::deserializePlugin(
        char const *name, void const *serialData, size_t serialLength) noexcept {
    try {
        return new HAVSamplingPluginQ(serialData, serialLength);
    }
    catch (std::exception const &e) {
        caughtError(e);
    }
    return nullptr;
}

void HAVSamplingPluginQCreator::setPluginNamespace(char const *libNamespace) noexcept {
    try {
        PLUGIN_VALIDATE(libNamespace != nullptr);
        mNamespace = libNamespace;
    }
    catch (std::exception const &e) {
        caughtError(e);
    }
}

char const *HAVSamplingPluginQCreator::getPluginNamespace() const noexcept {
    return mNamespace.c_str();
}

REGISTER_TENSORRT_PLUGIN(HAVSamplingPluginQCreator);
} // namespace plugin
} // namespace nvinfer1
