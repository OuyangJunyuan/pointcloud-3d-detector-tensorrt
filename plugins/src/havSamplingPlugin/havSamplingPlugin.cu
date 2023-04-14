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

namespace nvinfer1 {
namespace plugin {

using namespace nvinfer1;
using nvinfer1::plugin::HAVSamplingPlugin;
using nvinfer1::plugin::HAVSamplingPluginCreator;

namespace {
char const *const kHAV_SAMPLING_PLUGIN_VERSION{"1"};
char const *const kHAV_SAMPLING_PLUGIN_NAME{"HAVSampling"};
size_t constexpr kSERIALIZATION_SIZE{4 * sizeof(float) + 2 * sizeof(int32_t)};
} // namespace

// Static class fields initialization
PluginFieldCollection HAVSamplingPluginCreator::mFC{};
std::vector<PluginField> HAVSamplingPluginCreator::mPluginAttributes;

HAVSamplingPlugin::HAVSamplingPlugin(
        int32_t nbSamples, float voxelX, float voxelY, float voxelZ, float tolerance, int32_t maxIters)
        : mNbSamples(nbSamples), mMaxIters(maxIters), mVoxelX(voxelX), mVoxelY(voxelY), mVoxelZ(voxelZ),
          mTolerance(tolerance) {
}

HAVSamplingPlugin::HAVSamplingPlugin(void const *data, size_t length) {
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

nvinfer1::IPluginV2DynamicExt *HAVSamplingPlugin::clone() const noexcept {
    try {
        auto *plugin = new HAVSamplingPlugin(mNbSamples, mVoxelX, mVoxelY, mVoxelZ, mTolerance, mMaxIters);
        plugin->setPluginNamespace(mNamespace.c_str());
        return plugin;
    }
    catch (std::exception const &e) {
        caughtError(e);
    }
    return nullptr;
}

nvinfer1::DimsExprs HAVSamplingPlugin::getOutputDimensions(int32_t outputIndex, nvinfer1::DimsExprs const *inputs,
                                                           int32_t nbInputs,
                                                           nvinfer1::IExprBuilder &exprBuilder) noexcept {

    try {
        PLUGIN_VALIDATE(outputIndex >= 0 && outputIndex < this->getNbOutputs());
        auto batchSize = inputs[0].d[0];
        if (outputIndex == 0) {
            // new_xyz[b,n]
            nvinfer1::DimsExprs dim0{};
            dim0.nbDims = 2;
            dim0.d[0] = batchSize;
            dim0.d[1] = exprBuilder.constant(mNbSamples);
            return dim0;
        }
    }
    catch (std::exception const &e) {
        caughtError(e);
    }
    return nvinfer1::DimsExprs{};
}

bool HAVSamplingPlugin::supportsFormatCombination(
        int32_t pos, nvinfer1::PluginTensorDesc const *inOut, int32_t nbInputs, int32_t nbOutputs) noexcept {
    try {
        PLUGIN_VALIDATE(inOut != nullptr);
        PLUGIN_VALIDATE(nbInputs == 1);
        PLUGIN_VALIDATE(nbOutputs == 1);
        PluginTensorDesc const &in = inOut[pos];
        if (pos == 0) // xyz
        {
            return (in.type == nvinfer1::DataType::kFLOAT) && (in.format == TensorFormat::kLINEAR);
        }
        if (pos == 1) // ind
        {
            return (in.type == nvinfer1::DataType::kINT32) && (in.format == TensorFormat::kLINEAR);
        }
        return false;
    }
    catch (std::exception const &e) {
        caughtError(e);
    }
    return false;
}

void HAVSamplingPlugin::configurePlugin(nvinfer1::DynamicPluginTensorDesc const *in, int32_t nbInputs,
                                        nvinfer1::DynamicPluginTensorDesc const *out, int32_t nbOutputs) noexcept {

    try {
        PLUGIN_VALIDATE(in != nullptr);
        PLUGIN_VALIDATE(out != nullptr);
        PLUGIN_VALIDATE(nbInputs == 1);
        PLUGIN_VALIDATE(nbOutputs == 1);
    }
    catch (std::exception const &e) {
        caughtError(e);
    }
}

size_t HAVSamplingPlugin::getWorkspaceSize(nvinfer1::PluginTensorDesc const *inputs, int32_t nbInputs,
                                           nvinfer1::PluginTensorDesc const *outputs,
                                           int32_t nbOutputs) const noexcept {
    try {
        int32_t batchSize = inputs[0].dims.d[0];
        auto mNbPoints = inputs[0].dims.d[1];
        auto mTableSize = get_table_size(mNbPoints, 2048);

        size_t batchMaskSize = batchSize * sizeof(uint8_t);
        size_t hashTableSize = batchSize * mTableSize * sizeof(uint32_t);
        size_t distTableSize = batchSize * mTableSize * sizeof(float);
        size_t pointSlotSize = batchSize * mNbPoints * sizeof(uint32_t);
        size_t pointDistSize = batchSize * mNbPoints * sizeof(float);
        size_t nbSampledSize = batchSize * sizeof(uint32_t);
        size_t voxelSize = batchSize * sizeof(Voxel);

        size_t workspaces[7];
        workspaces[0] = batchMaskSize;
        workspaces[1] = hashTableSize;
        workspaces[2] = distTableSize;
        workspaces[3] = pointSlotSize;
        workspaces[4] = pointDistSize;
        workspaces[5] = nbSampledSize;
        workspaces[6] = voxelSize;
        return calculateTotalWorkspaceSize(workspaces, 7);
    }
    catch (std::exception const &e) {
        caughtError(e);
    }
    return 0U;
}

int32_t HAVSamplingPlugin::enqueue(nvinfer1::PluginTensorDesc const *inputDesc,
                                   nvinfer1::PluginTensorDesc const *outputDesc, void const *const *inputs,
                                   void *const *outputs, void *workspace,
                                   cudaStream_t stream) noexcept {

    try {
        int32_t batchSize = inputDesc[0].dims.d[0];
        auto mNbPoints = inputDesc[0].dims.d[1];
        auto mTableSize = get_table_size(mNbPoints, 2048);


        // TRT-input
        const auto *pointCloudPtr = static_cast<const float3 *>(inputs[0]);
        // TRT-output
        auto *sampledIndPtr = static_cast<uint32_t *>(outputs[0]);
        // Temporary
        size_t batchMaskSize = batchSize * sizeof(uint8_t);
        size_t hashTableSize = batchSize * mTableSize * sizeof(uint32_t);
        size_t distTableSize = batchSize * mTableSize * sizeof(float);
        size_t pointSlotSize = batchSize * mNbPoints * sizeof(uint32_t);
        size_t pointDistSize = batchSize * mNbPoints * sizeof(float);
        size_t nbSampledSize = batchSize * sizeof(uint32_t);
        size_t voxelSize = batchSize * sizeof(Voxel);

        auto *batchMaskPtr = static_cast<uint8_t *>(workspace);
        auto *hashTablePtr
                = reinterpret_cast<uint32_t *>(nextWorkspacePtr(reinterpret_cast<int8_t *>(batchMaskPtr),
                                                                batchMaskSize));
        auto *distTablePtr
                = reinterpret_cast<float *>(nextWorkspacePtr(reinterpret_cast<int8_t *>(hashTablePtr), hashTableSize));
        auto *pointSlotPtr
                = reinterpret_cast<uint32_t *>(nextWorkspacePtr(reinterpret_cast<int8_t *>(distTablePtr),
                                                                distTableSize));
        auto *pointDistPtr
                = reinterpret_cast<float *>(nextWorkspacePtr(reinterpret_cast<int8_t *>(pointSlotPtr), pointSlotSize));
        auto *sampledNumPtr
                = reinterpret_cast<uint32_t *>(nextWorkspacePtr(reinterpret_cast<int8_t *>(pointDistPtr),
                                                                pointDistSize));
        auto *voxelPtr
                = reinterpret_cast<Voxel *>(nextWorkspacePtr(reinterpret_cast<int8_t *>(sampledNumPtr), nbSampledSize));

        // Initialize workspace memory
        uint32_t sampledIndSize = batchSize * mNbSamples * sizeof(uint32_t);
        PLUGIN_CUASSERT(cudaMemsetAsync(batchMaskPtr, 0, batchMaskSize, stream));
        PLUGIN_CUASSERT(cudaMemsetAsync(sampledIndPtr, 0, sampledIndSize, stream));
        thrust::device_ptr<float> dev_dist_table_ptr(distTablePtr);
        thrust::fill(
                thrust::cuda::par.on(stream), dev_dist_table_ptr, dev_dist_table_ptr + batchSize * mTableSize, FLT_MAX);
        // compute potential voxel size.
        int iter = 0;
        float3 voxel = {mVoxelX, mVoxelY, mVoxelZ};
        thrust::device_ptr<uint8_t> dev_batch_mask_ptr(batchMaskPtr);
        auto max = mTableSize - 1;
        auto blocks = BLOCKS2D(mNbPoints, batchSize), threads = THREADS();
        while (++iter <= mMaxIters
               and (voxel_update_kernel<<<1, batchSize, 0, stream>>>(
                iter, batchSize, mNbSamples, mTolerance, sampledNumPtr, batchMaskPtr, voxelPtr, voxel),
                thrust::reduce(thrust::cuda::par.on(stream), dev_batch_mask_ptr, dev_batch_mask_ptr + batchSize,
                               (int) 0, thrust::plus<bool>())
                != batchSize)) {
            PLUGIN_CUASSERT(cudaMemsetAsync(sampledNumPtr, 0, nbSampledSize, stream));
            PLUGIN_CUASSERT(cudaMemsetAsync(hashTablePtr, 0xff, hashTableSize, stream));
#ifdef DEBUG
            cudaStreamSynchronize(stream);
            printf("%d begin\n", iter);
            printf("%d points\n", mNbPoints);
            print((float *) voxelPtr, {batchSize, 9}, "HAVSampling::voxelPtr");
            print((uint32_t *) hashTablePtr, {batchSize, (int) mTableSize}, "HAVSampling::hashTablePtr");
#endif
            valid_voxel_kernel<<<blocks, threads, 0, stream>>>(batchSize, mNbPoints, mTableSize, max, batchMaskPtr,
                                                               pointCloudPtr, voxelPtr, hashTablePtr, sampledNumPtr);
#ifdef DEBUG
            cudaStreamSynchronize(stream);
            printf("%d end\n", iter);
            print(sampledNumPtr, {batchSize}, "HAVSampling::sampledNumPtr");
#endif
        }
        PLUGIN_CUASSERT(cudaMemsetAsync(sampledNumPtr, 0, nbSampledSize, stream));
        unique_mini_dist_kernel<<<blocks, threads, 0, stream>>>(batchSize, mNbPoints, mTableSize, max, pointCloudPtr,
                                                                voxelPtr, hashTablePtr, distTablePtr, pointSlotPtr,
                                                                pointDistPtr);

        set_mask_kernel<<<blocks, threads, 0, stream>>>(batchSize, mNbPoints, mNbSamples, mTableSize, pointSlotPtr,
                                                        pointDistPtr, distTablePtr,
                                                        static_cast<uint32_t *>(sampledIndPtr), sampledNumPtr);
#ifdef DEBUG
        cudaStreamSynchronize(stream);
        print(sampledIndPtr, {batchSize, mNbSamples}, "HAVSampling::SampledInd");
#endif
        return 0;
    }
    catch (std::exception const &e) {
        caughtError(e);
    }
    return -1;
}

nvinfer1::DataType HAVSamplingPlugin::getOutputDataType(
        int32_t index, nvinfer1::DataType const *inputTypes, int32_t nbInputs) const noexcept {
    try {
        PLUGIN_VALIDATE(inputTypes != nullptr);
        if (index == 0) {
            return nvinfer1::DataType::kINT32;
        }
    }
    catch (std::exception const &e) {
        caughtError(e);
    }
    return nvinfer1::DataType{};
}

char const *HAVSamplingPlugin::getPluginType() const noexcept {
    return kHAV_SAMPLING_PLUGIN_NAME;
}

char const *HAVSamplingPlugin::getPluginVersion() const noexcept {
    return kHAV_SAMPLING_PLUGIN_VERSION;
}

int32_t HAVSamplingPlugin::getNbOutputs() const noexcept {
    return 1;
}

int32_t HAVSamplingPlugin::initialize() noexcept {
    return 0;
}

void HAVSamplingPlugin::terminate() noexcept {
}

size_t HAVSamplingPlugin::getSerializationSize() const noexcept {
    return kSERIALIZATION_SIZE;
}

void HAVSamplingPlugin::serialize(void *buffer) const noexcept {

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

void HAVSamplingPlugin::destroy() noexcept {
    delete this;
}

void HAVSamplingPlugin::setPluginNamespace(char const *libNamespace) noexcept {
    try {
        PLUGIN_VALIDATE(libNamespace != nullptr);
        mNamespace = libNamespace;
    }
    catch (std::exception const &e) {
        caughtError(e);
    }
}

char const *HAVSamplingPlugin::getPluginNamespace() const noexcept {
    return mNamespace.c_str();
}

HAVSamplingPluginCreator::HAVSamplingPluginCreator() {
    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(PluginField("sample_num", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("voxel_size", nullptr, PluginFieldType::kFLOAT32, 1));
    mPluginAttributes.emplace_back(PluginField("tolerance", nullptr, PluginFieldType::kFLOAT32, 1));
    mPluginAttributes.emplace_back(PluginField("max_iter", nullptr, PluginFieldType::kINT32, 1));
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

char const *HAVSamplingPluginCreator::getPluginName() const noexcept {
    return kHAV_SAMPLING_PLUGIN_NAME;
}

char const *HAVSamplingPluginCreator::getPluginVersion() const noexcept {
    return kHAV_SAMPLING_PLUGIN_VERSION;
}

PluginFieldCollection const *HAVSamplingPluginCreator::getFieldNames() noexcept {
    return &mFC;
}

IPluginV2 *HAVSamplingPluginCreator::createPlugin(char const *name, PluginFieldCollection const *fc) noexcept {
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
                = new HAVSamplingPlugin(nbSamples, voxelSize[0], voxelSize[1], voxelSize[2], tolerance, maxIters);
        return plugin;
    }
    catch (std::exception const &e) {
        caughtError(e);
    }
    return nullptr;
}

IPluginV2 *HAVSamplingPluginCreator::deserializePlugin(
        char const *name, void const *serialData, size_t serialLength) noexcept {
    try {
        return new HAVSamplingPlugin(serialData, serialLength);
    }
    catch (std::exception const &e) {
        caughtError(e);
    }
    return nullptr;
}

void HAVSamplingPluginCreator::setPluginNamespace(char const *libNamespace) noexcept {
    try {
        PLUGIN_VALIDATE(libNamespace != nullptr);
        mNamespace = libNamespace;
    }
    catch (std::exception const &e) {
        caughtError(e);
    }
}

char const *HAVSamplingPluginCreator::getPluginNamespace() const noexcept {
    return mNamespace.c_str();
}

REGISTER_TENSORRT_PLUGIN(HAVSamplingPluginCreator);
} // namespace plugin
} // namespace nvinfer1
