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

#include "gridBallQueryPlugin.h"

#include <cmath>
#include <cstring>
#include <iostream>

#include "common/print.h"
#include "common/plugin.h"
#include "common/common.h"
#include "common/hash.h"


namespace nvinfer1 {
namespace plugin {

using namespace nvinfer1;
using nvinfer1::plugin::GridBallQueryPlugin;
using nvinfer1::plugin::GridBallQueryPluginCreator;

namespace {
char const *const kBALL_QUERY_PLUGIN_VERSION{"1"};
char const *const kBALL_QUERY_PLUGIN_NAME{"GridBallQuery"};
size_t constexpr kSERIALIZATION_SIZE{1 * sizeof(float) + 1 * sizeof(int32_t)};
} // namespace

// Static class fields initialization
PluginFieldCollection GridBallQueryPluginCreator::mFC{};
std::vector<PluginField> GridBallQueryPluginCreator::mPluginAttributes;

GridBallQueryPlugin::GridBallQueryPlugin(int32_t samplePerBall, float radius)
        : mSamplePerBall(samplePerBall), mRadius(radius) {
}

GridBallQueryPlugin::GridBallQueryPlugin(void const *data, size_t length) {
    PLUGIN_ASSERT(data != nullptr);
    uint8_t const *d = reinterpret_cast<uint8_t const *>(data);
    auto const *a = d;
    mSamplePerBall = readFromBuffer<int32_t>(d);
    mRadius = readFromBuffer<float>(d);
    PLUGIN_ASSERT(d == a + length);
}

nvinfer1::IPluginV2DynamicExt *GridBallQueryPlugin::clone() const noexcept {
    try {
        auto *plugin = new GridBallQueryPlugin(mSamplePerBall, mRadius);
        plugin->setPluginNamespace(mNamespace.c_str());
        return plugin;
    }
    catch (std::exception const &e) {
        caughtError(e);
    }
    return nullptr;
}

void GridBallQueryPlugin::configurePlugin(nvinfer1::DynamicPluginTensorDesc const *in, int32_t nbInputs,
                                          nvinfer1::DynamicPluginTensorDesc const *out, int32_t nbOutputs) noexcept {
    try {
        PLUGIN_VALIDATE(in != nullptr);
        PLUGIN_VALIDATE(out != nullptr);
        PLUGIN_VALIDATE(nbInputs == 5);
        PLUGIN_VALIDATE(nbOutputs == 2);
    }
    catch (std::exception const &e) {
        caughtError(e);
    }
}

int32_t GridBallQueryPlugin::getNbOutputs() const noexcept {
    return 2;
}

nvinfer1::DimsExprs GridBallQueryPlugin::getOutputDimensions(int32_t outputIndex, nvinfer1::DimsExprs const *inputs,
                                                             int32_t nbInputs,
                                                             nvinfer1::IExprBuilder &exprBuilder) noexcept {
    try {
        PLUGIN_VALIDATE(outputIndex >= 0 && outputIndex < this->getNbOutputs());
        auto batchSize = inputs[0].d[0];
        auto querySize = inputs[1].d[1];
        if (outputIndex == 0) {
            // cnt[b,n]
            nvinfer1::DimsExprs dim0{};
            dim0.nbDims = 2;
            dim0.d[0] = batchSize;
            dim0.d[1] = querySize;
            return dim0;
        }
        if (outputIndex == 1) {
            // idx[b,n,s]
            nvinfer1::DimsExprs dim1{};
            dim1.nbDims = 3;
            dim1.d[0] = batchSize;
            dim1.d[1] = querySize;
            dim1.d[2] = exprBuilder.constant(mSamplePerBall);
            return dim1;
        }
    }
    catch (std::exception const &e) {
        caughtError(e);
    }
    return nvinfer1::DimsExprs{};
}

bool GridBallQueryPlugin::supportsFormatCombination(
        int32_t pos, nvinfer1::PluginTensorDesc const *inOut, int32_t nbInputs, int32_t nbOutputs) noexcept {
    try {
        PLUGIN_VALIDATE(inOut != nullptr);
        PLUGIN_VALIDATE(nbInputs == 2);
        PLUGIN_VALIDATE(nbOutputs == 2);
        PluginTensorDesc const &in = inOut[pos];
        switch (pos) {
            case 0:    // input: source_xyz float32[B,N,3]
            case 1: {  // input: query_xyz float32[B,M,3]
                return (in.type == nvinfer1::DataType::kFLOAT) && (in.format == TensorFormat::kLINEAR);
            }
            case 2:    // input: voxel_hash_table int32[B,T]
            case 3: {  // input: subset_ind_table int32[B,T]
                return (in.type == nvinfer1::DataType::kINT32) && (in.format == TensorFormat::kLINEAR);
            }
            case 4: {  // input: subset_ind_table float32[B,3*3]
                return (in.type == nvinfer1::DataType::kFLOAT) && (in.format == TensorFormat::kLINEAR);
            }
            case 5:    // output: cnt int32[B,M]
            case 6: {  // output: indices int32[B,M,S]
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

nvinfer1::DataType GridBallQueryPlugin::getOutputDataType(
        int32_t index, nvinfer1::DataType const *inputTypes, int32_t nbInputs) const noexcept {
    try {
        PLUGIN_VALIDATE(inputTypes != nullptr);

        if (index == 0) {
            return nvinfer1::DataType::kINT32;
        }
        if (index == 1) {
            return nvinfer1::DataType::kINT32;
        }
    }
    catch (std::exception const &e) {
        caughtError(e);
    }
    return nvinfer1::DataType{};
}


size_t GridBallQueryPlugin::getWorkspaceSize(nvinfer1::PluginTensorDesc const *inputs, int32_t nbInputs,
                                             nvinfer1::PluginTensorDesc const *outputs,
                                             int32_t nbOutputs) const noexcept {
    try {
        return 0U;
    }
    catch (std::exception const &e) {
        caughtError(e);
    }
    return 0U;
}


__global__
void grid_query_batch(const uint32_t N, const uint32_t M, const uint32_t S, const uint32_t T, const float search_radius,
                      const float3 *__restrict__ queries,
                      const float3 *__restrict__ sources,
                      const Voxel *__restrict__ voxels,
                      const uint32_t *__restrict__ hash_table,
                      const uint32_t *__restrict__ slots2queries,
                      uint32_t *__restrict__ indices,
                      uint32_t *__restrict__ nums) {

    uint32_t batch_idx = blockIdx.y;
    uint32_t pt_idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (pt_idx >= N)
        return;

    auto queries_batch = queries + M * batch_idx;
    auto hash_table_batch = hash_table + T * batch_idx;
    auto slots2queries_batch = slots2queries + T * batch_idx;

    auto nums_batch = nums + M * batch_idx;  // int[B,M]
    auto indices_batch = indices + M * S * batch_idx;  // long[B,M,S]

    auto source = sources[N * batch_idx + pt_idx], v = voxels[batch_idx].c;
    int s_grid_x = roundf(source.x / v.x);
    int s_grid_y = roundf(source.y / v.y);
    int s_grid_z = roundf(source.z / v.z);

    int step_x = ceilf(search_radius / v.x + 0.5f) - 1;
    int step_y = ceilf(search_radius / v.y + 0.5f) - 1;
    int step_z = ceilf(search_radius / v.z + 0.5f) - 1;

    auto r2 = search_radius * search_radius;

    uint32_t MAX = T - 1;
    for (int q_grid_z = s_grid_z - step_z; q_grid_z <= s_grid_z + step_z; ++q_grid_z) {
        for (int q_grid_y = s_grid_y - step_y; q_grid_y <= s_grid_y + step_y; ++q_grid_y) {
            for (int q_grid_x = s_grid_x - step_x; q_grid_x <= s_grid_x + step_x; ++q_grid_x) {
                uint32_t key = coord_hash_32(q_grid_x, q_grid_y, q_grid_z);
                uint32_t slot = key & MAX;
                while (true) {
                    if (hash_table_batch[slot] == key) {  // hit a non-empty neighbour voxel.
                        auto query_idx = slots2queries_batch[slot];
                        if (query_idx < M) {  // but this voxel isn't sampled since we have sampled enough points.
                            auto query = queries_batch[query_idx];
                            auto offset_x = source.x - query.x;
                            auto offset_y = source.y - query.y;
                            auto offset_z = source.z - query.z;
                            auto d2 = offset_x * offset_x + offset_y * offset_y + offset_z * offset_z;
                            if (d2 <= r2 and nums_batch[query_idx] <= S) {
                                auto neighbor_idx_of_q = atomicAdd(nums_batch + query_idx, 1);
                                if (neighbor_idx_of_q < S) {
                                    indices_batch[query_idx * S + neighbor_idx_of_q] = pt_idx;
                                }
                            }
                        }
                        break;
                    }
                    if (hash_table_batch[slot] == kEmpty) {  // empty neighbour voxel.
                        break;
                    }
                    slot = (slot + 1) & MAX;
                }
            }
        }
    }
}

__global__
void pad_indices(const uint32_t M, const uint32_t S,
                 uint32_t *__restrict__ indices,
                 uint32_t *__restrict__ nums) {
    uint32_t pt_idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (pt_idx >= M)
        return;

    uint32_t batch_idx = blockIdx.y;
    auto &num = nums[M * batch_idx + pt_idx];
    if (num > S) {
        num = S;
    } else {
        auto idx = indices + M * S * batch_idx + S * pt_idx;  // [B,M,S]
        if (num) {
            for (int l = 0; num < S; ++l, ++num) {
                idx[num] = idx[l];
            }
        }
    }
}

int32_t GridBallQueryPlugin::enqueue(nvinfer1::PluginTensorDesc const *inputDesc,
                                     nvinfer1::PluginTensorDesc const *outputDesc, void const *const *inputs,
                                     void *const *outputs, void *workspace,
                                     cudaStream_t stream) noexcept {
    try {
        const auto *srcPtr = static_cast<const float3 *>(inputs[0]);                // TRT-input 1  float32[B,N,3]
        const auto *queryPtr = static_cast<const float3 *>(inputs[1]);              // TRT-input 2  float32[B,M,3]
        const auto *voxelHashTablePtr = static_cast<const uint32_t *>(inputs[2]);  // TRT-input 3  int32[B,T]
        const auto *subsetIndTablePtr = static_cast<const uint32_t *>(inputs[3]);  // TRT-input 4  int32[B,T]
        const auto *voxelPtr = static_cast<const Voxel *>(inputs[4]);              // TRT-input 5  float32[B,3*3]

        auto *cntPtr = static_cast<uint32_t *>(outputs[0]);            // TRT-output 1 int32[B,M]
        auto *indPtr = static_cast<uint32_t *>(outputs[1]);            // TRT-output 2 int32[B,M,S]

        int32_t batchSize = inputDesc[0].dims.d[0];
        int32_t srcSize = inputDesc[0].dims.d[1];
        int32_t querySize = inputDesc[1].dims.d[1];
        int32_t tableSize = inputDesc[2].dims.d[1];

        PLUGIN_CUASSERT(cudaMemsetAsync(cntPtr, 0, batchSize * querySize * sizeof(int), stream));
        PLUGIN_CUASSERT(cudaMemsetAsync(indPtr, 0, batchSize * querySize * mSamplePerBall * sizeof(int), stream));

        auto blocks = BLOCKS2D(srcSize, batchSize), threads = THREADS();  // for each source point.
        // query: th
        grid_query_batch<<<blocks, threads, 0, stream>>>(
                srcSize, querySize, mSamplePerBall, tableSize, mRadius,
                queryPtr,
                srcPtr,
                voxelPtr,
                voxelHashTablePtr,
                subsetIndTablePtr,
                indPtr,
                cntPtr);

        // tail padding
        blocks = BLOCKS2D(querySize, batchSize);
        pad_indices<<<blocks, threads, 0, stream>>>(querySize, mSamplePerBall, indPtr, cntPtr);
        // print(cntPtr, {batchSize, querySize}, "GridBallQueryCounters");
        // print(indPtr, {batchSize, querySize, mSamplePerBall}, "GridBallQueryIndices");
        return 0;
    }
    catch (std::exception const &e) {
        caughtError(e);
    }
    return -1;
}

char const *GridBallQueryPlugin::getPluginType() const noexcept {
    return kBALL_QUERY_PLUGIN_NAME;
}

char const *GridBallQueryPlugin::getPluginVersion() const noexcept {
    return kBALL_QUERY_PLUGIN_VERSION;
}


int32_t GridBallQueryPlugin::initialize() noexcept {
    return 0;
}

void GridBallQueryPlugin::terminate() noexcept {}

size_t GridBallQueryPlugin::getSerializationSize() const noexcept {
    return kSERIALIZATION_SIZE;
}

void GridBallQueryPlugin::serialize(void *buffer) const noexcept {

    PLUGIN_ASSERT(buffer != nullptr);
    uint8_t *d = reinterpret_cast<uint8_t *>(buffer);
    auto *a = d;
    writeToBuffer<int32_t>(d, mSamplePerBall);
    writeToBuffer<float>(d, mRadius);
    PLUGIN_ASSERT(d == a + getSerializationSize());
}

void GridBallQueryPlugin::destroy() noexcept {
    delete this;
}

void GridBallQueryPlugin::setPluginNamespace(char const *libNamespace) noexcept {
    try {
        PLUGIN_VALIDATE(libNamespace != nullptr);
        mNamespace = libNamespace;
    }
    catch (std::exception const &e) {
        caughtError(e);
    }
}

char const *GridBallQueryPlugin::getPluginNamespace() const noexcept {
    return mNamespace.c_str();
}

GridBallQueryPluginCreator::GridBallQueryPluginCreator() {
    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(PluginField("nsample", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("radius", nullptr, PluginFieldType::kFLOAT32, 1));
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

char const *GridBallQueryPluginCreator::getPluginName() const noexcept {
    return kBALL_QUERY_PLUGIN_NAME;
}

char const *GridBallQueryPluginCreator::getPluginVersion() const noexcept {
    return kBALL_QUERY_PLUGIN_VERSION;
}

PluginFieldCollection const *GridBallQueryPluginCreator::getFieldNames() noexcept {
    return &mFC;
}

IPluginV2 *GridBallQueryPluginCreator::createPlugin(char const *name, PluginFieldCollection const *fc) noexcept {
    try {
        PLUGIN_VALIDATE(fc != nullptr);
        PluginField const *fields = fc->fields;
        int32_t nbFields = fc->nbFields;

        int32_t samplePerBall = 0;
        float radius = 0;

        for (int32_t i = 0; i < nbFields; ++i) {
            char const *attrName = fields[i].name;
            if (!strcmp(attrName, "nsample")) {
                auto const *d = static_cast<int32_t const *>(fields[i].data);
                samplePerBall = d[0];
            } else if (!strcmp(attrName, "radius")) {
                auto const *d = static_cast<float const *>(fields[i].data);
                radius = d[0];
            }
        }
        IPluginV2 *plugin = new GridBallQueryPlugin(samplePerBall, radius);
        return plugin;
    }
    catch (std::exception const &e) {
        caughtError(e);
    }
    return nullptr;
}

IPluginV2 *GridBallQueryPluginCreator::deserializePlugin(
        char const *name, void const *serialData, size_t serialLength) noexcept {
    try {
        return new GridBallQueryPlugin(serialData, serialLength);
    }
    catch (std::exception const &e) {
        caughtError(e);
    }
    return nullptr;
}

void GridBallQueryPluginCreator::setPluginNamespace(char const *libNamespace) noexcept {
    try {
        PLUGIN_VALIDATE(libNamespace != nullptr);
        mNamespace = libNamespace;
    }
    catch (std::exception const &e) {
        caughtError(e);
    }
}

char const *GridBallQueryPluginCreator::getPluginNamespace() const noexcept {
    return mNamespace.c_str();
}

REGISTER_TENSORRT_PLUGIN(GridBallQueryPluginCreator);
} // namespace plugin
} // namespace nvinfer1
