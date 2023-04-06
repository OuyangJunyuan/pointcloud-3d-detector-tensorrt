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

#include "ballQueryPlugin.h"

#include <cmath>
#include <cstring>
#include <iostream>

#include "common/print.h"
#include "common/plugin.h"
#include "common/common.h"


namespace nvinfer1
{
namespace plugin
{

using namespace nvinfer1;
using nvinfer1::plugin::BallQueryPlugin;
using nvinfer1::plugin::BallQueryPluginCreator;

namespace
{
char const* const kBALL_QUERY_PLUGIN_VERSION{"1"};
char const* const kBALL_QUERY_PLUGIN_NAME{"BallQuery"};
size_t constexpr kSERIALIZATION_SIZE{1 * sizeof(float) + 1 * sizeof(int32_t)};
} // namespace

// Static class fields initialization
PluginFieldCollection BallQueryPluginCreator::mFC{};
std::vector<PluginField> BallQueryPluginCreator::mPluginAttributes;

__global__ void ball_query_cnt_kernel(int b, int n, int m, float radius, int nsample, const float* __restrict__ new_xyz,
    const float* __restrict__ xyz, int* __restrict__ idx_cnt, int* __restrict__ idx)
{
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
    for (int k = 0; k < n; ++k)
    {
        float x = xyz[k * 3 + 0];
        float y = xyz[k * 3 + 1];
        float z = xyz[k * 3 + 2];
        float d2 = (new_x - x) * (new_x - x) + (new_y - y) * (new_y - y) + (new_z - z) * (new_z - z);
        if (d2 < radius2)
        {
            idx[cnt] = k;
            ++cnt;
            if (cnt >= nsample)
                break;
        }
    }
    idx_cnt[0] = cnt;
    for (int l = 0; cnt < nsample; ++l, ++cnt)
    {
        idx[cnt] = idx[l];
    }
}

BallQueryPlugin::BallQueryPlugin(int32_t samplePerBall, float radius)
    : mSamplePerBall(samplePerBall)
    , mRadius(radius)
{
}

BallQueryPlugin::BallQueryPlugin(void const* data, size_t length)
{
    PLUGIN_ASSERT(data != nullptr);
    uint8_t const* d = reinterpret_cast<uint8_t const*>(data);
    auto const* a = d;
    mSamplePerBall = readFromBuffer<int32_t>(d);
    mRadius = readFromBuffer<float>(d);
    PLUGIN_ASSERT(d == a + length);
}

nvinfer1::IPluginV2DynamicExt* BallQueryPlugin::clone() const noexcept
{
    try
    {
        auto* plugin = new BallQueryPlugin(mSamplePerBall, mRadius);
        plugin->setPluginNamespace(mNamespace.c_str());
        return plugin;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

nvinfer1::DimsExprs BallQueryPlugin::getOutputDimensions(int32_t outputIndex, nvinfer1::DimsExprs const* inputs,
    int32_t nbInputs, nvinfer1::IExprBuilder& exprBuilder) noexcept
{
    try
    {
        PLUGIN_VALIDATE(outputIndex >= 0 && outputIndex < this->getNbOutputs());
        auto batchSize = inputs[0].d[0];
        auto querySize = inputs[1].d[1];
        if (outputIndex == 0)
        {
            // cnt[b,n]
            nvinfer1::DimsExprs dim0{};
            dim0.nbDims = 2;
            dim0.d[0] = batchSize;
            dim0.d[1] = querySize;
            return dim0;
        }
        if (outputIndex == 1)
        {
            // idx[b,n,s]
            nvinfer1::DimsExprs dim1{};
            dim1.nbDims = 3;
            dim1.d[0] = batchSize;
            dim1.d[1] = querySize;
            dim1.d[2] = exprBuilder.constant(mSamplePerBall);
            return dim1;
        }
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nvinfer1::DimsExprs{};
}

bool BallQueryPlugin::supportsFormatCombination(
    int32_t pos, nvinfer1::PluginTensorDesc const* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept
{
    try
    {
        PLUGIN_VALIDATE(inOut != nullptr);
        PLUGIN_VALIDATE(nbInputs == 2);
        PLUGIN_VALIDATE(nbOutputs == 2);
        PluginTensorDesc const& in = inOut[pos];
        if (pos == 0) // input: xyz
        {
            return (in.type == nvinfer1::DataType::kFLOAT) && (in.format == TensorFormat::kLINEAR);
        }
        if (pos == 1) // input: new xyz
        {
            return (in.type == nvinfer1::DataType::kFLOAT) && (in.format == TensorFormat::kLINEAR);
        }
        if (pos == 2) // output: cnt
        {
            return (in.type == nvinfer1::DataType::kINT32) && (in.format == TensorFormat::kLINEAR);
        }
        if (pos == 3) // output: ind
        {
            return (in.type == nvinfer1::DataType::kINT32) && (in.format == TensorFormat::kLINEAR);
        }
        return false;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return false;
}

void BallQueryPlugin::configurePlugin(nvinfer1::DynamicPluginTensorDesc const* in, int32_t nbInputs,
    nvinfer1::DynamicPluginTensorDesc const* out, int32_t nbOutputs) noexcept
{
    try
    {
        PLUGIN_VALIDATE(in != nullptr);
        PLUGIN_VALIDATE(out != nullptr);
        PLUGIN_VALIDATE(nbInputs == 2);
        PLUGIN_VALIDATE(nbOutputs == 2);
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
}

size_t BallQueryPlugin::getWorkspaceSize(nvinfer1::PluginTensorDesc const* inputs, int32_t nbInputs,
    nvinfer1::PluginTensorDesc const* outputs, int32_t nbOutputs) const noexcept
{
    try
    {
        return 0U;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return 0U;
}

int32_t BallQueryPlugin::enqueue(nvinfer1::PluginTensorDesc const* inputDesc,
    nvinfer1::PluginTensorDesc const* outputDesc, void const* const* inputs, void* const* outputs, void* workspace,
    cudaStream_t stream) noexcept
{
    try
    {
        const auto* xyzPtr = static_cast<const float*>(inputs[0]);   // TRT-input 1
        const auto* queryPtr = static_cast<const float*>(inputs[1]); // TRT-input 2
        auto* cntPtr = static_cast<int32_t*>(outputs[0]);            // TRT-output 1
        auto* indPtr = static_cast<int32_t*>(outputs[1]);            // TRT-output 2

        int32_t batchSize = inputDesc[0].dims.d[0];
        int32_t pointSize = inputDesc[0].dims.d[1];
        int32_t querySize = inputDesc[1].dims.d[1];

//        print(const_cast<float*>(xyzPtr), {batchSize, pointSize, 3}, "BallQuery::xyzPtr");
        PLUGIN_CUASSERT(cudaMemsetAsync(cntPtr, 0, batchSize * querySize * sizeof(int), stream));
        PLUGIN_CUASSERT(cudaMemsetAsync(indPtr, 0, batchSize * querySize * mSamplePerBall * sizeof(int), stream));
        dim3 blocks(DIVUP(querySize, THREADS_PER_BLOCK), batchSize);
        dim3 threads(THREADS_PER_BLOCK);
        //        print_device<float>(xyzPtr, {batchSize, querySize, 3});
        ball_query_cnt_kernel<<<blocks, threads, 0, stream>>>(
            batchSize, pointSize, querySize, mRadius, mSamplePerBall, queryPtr, xyzPtr, cntPtr, indPtr);
//        print(indPtr, {batchSize, querySize, mSamplePerBall}, "BallQuery::IndPtr");
        return 0;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return -1;
}

nvinfer1::DataType BallQueryPlugin::getOutputDataType(
    int32_t index, nvinfer1::DataType const* inputTypes, int32_t nbInputs) const noexcept
{
    try
    {
        PLUGIN_VALIDATE(inputTypes != nullptr);
        if (index == 0)
        {
            return nvinfer1::DataType::kINT32;
        }
        if (index == 1)
        {
            return nvinfer1::DataType::kINT32;
        }
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nvinfer1::DataType{};
}

char const* BallQueryPlugin::getPluginType() const noexcept
{
    return kBALL_QUERY_PLUGIN_NAME;
}

char const* BallQueryPlugin::getPluginVersion() const noexcept
{
    return kBALL_QUERY_PLUGIN_VERSION;
}

int32_t BallQueryPlugin::getNbOutputs() const noexcept
{
    return 2;
}

int32_t BallQueryPlugin::initialize() noexcept
{
    return 0;
}

void BallQueryPlugin::terminate() noexcept {}

size_t BallQueryPlugin::getSerializationSize() const noexcept
{
    return kSERIALIZATION_SIZE;
}

void BallQueryPlugin::serialize(void* buffer) const noexcept
{

    PLUGIN_ASSERT(buffer != nullptr);
    uint8_t* d = reinterpret_cast<uint8_t*>(buffer);
    auto* a = d;
    writeToBuffer<int32_t>(d, mSamplePerBall);
    writeToBuffer<float>(d, mRadius);
    PLUGIN_ASSERT(d == a + getSerializationSize());
}

void BallQueryPlugin::destroy() noexcept
{
    delete this;
}

void BallQueryPlugin::setPluginNamespace(char const* libNamespace) noexcept
{
    try
    {
        PLUGIN_VALIDATE(libNamespace != nullptr);
        mNamespace = libNamespace;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
}

char const* BallQueryPlugin::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

BallQueryPluginCreator::BallQueryPluginCreator()
{
    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(PluginField("nsample", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("radius", nullptr, PluginFieldType::kFLOAT32, 1));
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

char const* BallQueryPluginCreator::getPluginName() const noexcept
{
    return kBALL_QUERY_PLUGIN_NAME;
}

char const* BallQueryPluginCreator::getPluginVersion() const noexcept
{
    return kBALL_QUERY_PLUGIN_VERSION;
}

PluginFieldCollection const* BallQueryPluginCreator::getFieldNames() noexcept
{
    return &mFC;
}

IPluginV2* BallQueryPluginCreator::createPlugin(char const* name, PluginFieldCollection const* fc) noexcept
{
    try
    {
        PLUGIN_VALIDATE(fc != nullptr);
        PluginField const* fields = fc->fields;
        int32_t nbFields = fc->nbFields;

        int32_t samplePerBall = 0;
        float radius = 0;

        for (int32_t i = 0; i < nbFields; ++i)
        {
            char const* attrName = fields[i].name;
            if (!strcmp(attrName, "nsample"))
            {
                auto const* d = static_cast<int32_t const*>(fields[i].data);
                samplePerBall = d[0];
            }
            else if (!strcmp(attrName, "radius"))
            {
                auto const* d = static_cast<float const*>(fields[i].data);
                radius = d[0];
            }
        }
        IPluginV2* plugin = new BallQueryPlugin(samplePerBall, radius);
        return plugin;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

IPluginV2* BallQueryPluginCreator::deserializePlugin(
    char const* name, void const* serialData, size_t serialLength) noexcept
{
    try
    {
        return new BallQueryPlugin(serialData, serialLength);
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

void BallQueryPluginCreator::setPluginNamespace(char const* libNamespace) noexcept
{
    try
    {
        PLUGIN_VALIDATE(libNamespace != nullptr);
        mNamespace = libNamespace;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
}

char const* BallQueryPluginCreator::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}
REGISTER_TENSORRT_PLUGIN(BallQueryPluginCreator);
} // namespace plugin
} // namespace nvinfer1
