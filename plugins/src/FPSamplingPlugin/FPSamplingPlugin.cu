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

#include "FPSamplingPlugin.h"

#include <cmath>
#include <cstring>
#include <iostream>
#include <thrust/device_vector.h>

#include "havSampling.h"
#include "common/print.h"
#include "common/plugin.h"
#include "common/common.h"
namespace nvinfer1
{
namespace plugin
{

using namespace nvinfer1;
using nvinfer1::plugin::FPSamplingPlugin;
using nvinfer1::plugin::FPSamplingPluginCreator;

namespace
{
char const* const kHAV_SAMPLING_PLUGIN_VERSION{"1"};
char const* const kHAV_SAMPLING_PLUGIN_NAME{"FPSampling"};
size_t constexpr kSERIALIZATION_SIZE{1 * sizeof(int32_t)};
} // namespace

// Static class fields initialization
PluginFieldCollection FPSamplingPluginCreator::mFC{};
std::vector<PluginField> FPSamplingPluginCreator::mPluginAttributes;

inline auto get_table_size(size_t N, size_t min_size)
{
    size_t table_size = std::max(min_size, N * 2);
    table_size = (2 << ((size_t) ceil((log((double) table_size) / log(2.0))) - 1));
    return table_size;
}

FPSamplingPlugin::FPSamplingPlugin(int32_t nbSamples)
    : mNbSamples(nbSamples)
{
}

FPSamplingPlugin::FPSamplingPlugin(void const* data, size_t length)
{
    PLUGIN_ASSERT(data != nullptr);
    uint8_t const* d = reinterpret_cast<uint8_t const*>(data);
    auto const* a = d;
    mNbSamples = readFromBuffer<int32_t>(d);
    PLUGIN_ASSERT(d == a + length);
}

nvinfer1::IPluginV2DynamicExt* FPSamplingPlugin::clone() const noexcept
{
    try
    {
        auto* plugin = new FPSamplingPlugin(mNbSamples);
        plugin->setPluginNamespace(mNamespace.c_str());
        return plugin;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

nvinfer1::DimsExprs FPSamplingPlugin::getOutputDimensions(int32_t outputIndex, nvinfer1::DimsExprs const* inputs,
    int32_t nbInputs, nvinfer1::IExprBuilder& exprBuilder) noexcept
{
    try
    {
        PLUGIN_VALIDATE(outputIndex >= 0 && outputIndex < this->getNbOutputs());
        auto batchSize = inputs[0].d[0];
        if (outputIndex == 0)
        {
            // new_xyz[b,n]
            nvinfer1::DimsExprs dim0{};
            dim0.nbDims = 2;
            dim0.d[0] = batchSize;
            dim0.d[1] = exprBuilder.constant(mNbSamples);
            return dim0;
        }
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nvinfer1::DimsExprs{};
}

bool FPSamplingPlugin::supportsFormatCombination(
    int32_t pos, nvinfer1::PluginTensorDesc const* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept
{
    try
    {
        PLUGIN_VALIDATE(inOut != nullptr);
        PLUGIN_VALIDATE(nbInputs == 1);
        PLUGIN_VALIDATE(nbOutputs == 1);
        PluginTensorDesc const& in = inOut[pos];
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
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return false;
}

void FPSamplingPlugin::configurePlugin(nvinfer1::DynamicPluginTensorDesc const* in, int32_t nbInputs,
    nvinfer1::DynamicPluginTensorDesc const* out, int32_t nbOutputs) noexcept
{
    try
    {
        PLUGIN_VALIDATE(in != nullptr);
        PLUGIN_VALIDATE(out != nullptr);
        PLUGIN_VALIDATE(nbInputs == 1);
        PLUGIN_VALIDATE(nbOutputs == 1);
        mNbPoints = in[0].desc.dims.d[1];
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
}

size_t FPSamplingPlugin::getWorkspaceSize(nvinfer1::PluginTensorDesc const* inputs, int32_t nbInputs,
    nvinfer1::PluginTensorDesc const* outputs, int32_t nbOutputs) const noexcept
{
    try
    {
        int32_t batchSize = inputs[0].dims.d[0];
        size_t pointDistSize = batchSize * mNbPoints * sizeof(float);

        size_t workspaces[1];
        workspaces[0] = pointDistSize;
        return calculateTotalWorkspaceSize(workspaces, 1);
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return 0U;
}


inline int opt_n_threads(int work_size)
{
    const int pow_2 = std::log(static_cast<double>(work_size)) / std::log(2.0);

    return std::max(std::min(1 << pow_2, TOTAL_THREADS), 1);
}
__device__ void __update(float *__restrict__ dists, int *__restrict__ dists_i, int idx1, int idx2) {
    const float v1 = dists[idx1], v2 = dists[idx2];
    const int i1 = dists_i[idx1], i2 = dists_i[idx2];
    dists[idx1] = max(v1, v2);
    dists_i[idx1] = v2 > v1 ? i2 : i1;
}
template<unsigned int block_size>
__global__ void farthest_point_sampling_kernel(int b, int n, int m,
    const float *__restrict__ dataset, float *__restrict__ temp,
    int *__restrict__ idxs) {
    // dataset: (B, N, 3)
    // tmp: (B, N)
    // output:
    //      idx: (B, M)

    if (m <= 0) return;
    __shared__ float dists[block_size];
    __shared__ int dists_i[block_size];

    int batch_index = blockIdx.x;
    dataset += batch_index * n * 3;
    temp += batch_index * n;
    idxs += batch_index * m;

    int tid = threadIdx.x;
    const int stride = block_size;

    int old = 0;
    if (threadIdx.x == 0)
        idxs[0] = old;

    __syncthreads();
    for (int j = 1; j < m; j++) {
        int besti = 0;
        float best = -1;
        float x1 = dataset[old * 3 + 0];
        float y1 = dataset[old * 3 + 1];
        float z1 = dataset[old * 3 + 2];
        for (int k = tid; k < n; k += stride) {
            float x2, y2, z2;
            x2 = dataset[k * 3 + 0];
            y2 = dataset[k * 3 + 1];
            z2 = dataset[k * 3 + 2];
            // float mag = (x2 * x2) + (y2 * y2) + (z2 * z2);
            // if (mag <= 1e-3)
            // continue;

            float d = (x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1) + (z2 - z1) * (z2 - z1);
            float d2 = min(d, temp[k]);
            temp[k] = d2;
            besti = d2 > best ? k : besti;
            best = d2 > best ? d2 : best;
        }
        dists[tid] = best;
        dists_i[tid] = besti;
        __syncthreads();

        if (block_size >= 1024) {
            if (tid < 512) {
                __update(dists, dists_i, tid, tid + 512);
            }
            __syncthreads();
        }

        if (block_size >= 512) {
            if (tid < 256) {
                __update(dists, dists_i, tid, tid + 256);
            }
            __syncthreads();
        }
        if (block_size >= 256) {
            if (tid < 128) {
                __update(dists, dists_i, tid, tid + 128);
            }
            __syncthreads();
        }
        if (block_size >= 128) {
            if (tid < 64) {
                __update(dists, dists_i, tid, tid + 64);
            }
            __syncthreads();
        }
        if (block_size >= 64) {
            if (tid < 32) {
                __update(dists, dists_i, tid, tid + 32);
            }
            __syncthreads();
        }
        if (block_size >= 32) {
            if (tid < 16) {
                __update(dists, dists_i, tid, tid + 16);
            }
            __syncthreads();
        }
        if (block_size >= 16) {
            if (tid < 8) {
                __update(dists, dists_i, tid, tid + 8);
            }
            __syncthreads();
        }
        if (block_size >= 8) {
            if (tid < 4) {
                __update(dists, dists_i, tid, tid + 4);
            }
            __syncthreads();
        }
        if (block_size >= 4) {
            if (tid < 2) {
                __update(dists, dists_i, tid, tid + 2);
            }
            __syncthreads();
        }
        if (block_size >= 2) {
            if (tid < 1) {
                __update(dists, dists_i, tid, tid + 1);
            }
            __syncthreads();
        }

        old = dists_i[0];
        if (tid == 0)
            idxs[j] = old;
    }
}
void farthest_point_sampling_kernel_launcher(int b, int n, int m, const float* dataset, float* temp, int* idxs)
{
    // dataset: (B, N, 3)
    // tmp: (B, N)
    // output:
    //      idx: (B, M)

    cudaError_t err;
    unsigned int n_threads = opt_n_threads(n);

    switch (n_threads)
    {
    case 1024: farthest_point_sampling_kernel<1024><<<b, n_threads>>>(b, n, m, dataset, temp, idxs); break;
    case 512: farthest_point_sampling_kernel<512><<<b, n_threads>>>(b, n, m, dataset, temp, idxs); break;
    case 256: farthest_point_sampling_kernel<256><<<b, n_threads>>>(b, n, m, dataset, temp, idxs); break;
    case 128: farthest_point_sampling_kernel<128><<<b, n_threads>>>(b, n, m, dataset, temp, idxs); break;
    case 64: farthest_point_sampling_kernel<64><<<b, n_threads>>>(b, n, m, dataset, temp, idxs); break;
    case 32: farthest_point_sampling_kernel<32><<<b, n_threads>>>(b, n, m, dataset, temp, idxs); break;
    case 16: farthest_point_sampling_kernel<16><<<b, n_threads>>>(b, n, m, dataset, temp, idxs); break;
    case 8: farthest_point_sampling_kernel<8><<<b, n_threads>>>(b, n, m, dataset, temp, idxs); break;
    case 4: farthest_point_sampling_kernel<4><<<b, n_threads>>>(b, n, m, dataset, temp, idxs); break;
    case 2: farthest_point_sampling_kernel<2><<<b, n_threads>>>(b, n, m, dataset, temp, idxs); break;
    case 1: farthest_point_sampling_kernel<1><<<b, n_threads>>>(b, n, m, dataset, temp, idxs); break;
    default: farthest_point_sampling_kernel<512><<<b, n_threads>>>(b, n, m, dataset, temp, idxs);
    }

    err = cudaGetLastError();
    if (cudaSuccess != err)
    {
        fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}
int32_t FPSamplingPlugin::enqueue(nvinfer1::PluginTensorDesc const* inputDesc,
    nvinfer1::PluginTensorDesc const* outputDesc, void const* const* inputs, void* const* outputs, void* workspace,
    cudaStream_t stream) noexcept
{
    try
    {
        int32_t batchSize = inputDesc[0].dims.d[0];

        // TRT-input
        const auto* pointCloudPtr = static_cast<const float3*>(inputs[0]);
        // TRT-output
        auto* sampledIndPtr = static_cast<uint32_t*>(outputs[0]);
        // Temporary
        size_t pointDistSize = batchSize * mNbPoints * sizeof(float);

        size_t workspaces[1];
        workspaces[0] = pointDistSize;
        auto* pointDistPtr = static_cast<float*>(workspace);
        thrust::device_ptr<float> dev_pointDistPtr(pointDistPtr);
        thrust::fill(dev_pointDistPtr, dev_pointDistPtr + batchSize * mNbPoints, 1e10);

        // compute potential voxel size.

        return 0;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return -1;
}

nvinfer1::DataType FPSamplingPlugin::getOutputDataType(
    int32_t index, nvinfer1::DataType const* inputTypes, int32_t nbInputs) const noexcept
{
    try
    {
        PLUGIN_VALIDATE(inputTypes != nullptr);
        if (index == 0)
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

char const* FPSamplingPlugin::getPluginType() const noexcept
{
    return kHAV_SAMPLING_PLUGIN_NAME;
}

char const* FPSamplingPlugin::getPluginVersion() const noexcept
{
    return kHAV_SAMPLING_PLUGIN_VERSION;
}

int32_t FPSamplingPlugin::getNbOutputs() const noexcept
{
    return 1;
}

int32_t FPSamplingPlugin::initialize() noexcept
{
    return 0;
}

void FPSamplingPlugin::terminate() noexcept {}

size_t FPSamplingPlugin::getSerializationSize() const noexcept
{
    return kSERIALIZATION_SIZE;
}

void FPSamplingPlugin::serialize(void* buffer) const noexcept
{

    PLUGIN_ASSERT(buffer != nullptr);
    uint8_t* d = reinterpret_cast<uint8_t*>(buffer);
    auto* a = d;
    writeToBuffer<int32_t>(d, mNbSamples);
    PLUGIN_ASSERT(d == a + getSerializationSize());
}

void FPSamplingPlugin::destroy() noexcept
{
    delete this;
}

void FPSamplingPlugin::setPluginNamespace(char const* libNamespace) noexcept
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

char const* FPSamplingPlugin::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

FPSamplingPluginCreator::FPSamplingPluginCreator()
{
    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(PluginField("sample_num", nullptr, PluginFieldType::kINT32, 1));
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

char const* FPSamplingPluginCreator::getPluginName() const noexcept
{
    return kHAV_SAMPLING_PLUGIN_NAME;
}

char const* FPSamplingPluginCreator::getPluginVersion() const noexcept
{
    return kHAV_SAMPLING_PLUGIN_VERSION;
}

PluginFieldCollection const* FPSamplingPluginCreator::getFieldNames() noexcept
{
    return &mFC;
}

IPluginV2* FPSamplingPluginCreator::createPlugin(char const* name, PluginFieldCollection const* fc) noexcept
{
    try
    {
        PLUGIN_VALIDATE(fc != nullptr);
        PluginField const* fields = fc->fields;
        int32_t nbFields = fc->nbFields;

        int32_t nbSamples = 0;

        for (int32_t i = 0; i < nbFields; ++i)
        {
            char const* attrName = fields[i].name;
            if (!strcmp(attrName, "sample_num"))
            {
                int32_t const* d = static_cast<int32_t const*>(fields[i].data);
                nbSamples = d[0];
            }
        }
        IPluginV2* plugin = new FPSamplingPlugin(nbSamples);
        return plugin;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

IPluginV2* FPSamplingPluginCreator::deserializePlugin(
    char const* name, void const* serialData, size_t serialLength) noexcept
{
    try
    {
        return new FPSamplingPlugin(serialData, serialLength);
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

void FPSamplingPluginCreator::setPluginNamespace(char const* libNamespace) noexcept
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

char const* FPSamplingPluginCreator::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}
REGISTER_TENSORRT_PLUGIN(FPSamplingPluginCreator);
} // namespace plugin
} // namespace nvinfer1
