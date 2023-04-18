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

#ifndef TRT_VOXEL_GENERATOR_H
#define TRT_VOXEL_GENERATOR_H
#include <memory>
#include <string>
#include <vector>

#include <NvInferPlugin.h>
#include <cuda_runtime_api.h>
// #define TENSORRT_PLUGIN_DEBUG
// #define TENSORRT_PLUGIN                                                                                                 \
// Setting(                                                                                                                \
//     Name(HAVSampling),                                                                                                  \
//     Version("1"),                                                                                                       \
//     (                                                                                                                   \
//         Define(size_t, num_batch, Input(0,0))                                                                           \
//         Define(size_t, num_point, Input(0,1))                                                                           \
//         Define(size_t, table_size, get_table_size(num_point, 2048))                                                     \
//     ),                                                                                                                  \
//     (                                                                                                                   \
//         Input(float, xyz, Dim3(num_batch, num_point, 3))                                                                \
//     ),                                                                                                                  \
//     (                                                                                                                   \
//         Output(int32_t, indices, Dim2(Input(0,0), Const(sample_num)))                                                   \
//     ),                                                                                                                  \
//     (                                                                                                                   \
//         Workspace(int8_t    , batch_mask,   Dim1( Input(0,0) ) )                                                        \
//         Workspace(uint32_t  , hash_table,   Dim2( Input(0,0), table_size ) )                                            \
//     ),                                                                                                                  \
//     (                                                                                                                   \
//         Attribute(int, sample_num, 1)                                                                                   \
//         Attribute(float[3], voxel_size, {0.5,0.5,0.5})                                                                  \
//         Attribute(float, tolerance, 0.05)                                                                               \
//         Attribute(int , max_iter, 10)                                                                                   \
//     )                                                                                                                   \
// )

namespace nvinfer1
{
namespace plugin
{

class HAVSamplingPlugin : public nvinfer1::IPluginV2DynamicExt
{
public:
    HAVSamplingPlugin() = delete;
    HAVSamplingPlugin(int32_t nbSamples, float voxelX, float voxelY, float voxelZ, float tolerance, int32_t maxIters);
    HAVSamplingPlugin(void const* data, size_t length);
    // IPluginV2DynamicExt Methods
    nvinfer1::IPluginV2DynamicExt* clone() const noexcept override;
    nvinfer1::DimsExprs getOutputDimensions(int32_t outputIndex, nvinfer1::DimsExprs const* inputs, int32_t nbInputs,
        nvinfer1::IExprBuilder& exprBuilder) noexcept override;
    bool supportsFormatCombination(
        int32_t pos, nvinfer1::PluginTensorDesc const* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept override;
    void configurePlugin(nvinfer1::DynamicPluginTensorDesc const* in, int32_t nbInputs,
        nvinfer1::DynamicPluginTensorDesc const* out, int32_t nbOutputs) noexcept override;
    size_t getWorkspaceSize(nvinfer1::PluginTensorDesc const* inputs, int32_t nbInputs,
        nvinfer1::PluginTensorDesc const* outputs, int32_t nbOutputs) const noexcept override;
    int32_t enqueue(nvinfer1::PluginTensorDesc const* inputDesc, nvinfer1::PluginTensorDesc const* outputDesc,
        void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept override;
    // IPluginV2Ext Methods
    nvinfer1::DataType getOutputDataType(
        int32_t index, nvinfer1::DataType const* inputTypes, int32_t nbInputs) const noexcept override;
    // IPluginV2 Methods
    char const* getPluginType() const noexcept override;
    char const* getPluginVersion() const noexcept override;
    int32_t getNbOutputs() const noexcept override;
    int32_t initialize() noexcept override;
    void terminate() noexcept override;
    size_t getSerializationSize() const noexcept override;
    void serialize(void* buffer) const noexcept override;
    void destroy() noexcept override;
    void setPluginNamespace(char const* pluginNamespace) noexcept override;
    char const* getPluginNamespace() const noexcept override;

private:
    std::string mNamespace;
    int32_t mNbSamples;
    float mVoxelX;
    float mVoxelY;
    float mVoxelZ;
    float mTolerance;
    int32_t mMaxIters;
};

class HAVSamplingPluginCreator : public nvinfer1::IPluginCreator
{
public:
    HAVSamplingPluginCreator();
    char const* getPluginName() const noexcept override;
    char const* getPluginVersion() const noexcept override;
    nvinfer1::PluginFieldCollection const* getFieldNames() noexcept override;
    nvinfer1::IPluginV2* createPlugin(char const* name, nvinfer1::PluginFieldCollection const* fc) noexcept override;
    nvinfer1::IPluginV2* deserializePlugin(
        char const* name, void const* serialData, size_t serialLength) noexcept override;
    void setPluginNamespace(char const* pluginNamespace) noexcept override;
    char const* getPluginNamespace() const noexcept override;

private:
    static nvinfer1::PluginFieldCollection mFC;
    static std::vector<nvinfer1::PluginField> mPluginAttributes;
    std::string mNamespace;
};

} // namespace plugin
} // namespace nvinfer1

#endif
