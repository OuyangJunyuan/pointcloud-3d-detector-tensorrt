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

#include <cuda_runtime_api.h>
#include <memory>
#include <string>
#include <vector>

#include <NvInferPlugin.h>

namespace nvinfer1 {
namespace plugin {

class nms3DPlugin : public nvinfer1::IPluginV2DynamicExt {
 public:
    nms3DPlugin() = delete;

    nms3DPlugin(float score_thresh, float iou_thresh, int max_num_nms);

    nms3DPlugin(void const *data, size_t length);

    // IPluginV2DynamicExt Methods
    nvinfer1::IPluginV2DynamicExt *clone() const noexcept override;

    nvinfer1::DimsExprs getOutputDimensions(int32_t outputIndex, nvinfer1::DimsExprs const *inputs, int32_t nbInputs,
                                            nvinfer1::IExprBuilder &exprBuilder) noexcept override;

    bool supportsFormatCombination(
            int32_t pos, nvinfer1::PluginTensorDesc const *inOut, int32_t nbInputs,
            int32_t nbOutputs) noexcept override;

    void configurePlugin(nvinfer1::DynamicPluginTensorDesc const *in, int32_t nbInputs,
                         nvinfer1::DynamicPluginTensorDesc const *out, int32_t nbOutputs) noexcept override;

    size_t getWorkspaceSize(nvinfer1::PluginTensorDesc const *inputs, int32_t nbInputs,
                            nvinfer1::PluginTensorDesc const *outputs, int32_t nbOutputs) const noexcept override;

    int32_t enqueue(nvinfer1::PluginTensorDesc const *inputDesc, nvinfer1::PluginTensorDesc const *outputDesc,
                    void const *const *inputs, void *const *outputs, void *workspace,
                    cudaStream_t stream) noexcept override;

    // IPluginV2Ext Methods
    nvinfer1::DataType getOutputDataType(
            int32_t index, nvinfer1::DataType const *inputTypes, int32_t nbInputs) const noexcept override;

    // IPluginV2 Methods
    char const *getPluginType() const noexcept override;

    char const *getPluginVersion() const noexcept override;

    int32_t getNbOutputs() const noexcept override;

    int32_t initialize() noexcept override;

    void terminate() noexcept override;

    size_t getSerializationSize() const noexcept override;

    void serialize(void *buffer) const noexcept override;

    void destroy() noexcept override;

    void setPluginNamespace(char const *pluginNamespace) noexcept override;

    char const *getPluginNamespace() const noexcept override;

 private:
    std::string mNamespace;

    struct {
        float score_thresh{0.0f};
        float iou_thresh{0.0f};
        int max_nms_num{0};
    } mAttrs;
};

class nms3DPluginCreator : public nvinfer1::IPluginCreator {
 public:
    nms3DPluginCreator();

    char const *getPluginName() const noexcept override;

    char const *getPluginVersion() const noexcept override;

    nvinfer1::PluginFieldCollection const *getFieldNames() noexcept override;

    nvinfer1::IPluginV2 *createPlugin(char const *name, nvinfer1::PluginFieldCollection const *fc) noexcept override;

    nvinfer1::IPluginV2 *deserializePlugin(
            char const *name, void const *serialData, size_t serialLength) noexcept override;

    void setPluginNamespace(char const *pluginNamespace) noexcept override;

    char const *getPluginNamespace() const noexcept override;

 private:
    static nvinfer1::PluginFieldCollection mFC;
    static std::vector<nvinfer1::PluginField> mPluginAttributes;
    std::string mNamespace;
};

} // namespace plugin
} // namespace nvinfer1

#endif
