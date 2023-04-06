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

#include "nms3DPlugin.h"

#include <cstring>
#include <iostream>

#include "nms3D.h"
#include "common/print.h"
#include "common/plugin.h"
#include "common/common.h"

namespace nvinfer1 {
namespace plugin {

using namespace nvinfer1;
using nvinfer1::plugin::nms3DPlugin;
using nvinfer1::plugin::nms3DPluginCreator;

namespace {
char const *const kNMSBEV_PLUGIN_VERSION{"1"};
char const *const kBEVNMS_PLUGIN_NAME{"NMSBEV"};
size_t constexpr kSERIALIZATION_SIZE{2 * sizeof(float) + 1 * sizeof(int)};
} // namespace

// Static class fields initialization
PluginFieldCollection nms3DPluginCreator::mFC{};
std::vector<PluginField> nms3DPluginCreator::mPluginAttributes;

nms3DPlugin::nms3DPlugin(float score_thresh, float iou_thresh, int max_num_nms)
        : mAttrs{score_thresh, iou_thresh, max_num_nms} {
}

nms3DPluginCreator::nms3DPluginCreator() {
    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(PluginField("score_threshold", nullptr, PluginFieldType::kFLOAT32, 1));
    mPluginAttributes.emplace_back(PluginField("iou_threshold", nullptr, PluginFieldType::kFLOAT32, 1));
    mPluginAttributes.emplace_back(PluginField("num_post_nms", nullptr, PluginFieldType::kINT32, 1));
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

IPluginV2 *nms3DPluginCreator::createPlugin(char const *name, PluginFieldCollection const *fc) noexcept {
    try {
        PLUGIN_VALIDATE(fc != nullptr);
        PluginField const *fields = fc->fields;
        int32_t nbFields = fc->nbFields;

        float iou_threshold = 0;
        float score_threshold = 0;
        int num_post_nms = 0;

        for (int32_t i = 0; i < nbFields; ++i) {
            char const *attrName = fields[i].name;
            if (!strcmp(attrName, "iou_threshold")) {
                auto const *d = static_cast<float const *>(fields[i].data);
                iou_threshold = d[0];
            } else if (!strcmp(attrName, "score_threshold")) {
                auto const *d = static_cast<float const *>(fields[i].data);
                score_threshold = d[0];
            } else if (!strcmp(attrName, "num_post_nms")) {
                auto const *d = static_cast<int const *>(fields[i].data);
                num_post_nms = d[0];
            }
        }
        IPluginV2 *plugin = new nms3DPlugin(score_threshold, iou_threshold, num_post_nms);
        return plugin;
    }
    catch (std::exception const &e) {
        caughtError(e);
    }
    return nullptr;
}

void nms3DPlugin::serialize(void *buffer) const noexcept {

    PLUGIN_ASSERT(buffer != nullptr);
    uint8_t *d = reinterpret_cast<uint8_t *>(buffer);
    auto *a = d;
    writeToBuffer<float>(d, mAttrs.score_thresh);
    writeToBuffer<float>(d, mAttrs.iou_thresh);
    writeToBuffer<int>(d, mAttrs.max_nms_num);
    PLUGIN_ASSERT(d == a + getSerializationSize());
}

nms3DPlugin::nms3DPlugin(void const *data, size_t length) {
    PLUGIN_ASSERT(data != nullptr);
    uint8_t const *d = reinterpret_cast<uint8_t const *>(data);
    auto const *a = d;
    mAttrs.score_thresh = readFromBuffer<float>(d);
    mAttrs.iou_thresh = readFromBuffer<float>(d);
    mAttrs.max_nms_num = readFromBuffer<int>(d);
    PLUGIN_ASSERT(d == a + length);
}

IPluginV2 *
nms3DPluginCreator::deserializePlugin(char const *name, void const *serialData, size_t serialLength) noexcept {
    try {
        return new nms3DPlugin(serialData, serialLength);
    }
    catch (std::exception const &e) {
        caughtError(e);
    }
    return nullptr;
}

nvinfer1::IPluginV2DynamicExt *nms3DPlugin::clone() const noexcept {
    try {
        auto *plugin = new nms3DPlugin(mAttrs.score_thresh, mAttrs.iou_thresh, mAttrs.max_nms_num);
        plugin->setPluginNamespace(mNamespace.c_str());
        return plugin;
    }
    catch (std::exception const &e) {
        caughtError(e);
    }
    return nullptr;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
bool nms3DPlugin::supportsFormatCombination(
        int32_t pos, nvinfer1::PluginTensorDesc const *inOut, int32_t nbInputs, int32_t nbOutputs) noexcept {
    try {
        PLUGIN_VALIDATE(inOut != nullptr);
        PLUGIN_VALIDATE(nbInputs == 2);
        PLUGIN_VALIDATE(nbOutputs == 3);
        PluginTensorDesc const &in = inOut[pos];
        // input boxes
        if (pos == 0) {
            return (in.type == nvinfer1::DataType::kFLOAT) && (in.format == TensorFormat::kLINEAR);
        }
        // input scores
        if (pos == 1) {
            return (in.type == nvinfer1::DataType::kFLOAT) && (in.format == TensorFormat::kLINEAR);
        }
        // output boxes
        if (pos == 2) {
            return (in.type == nvinfer1::DataType::kFLOAT) && (in.format == TensorFormat::kLINEAR);
        }
        // output scores
        if (pos == 3) {
            return (in.type == nvinfer1::DataType::kFLOAT) && (in.format == TensorFormat::kLINEAR);
        }
        // output nums
        if (pos == 4) {
            return (in.type == nvinfer1::DataType::kINT32) && (in.format == TensorFormat::kLINEAR);
        }
        return false;
    }
    catch (std::exception const &e) {
        caughtError(e);
    }
    return false;
}

int32_t nms3DPlugin::getNbOutputs() const noexcept {
    return 3;
}

nvinfer1::DataType nms3DPlugin::getOutputDataType(
        int32_t index, nvinfer1::DataType const *inputTypes, int32_t nbInputs) const noexcept {
    try {
        PLUGIN_VALIDATE(inputTypes != nullptr);
        if (index == 0) {
            return nvinfer1::DataType::kFLOAT;
        }
        if (index == 1) {
            return nvinfer1::DataType::kFLOAT;
        }
        if (index == 2) {
            return nvinfer1::DataType::kINT32;
        }
    }
    catch (std::exception const &e) {
        caughtError(e);
    }
    return nvinfer1::DataType{};
}

nvinfer1::DimsExprs nms3DPlugin::getOutputDimensions(int32_t outputIndex, nvinfer1::DimsExprs const *inputs,
                                                     int32_t nbInputs, nvinfer1::IExprBuilder &exprBuilder) noexcept {
    try {
        PLUGIN_VALIDATE(outputIndex >= 0 && outputIndex < this->getNbOutputs());
        nvinfer1::DimsExprs dim{};
        if (outputIndex == 0) {
            // final_boxes
            dim.nbDims = 3;
            dim.d[0] = inputs[0].d[0];
            dim.d[1] = exprBuilder.constant(mAttrs.max_nms_num);
            dim.d[2] = inputs[0].d[2];
        }
        if (outputIndex == 1) {
            // final_scores
            dim.nbDims = 2;
            dim.d[0] = inputs[0].d[0];
            dim.d[1] = exprBuilder.constant(mAttrs.max_nms_num);
            dim.d[2] = exprBuilder.constant(1);
        }
        if (outputIndex == 2) {
            // num_detections
            dim.nbDims = 2;
            dim.d[0] = inputs[0].d[0];
            dim.d[1] = exprBuilder.constant(1);
        }

        return dim;
    }
    catch (std::exception const &e) {
        caughtError(e);
    }
    return nvinfer1::DimsExprs{};
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void nms3DPlugin::configurePlugin(nvinfer1::DynamicPluginTensorDesc const *in, int32_t nbInputs,
                                  nvinfer1::DynamicPluginTensorDesc const *out, int32_t nbOutputs) noexcept {
    try {
        PLUGIN_VALIDATE(in != nullptr);
        PLUGIN_VALIDATE(out != nullptr);
        PLUGIN_VALIDATE(nbInputs == 2);
        PLUGIN_VALIDATE(nbOutputs == 3);
    }
    catch (std::exception const &e) {
        caughtError(e);
    }
}

size_t nms3DPlugin::getWorkspaceSize(nvinfer1::PluginTensorDesc const *inputDesc, int32_t nbInputs,
                                     nvinfer1::PluginTensorDesc const *outputDesc, int32_t nbOutputs) const noexcept {
    try {
        // boxes (b,n,7+c)
        int batch_size = inputDesc[0].dims.d[0];
        int box_nums = inputDesc[0].dims.d[1];
        size_t total_box_num = batch_size * box_nums;

        size_t validIndices = total_box_num * sizeof(uint32_t);
        size_t validScores = total_box_num * sizeof(float);
        size_t validNums = batch_size * sizeof(uint32_t);
        size_t validIndStart = batch_size * sizeof(uint32_t);
        size_t validIndEnd = batch_size * sizeof(uint32_t);
        size_t sortTempWorkSpace = sortTempWorkSpaceSize(batch_size, box_nums);
        size_t sortedIndices = total_box_num * sizeof(uint32_t);
        size_t sortedScores = total_box_num * sizeof(float);

        std::vector<size_t> workspaces({validIndices, validScores, validNums, validIndStart, validIndEnd,
                                        sortTempWorkSpace, sortedIndices, sortedScores});
        return calculateTotalWorkspaceSize(workspaces.data(), workspaces.size());
    }
    catch (std::exception const &e) {
        caughtError(e);
    }
    return 0U;
}

int32_t nms3DPlugin::enqueue(nvinfer1::PluginTensorDesc const *inputDesc, nvinfer1::PluginTensorDesc const *outputDesc,
                             void const *const *inputs, void *const *outputs, void *workspace,
                             cudaStream_t stream) noexcept {
    try {
        int batch_size = inputDesc[0].dims.d[0];
        int box_nums = inputDesc[0].dims.d[1];
        int box_dims = inputDesc[0].dims.d[2]; // 7+c

        auto const *const boxesInput = static_cast<float const *>(inputs[0]);
        auto const *const scoresInput = static_cast<float const *>(inputs[1]);

        auto *const boxesOutput = static_cast<float *>(outputs[0]);
        auto *const scoresOutput = static_cast<float *>(outputs[1]);
        auto *const numsOutput = static_cast<uint32_t *>(outputs[2]);

        nms3DInference(boxesInput, scoresInput, boxesOutput, scoresOutput, numsOutput, mAttrs.score_thresh,
                       mAttrs.iou_thresh, mAttrs.max_nms_num, batch_size, box_nums, box_dims, workspace, stream);
        return 0;
    }
    catch (std::exception const &e) {
        caughtError(e);
    }
    return -1;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
char const *nms3DPlugin::getPluginType() const noexcept {
    return kBEVNMS_PLUGIN_NAME;
}

char const *nms3DPlugin::getPluginVersion() const noexcept {
    return kNMSBEV_PLUGIN_VERSION;
}

int32_t nms3DPlugin::initialize() noexcept {
    return 0;
}

void nms3DPlugin::terminate() noexcept {}

size_t nms3DPlugin::getSerializationSize() const noexcept {
    return kSERIALIZATION_SIZE;
}

void nms3DPlugin::destroy() noexcept {
    delete this;
}

void nms3DPlugin::setPluginNamespace(char const *libNamespace) noexcept {
    try {
        PLUGIN_VALIDATE(libNamespace != nullptr);
        mNamespace = libNamespace;
    }
    catch (std::exception const &e) {
        caughtError(e);
    }
}

char const *nms3DPlugin::getPluginNamespace() const noexcept {
    return mNamespace.c_str();
}

char const *nms3DPluginCreator::getPluginName() const noexcept {
    return kBEVNMS_PLUGIN_NAME;
}

char const *nms3DPluginCreator::getPluginVersion() const noexcept {
    return kNMSBEV_PLUGIN_VERSION;
}

PluginFieldCollection const *nms3DPluginCreator::getFieldNames() noexcept {
    return &mFC;
}

void nms3DPluginCreator::setPluginNamespace(char const *libNamespace) noexcept {
    try {
        PLUGIN_VALIDATE(libNamespace != nullptr);
        mNamespace = libNamespace;
    }
    catch (std::exception const &e) {
        caughtError(e);
    }
}

char const *nms3DPluginCreator::getPluginNamespace() const noexcept {
    return mNamespace.c_str();
}

REGISTER_TENSORRT_PLUGIN(nms3DPluginCreator);
} // namespace plugin
} // namespace nvinfer1
