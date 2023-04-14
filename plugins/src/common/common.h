/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#ifndef TRT_PLUGIN_COMMON_TEMPLATES_H_
#define TRT_PLUGIN_COMMON_TEMPLATES_H_

#define DIVUP(m, n) ((m) / (n) + ((m) % (n) > 0))
#define TOTAL_THREADS 1024
#define THREADS_PER_BLOCK 256
namespace nvinfer1
{
namespace plugin
{
template <typename ToType, typename FromType>
ToType* toPointer(FromType* ptr)
{
    return static_cast<ToType*>(static_cast<void*>(ptr));
}
template <typename ToType, typename FromType>
ToType const* toPointer(FromType const* ptr)
{
    return static_cast<ToType const*>(static_cast<void const*>(ptr));
}
// Helper function for serializing plugin
template <typename ValType, typename BufferType>
void writeToBuffer(BufferType*& buffer, ValType const& val)
{
    *toPointer<ValType>(buffer) = val;
    buffer += sizeof(ValType);
}

// Helper function for deserializing plugin
template <typename ValType, typename BufferType>
ValType readFromBuffer(BufferType const*& buffer)
{
    auto val = *toPointer<ValType const>(buffer);
    buffer += sizeof(ValType);
    return val;
}
size_t calculateTotalWorkspaceSize(size_t* workspaces, int32_t count);


int8_t* alignPtr(int8_t* ptr, uintptr_t to);

int8_t* nextWorkspacePtr(int8_t* ptr, uintptr_t previousWorkspaceSize);

} // namespace plugin
} // namespace nvinfer1
#endif // TRT_PLUGIN_COMMON_TEMPLATES_H_
