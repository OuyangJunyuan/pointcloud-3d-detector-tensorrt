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

#include "common/hash.h"

#define TENSORRT_PLUGIN_DEBUG
#define TENSORRT_PLUGIN                                                                                                 \
Setting(                                                                                                                \
    Name(HAVSampling),                                                                                                  \
    Version("1"),                                                                                                       \
    (                                                                                                                   \
        Define(batch, Input(0,0))                                                                                       \
        Define(source, Input(0,1))                                                                                      \
        Define(target, Attr(sample_num))                                                                                \
        Define(table_size, get_table_size(source, 2048))                                                                \
    ),                                                                                                                  \
    (                                                                                                                   \
        Input(float, xyz, Dim3(batch, source, 3))                                                                       \
    ),                                                                                                                  \
    (                                                                                                                   \
        Output(int32_t, indices, Dim2(batch, target))                                                                   \
    ),                                                                                                                  \
    (                                                                                                                   \
        Workspace(int8_t    , batch_mask,   Dim1(batch) )                                                               \
        Workspace(uint32_t  , hash_table,   Dim2(batch, table_size) )                                                   \
        Workspace(float     , dist_table,   Dim2(batch, table_size) )                                                   \
        Workspace(uint32_t  , point_slot,   Dim2(batch, source) )                                                       \
        Workspace(float  , point_dist,   Dim2(batch, source) )                                                          \
        Workspace(uint32_t  , sampled,   Dim1(batch) )                                                                  \
        Workspace(Voxel  , voxel,   Dim1(batch) )                                                                       \
    ),                                                                                                                  \
    (                                                                                                                   \
        Attribute(int, sample_num, 1)                                                                                   \
        Attribute(float[3], voxel_size, {0.5,0.5,0.5})                                                                  \
        Attribute(float, tolerance, 0.05)                                                                               \
        Attribute(int , max_iter, 10)                                                                                   \
    )                                                                                                                   \
)

#include "common/plugin_auto_declare.h"

#endif
