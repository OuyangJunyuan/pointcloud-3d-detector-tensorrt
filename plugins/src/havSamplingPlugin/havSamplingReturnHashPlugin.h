//
// Created by nrsl on 23-5-1.
//

#ifndef POINT_DETECTION_HAVSAMPLINGRETURNHASHPLUGIN_H
#define POINT_DETECTION_HAVSAMPLINGRETURNHASHPLUGIN_H

using voxel3 = float[3];
// #define TENSORRT_PLUGIN_DEBUG
#define TENSORRT_PLUGIN_SETTING                                                                                         \
(                                                                                                                       \
    name(HAVSamplingReturnHash),                                                                                        \
    version("1"),                                                                                                       \
    attribute(                                                                                                          \
        (int        , num_sample    , 0),                                                                               \
        (voxel3     , init_voxel    , {0.f, 0.f, 0.f}),                                                                 \
        (float      , tolerance     , 0.f),                                                                             \
        (int        , max_iteration , 10)                                                                               \
    ),                                                                                                                  \
    define(                                                                                                             \
        (num_batch  , inputs(0)(0)),                                                                                    \
        (num_source , inputs(0)(1)),                                                                                    \
        (num_sample , attr.num_sample),                                                                                 \
        (num_hash   , get_table_size(const(num_source)))                                                                \
    ),                                                                                                                  \
    input(                                                                                                              \
        (float      , sources       , dim(num_batch, num_source, 3))                                                    \
    ),                                                                                                                  \
    output(                                                                                                             \
        (uint32_t   , indices       , dim(num_batch, num_sample)),                                                      \
        (float      , voxel_infos   , dim(num_batch, 3 * 3)),                                                           \
        (uint32_t   , hash_tables   , dim(num_batch, num_hash)),                                                        \
        (float      , dist_tables   , dim(num_batch, num_hash))                                                         \
    ),                                                                                                                  \
    workspace(                                                                                                          \
        (uint32_t   , batch_masks   , dim(num_batch + 1)),                                                              \
        (uint32_t   , num_sampled   , dim(num_batch)),                                                                  \
        (uint32_t   , point_slots   , dim(num_batch, num_source)),                                                      \
        (float      , point_dists   , dim(num_batch, num_source)),                                                      \
        (uint8_t    , point_masks   , dim(num_batch, num_source))                                                       \
    )                                                                                                                   \
)

#include "common/plugin_auto_declare.h"

#endif //POINT_DETECTION_HAVSAMPLINGRETURNHASHPLUGIN_H
