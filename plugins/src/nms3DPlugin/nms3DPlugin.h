//
// Created by nrsl on 23-5-1.
//

#ifndef POINT_DETECTION_NMS3DPLUGIN_H
#define POINT_DETECTION_NMS3DPLUGIN_H

#include "nms3D.h"

// #define TENSORRT_PLUGIN_DEBUG
#define TENSORRT_PLUGIN_SETTING                                                                                         \
(                                                                                                                       \
    name(NMSBEV),                                                                                                       \
    version("1"),                                                                                                       \
    attribute(                                                                                                          \
        (float, score_threshold , 0.1f),                                                                                \
        (float, iou_threshold   , 0.01f),                                                                               \
        (int  , num_max_nms     , 256)                                                                                  \
    ),                                                                                                                  \
    define(                                                                                                             \
        (num_batch      , inputs(0)(0)),                                                                                \
        (num_box        , inputs(0)(1)),                                                                                \
        (num_box_feat   , inputs(0)(2)),                                                                                \
        (num_max_box    , attr.num_max_nms),                                                                            \
        (num_sort_temp  , sortTempWorkSpaceSize(def.num_batch, def.num_box))                                            \
    ),                                                                                                                  \
    input(                                                                                                              \
        (float, boxes   , dim(num_batch, num_box, num_box_feat)),                                                       \
        (float, scores  , dim(num_batch, num_box, 1))                                                                   \
    ),                                                                                                                  \
    output(                                                                                                             \
        (float      ,   final_boxes , dim(num_batch, num_max_box, num_box_feat)),                                       \
        (float      ,   final_scores, dim(num_batch, num_max_box)),                                                     \
        (uint32_t   ,   num_valid   , dim(num_batch, 1))                                                                \
    ),                                                                                                                  \
    workspace(                                                                                                          \
        (uint32_t   , valid_indices     , dim(num_batch, num_box)),                                                     \
        (float      , valid_scores      , dim(num_batch, num_box)),                                                     \
        (uint32_t   , valid_nums        , dim(num_batch)),                                                              \
        (uint32_t   , valid_ind_start   , dim(num_batch)),                                                              \
        (uint32_t   , valid_ind_end     , dim(num_batch)),                                                              \
        (uint8_t    , sort_temp_memory  , dim(num_sort_temp)),                                                          \
        (uint32_t   , sorted_indices    , dim(num_batch, num_box)),                                                     \
        (float      , sorted_scores     , dim(num_batch, num_box))                                                      \
    )                                                                                                                   \
)

#include "common/plugin_auto_declare.h"

#endif //POINT_DETECTION_NMS3DPLUGIN_H
