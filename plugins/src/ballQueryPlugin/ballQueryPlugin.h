//
// Created by nrsl on 23-5-1.
//

#ifndef POINT_DETECTION_BALLQUERYPLUGIN_H
#define POINT_DETECTION_BALLQUERYPLUGIN_H

#define TENSORRT_PLUGIN_DEBUG
#define TENSORRT_PLUGIN_SETTING                                                                                         \
(                                                                                                                       \
    name(BallQuery),                                                                                                    \
    version("1"),                                                                                                       \
    attribute(                                                                                                          \
        (int    , num_neighbor  , 16),                                                                                  \
        (float  , radius        , 0.4f)                                                                                 \
    ),                                                                                                                  \
    define(                                                                                                             \
        (num_batch  , inputs(0)(0)),                                                                                    \
        (num_source , inputs(0)(1)),                                                                                    \
        (num_query  , inputs(1)(1))                                                                                     \
    ),                                                                                                                  \
    input(                                                                                                              \
        (float, sources, dim(num_batch, num_source  , 3)),                                                              \
        (float, queries, dim(num_batch, num_query   , 3))                                                               \
    ),                                                                                                                  \
    output(                                                                                                             \
        (int32_t, num_valid , dim(num_batch, num_query)),                                                               \
        (int32_t, indices   , dim(num_batch, num_query, attr.num_neighbor))                                             \
    ),                                                                                                                  \
    workspace(                                                                                                          \
        (uint8_t, dummy, dim(0))                                                                                        \
    )                                                                                                                   \
)

#include "common/plugin_auto_declare.h"

#endif //POINT_DETECTION_BALLQUERYPLUGIN_H
