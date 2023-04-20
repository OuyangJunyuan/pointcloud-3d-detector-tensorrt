#ifndef TRT_VOXEL_GENERATOR_H
#define TRT_VOXEL_GENERATOR_H

#define TENSORRT_PLUGIN_DEBUG
#define TENSORRT_PLUGIN_SETTING                                                                                         \
(                                                                                                                       \
    name(FPSampling),                                                                                                   \
    version("1"),                                                                                                       \
    attribute(                                                                                                          \
        (int, sample_num, 1),                                                                                           \
        (float[3], sample_num2, {0.5,0.5,0.5})                                                                          \
    ),                                                                                                                  \
    define(                                                                                                             \
        (batch    , Input(0,0)),                                                                                        \
        (source   , Input(0,1)),                                                                                        \
        (target   , Attr(sample_num))                                                                                   \
    ),                                                                                                                  \
    input(                                                                                                              \
        (float, xyz, dim(batch, source, 3))                                                                             \
    ),                                                                                                                  \
    output(                                                                                                             \
        (int32_t, indices, dim(batch, target))                                                                          \
    ),                                                                                                                  \
    workspace(                                                                                                          \
        (float, furthest_dists, dim(batch, source))                                                                     \
    )                                                                                                                   \
)

#include "common/plugin_auto_declare.h"

#endif
