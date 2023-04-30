#ifndef TRT_VOXEL_GENERATOR_H
#define TRT_VOXEL_GENERATOR_H

#define TENSORRT_PLUGIN_DEBUG
#define TENSORRT_PLUGIN_SETTING                                                                                         \
(                                                                                                                       \
    name(FPSampling),                                                                                                   \
    version("1"),                                                                                                       \
    attribute(                                                                                                          \
        (int, sample_num, 1)                                                                                            \
    ),                                                                                                                  \
    define(                                                                                                             \
        (batch    , inputs(0)(0)),                                                                                      \
        (source   , inputs(0)(1)),                                                                                      \
        (target   , attr.sample_num)                                                                                    \
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
