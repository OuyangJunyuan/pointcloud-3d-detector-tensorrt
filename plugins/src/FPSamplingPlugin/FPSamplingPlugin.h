#ifndef TRT_VOXEL_GENERATOR_H
#define TRT_VOXEL_GENERATOR_H

#define TENSORRT_PLUGIN_DEBUG
#define TENSORRT_PLUGIN                                                                                                 \
Setting(                                                                                                                \
    Name(FPSampling),                                                                                                   \
    Version("1"),                                                                                                       \
    (                                                                                                                   \
        Define(batch    , Input(0,0))                                                                                   \
        Define(source   , Input(0,1))                                                                                   \
        Define(target   , Attr(sample_num))                                                                             \
    ),                                                                                                                  \
    (                                                                                                                   \
        Input(float, xyz, Dim3(batch, source, 3))                                                                       \
    ),                                                                                                                  \
    (                                                                                                                   \
        Output(int32_t, indices, Dim2(batch, target))                                                                   \
    ),                                                                                                                  \
    (                                                                                                                   \
        Workspace(float, furthest_dists, Dim2(batch, source))                                                           \
    ),                                                                                                                  \
    (                                                                                                                   \
        Attribute(int32_t, sample_num, 1)                                                                               \
    )                                                                                                                   \
)

#include "common/plugin_auto_declare.h"

#endif
