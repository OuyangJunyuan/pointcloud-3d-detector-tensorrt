#ifndef TRT_VOXEL_GENERATOR_H
#define TRT_VOXEL_GENERATOR_H

#define TENSORRT_PLUGIN_DEBUG
#define TENSORRT_PLUGIN                                                                                                 \
Setting(                                                                                                                \
    Name(FPSampling),                                                                                                   \
    Version("1"),                                                                                                       \
    (                                                                                                                   \
        Define(size_t, num_batch, Input(0,0))                                                                           \
        Define(size_t, num_point, Input(0,1))                                                                           \
    ),                                                                                                                  \
    (                                                                                                                   \
        Input(float, xyz, Dim3(num_batch, num_point, 3))                                                                \
    ),                                                                                                                  \
    (                                                                                                                   \
        Output(int32_t, indices, Dim2(Input(0,0), Const(sample_num)))                                                   \
    ),                                                                                                                  \
    (                                                                                                                   \
        Workspace(float, furthest_dists, Dim2(Input(0,0), Input(0,1)))                                                  \
    ),                                                                                                                  \
    (                                                                                                                   \
        Attribute(int32_t, sample_num, 1)                                                                               \
    )                                                                                                                   \
)


#include "common/plugin_auto_declare.h"

#endif
