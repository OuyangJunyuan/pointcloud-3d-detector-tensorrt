#ifndef TRT_VOXEL_GENERATOR_H
#define TRT_VOXEL_GENERATOR_H

#define TENSORRT_PLUGIN                                                                                                 \
Setting(                                                                                                                \
    Name(FPSampling),                                                                                                   \
    Version("1"),                                                                                                       \
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

struct FPSamplingUser {

};

#include "common/plugin_auto_declare.h"

#endif
