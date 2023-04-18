//
// Created by nrsl on 23-4-16.
//

#ifndef POINT_DETECTION_FPSAMPLING_H
#define POINT_DETECTION_FPSAMPLING_H

#include <cuda_runtime.h>

void farthest_point_sampling_kernel_launcher(int b, int n, int m, const float *xyz, float *temp, int *indices,
                                             cudaStream_t stream);

#endif //POINT_DETECTION_FPSAMPLING_H
