//
// Created by nrsl on 23-5-1.
//

#ifndef POINT_DETECTION_BALLQUERY_CUH
#define POINT_DETECTION_BALLQUERY_CUH

void BallQueryLauncher(int batchSize, int pointSize, int querySize, float radius, int neighborSize,
                       const float *xyzPtr, const float *queryPtr,
                       int32_t *cntPtr, int32_t *indPtr,
                       cudaStream_t stream);

#endif //POINT_DETECTION_BALLQUERY_CUH
