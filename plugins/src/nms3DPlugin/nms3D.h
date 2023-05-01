//
// Created by nrsl on 23-3-21.
//

#ifndef TENSORRT_NMS3D_H
#define TENSORRT_NMS3D_H

#include <cinttypes>
#include <cuda_runtime.h>

//#define DEBUG1

#define THREAD_SIZE 256
#define BLOCKS2D(M, B) dim3((((M) + THREAD_SIZE - 1) / THREAD_SIZE), B)
#define THREADS() dim3(THREAD_SIZE)
#define NMS_TILES 5
#define BOX_DIMS_MAX 15  // (x,y,z,l,w,h,y,(p,r,vx,vy,iou,)cls)

/**
 * STATE MACRO
 */
#define BOX_UNPROCESSED 0
#define BOX_NEED_SAVE 1
#define BOX_PROCESSED (-1)

#define NMS_STATE_READY 0
#define NMS_NEED_IOU 1
#define NMS_SKIP_ITERATION (-1)
#define NMS_EARLY_STOP (-2)


void nms3DInference(const float *const boxesInput, const float *const scoresInput,
                    float *const boxesOutput, float *const scoresOutput, uint32_t *const numsOutput,
                    float score_threshold, float iou_threshold, int max_nms_num,
                    int batch_size, int box_nums, int box_dims, size_t sortTempSize,
                    uint32_t *const validIndices, float *const validScores, uint32_t *const validNums,
                    uint32_t *const validIndStart, uint32_t *const validIndEnd,
                    uint8_t *sortTempWorkspace, uint32_t *const sortedIndices, float *const sortedScores,
                    cudaStream_t stream);

size_t sortTempWorkSpaceSize(int batchSize, int box_nums);

#endif // TENSORRT_NMS3D_H
