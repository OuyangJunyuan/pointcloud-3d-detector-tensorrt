
#include "nms3D.h"
#include "common/print.h"
#include "common/plugin.h"
#include "common/common.h"

#include <cub/cub.cuh>

const float EPS = 1e-8;

struct Point {
    float x, y;

    __device__ Point() {}

    __device__ Point(double _x, double _y) {
        x = _x, y = _y;
    }

    __device__ void set(float _x, float _y) {
        x = _x;
        y = _y;
    }

    __device__ Point operator+(const Point &b) const {
        return Point(x + b.x, y + b.y);
    }

    __device__ Point operator-(const Point &b) const {
        return Point(x - b.x, y - b.y);
    }
};

__device__ inline void
rotate_around_center(const Point &center, const float angle_cos, const float angle_sin, Point &p) {
    float new_x = (p.x - center.x) * angle_cos + (p.y - center.y) * (-angle_sin) + center.x;
    float new_y = (p.x - center.x) * angle_sin + (p.y - center.y) * angle_cos + center.y;
    p.set(new_x, new_y);
}

__device__ inline float cross(const Point &a, const Point &b) {
    return a.x * b.y - a.y * b.x;
}

__device__ inline float cross(const Point &p1, const Point &p2, const Point &p0) {
    return (p1.x - p0.x) * (p2.y - p0.y) - (p2.x - p0.x) * (p1.y - p0.y);
}

__device__ int check_rect_cross(const Point &p1, const Point &p2, const Point &q1, const Point &q2) {
    int ret = min(p1.x, p2.x) <= max(q1.x, q2.x) && min(q1.x, q2.x) <= max(p1.x, p2.x)
              && min(p1.y, p2.y) <= max(q1.y, q2.y) && min(q1.y, q2.y) <= max(p1.y, p2.y);
    return ret;
}

__device__ inline int intersection(const Point &p1, const Point &p0, const Point &q1, const Point &q0, Point &ans) {
    // fast exclusion
    if (check_rect_cross(p0, p1, q0, q1) == 0)
        return 0;

    // check cross standing
    float s1 = cross(q0, p1, p0);
    float s2 = cross(p1, q1, p0);
    float s3 = cross(p0, q1, q0);
    float s4 = cross(q1, p1, q0);

    if (!(s1 * s2 > 0 && s3 * s4 > 0))
        return 0;

    // calculate intersection of two lines
    float s5 = cross(q1, p1, p0);
    if (fabs(s5 - s1) > EPS) {
        ans.x = (s5 * q0.x - s1 * q1.x) / (s5 - s1);
        ans.y = (s5 * q0.y - s1 * q1.y) / (s5 - s1);
    } else {
        float a0 = p0.y - p1.y, b0 = p1.x - p0.x, c0 = p0.x * p1.y - p1.x * p0.y;
        float a1 = q0.y - q1.y, b1 = q1.x - q0.x, c1 = q0.x * q1.y - q1.x * q0.y;
        float D = a0 * b1 - a1 * b0;

        ans.x = (b0 * c1 - b1 * c0) / D;
        ans.y = (a1 * c0 - a0 * c1) / D;
    }

    return 1;
}

__device__ inline int check_in_box2d(const float *box, const Point &p) {
    // params: (7) [x, y, z, dx, dy, dz, heading]
    const float MARGIN = 1e-2;

    float center_x = box[0], center_y = box[1];
    float angle_cos = cos(-box[6]), angle_sin = sin(-box[6]); // rotate the point in the opposite direction of box
    float rot_x = (p.x - center_x) * angle_cos + (p.y - center_y) * (-angle_sin);
    float rot_y = (p.x - center_x) * angle_sin + (p.y - center_y) * angle_cos;

    return (fabs(rot_x) < box[3] / 2 + MARGIN && fabs(rot_y) < box[4] / 2 + MARGIN);
}

__device__ inline int point_cmp(const Point &a, const Point &b, const Point &center) {
    return atan2(a.y - center.y, a.x - center.x) > atan2(b.y - center.y, b.x - center.x);
}

__device__ inline float box_overlap(const float *box_a, const float *box_b) {
    // params box_a: [x, y, z, dx, dy, dz, heading]
    // params box_b: [x, y, z, dx, dy, dz, heading]

    float a_angle = box_a[6], b_angle = box_b[6];
    float a_dx_half = box_a[3] / 2, b_dx_half = box_b[3] / 2, a_dy_half = box_a[4] / 2, b_dy_half = box_b[4] / 2;
    float a_x1 = box_a[0] - a_dx_half, a_y1 = box_a[1] - a_dy_half;
    float a_x2 = box_a[0] + a_dx_half, a_y2 = box_a[1] + a_dy_half;
    float b_x1 = box_b[0] - b_dx_half, b_y1 = box_b[1] - b_dy_half;
    float b_x2 = box_b[0] + b_dx_half, b_y2 = box_b[1] + b_dy_half;

    Point center_a(box_a[0], box_a[1]);
    Point center_b(box_b[0], box_b[1]);

#ifdef DEBUG
    printf("a: (%.3f, %.3f, %.3f, %.3f, %.3f), b: (%.3f, %.3f, %.3f, %.3f, %.3f)\n", a_x1, a_y1, a_x2, a_y2, a_angle,
        b_x1, b_y1, b_x2, b_y2, b_angle);
    printf("center a: (%.3f, %.3f), b: (%.3f, %.3f)\n", center_a.x, center_a.y, center_b.x, center_b.y);
#endif

    Point box_a_corners[5];
    box_a_corners[0].set(a_x1, a_y1);
    box_a_corners[1].set(a_x2, a_y1);
    box_a_corners[2].set(a_x2, a_y2);
    box_a_corners[3].set(a_x1, a_y2);

    Point box_b_corners[5];
    box_b_corners[0].set(b_x1, b_y1);
    box_b_corners[1].set(b_x2, b_y1);
    box_b_corners[2].set(b_x2, b_y2);
    box_b_corners[3].set(b_x1, b_y2);

    // get oriented corners
    float a_angle_cos = cos(a_angle), a_angle_sin = sin(a_angle);
    float b_angle_cos = cos(b_angle), b_angle_sin = sin(b_angle);

    for (int k = 0; k < 4; k++) {
#ifdef DEBUG
        printf("before corner %d: a(%.3f, %.3f), b(%.3f, %.3f) \n", k, box_a_corners[k].x, box_a_corners[k].y,
            box_b_corners[k].x, box_b_corners[k].y);
#endif
        rotate_around_center(center_a, a_angle_cos, a_angle_sin, box_a_corners[k]);
        rotate_around_center(center_b, b_angle_cos, b_angle_sin, box_b_corners[k]);
#ifdef DEBUG
        printf("corner %d: a(%.3f, %.3f), b(%.3f, %.3f) \n", k, box_a_corners[k].x, box_a_corners[k].y,
            box_b_corners[k].x, box_b_corners[k].y);
#endif
    }

    box_a_corners[4] = box_a_corners[0];
    box_b_corners[4] = box_b_corners[0];

    // get intersection of lines
    Point cross_points[16];
    Point poly_center;
    int cnt = 0, flag = 0;

    poly_center.set(0, 0);
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            flag = intersection(
                    box_a_corners[i + 1], box_a_corners[i], box_b_corners[j + 1], box_b_corners[j], cross_points[cnt]);
            if (flag) {
                poly_center = poly_center + cross_points[cnt];
                cnt++;
#ifdef DEBUG
                printf("Cross points (%.3f, %.3f): a(%.3f, %.3f)->(%.3f, %.3f), b(%.3f, %.3f)->(%.3f, %.3f) \n",
                    cross_points[cnt - 1].x, cross_points[cnt - 1].y, box_a_corners[i].x, box_a_corners[i].y,
                    box_a_corners[i + 1].x, box_a_corners[i + 1].y, box_b_corners[i].x, box_b_corners[i].y,
                    box_b_corners[i + 1].x, box_b_corners[i + 1].y);
#endif
            }
        }
    }

    // check corners
    for (int k = 0; k < 4; k++) {
        if (check_in_box2d(box_a, box_b_corners[k])) {
            poly_center = poly_center + box_b_corners[k];
            cross_points[cnt] = box_b_corners[k];
            cnt++;
#ifdef DEBUG
            printf("b corners in a: corner_b(%.3f, %.3f)", cross_points[cnt - 1].x, cross_points[cnt - 1].y);
#endif
        }
        if (check_in_box2d(box_b, box_a_corners[k])) {
            poly_center = poly_center + box_a_corners[k];
            cross_points[cnt] = box_a_corners[k];
            cnt++;
#ifdef DEBUG
            printf("a corners in b: corner_a(%.3f, %.3f)", cross_points[cnt - 1].x, cross_points[cnt - 1].y);
#endif
        }
    }

    poly_center.x /= cnt;
    poly_center.y /= cnt;

    // sort the points of polygon
    Point temp;
    for (int j = 0; j < cnt - 1; j++) {
        for (int i = 0; i < cnt - j - 1; i++) {
            if (point_cmp(cross_points[i], cross_points[i + 1], poly_center)) {
                temp = cross_points[i];
                cross_points[i] = cross_points[i + 1];
                cross_points[i + 1] = temp;
            }
        }
    }

#ifdef DEBUG
    printf("cnt=%d\n", cnt);
    for (int i = 0; i < cnt; i++)
    {
        printf("All cross point %d: (%.3f, %.3f)\n", i, cross_points[i].x, cross_points[i].y);
    }
#endif

    // get the overlap areas
    float area = 0;
    for (int k = 0; k < cnt - 1; k++) {
        area += cross(cross_points[k] - cross_points[0], cross_points[k + 1] - cross_points[0]);
    }

    return fabs(area) / 2.0;
}

__device__ inline float iou_bev(const float *box_a, const float *box_b) {
    // params box_a: [x, y, z, dx, dy, dz, heading]
    // params box_b: [x, y, z, dx, dy, dz, heading]
    float sa = box_a[3] * box_a[4];
    float sb = box_b[3] * box_b[4];
    float s_overlap = box_overlap(box_a, box_b);
    return s_overlap / fmaxf(sa + sb - s_overlap, EPS);
}

size_t sortTempWorkSpaceSize(int batch_size, int box_nums) {
    size_t sortedWorkspaceSize = 0;
    cub::DoubleBuffer<float> keysDB(nullptr, nullptr);
    cub::DoubleBuffer<uint32_t> valuesDB(nullptr, nullptr);
    cub::DeviceSegmentedRadixSort::SortPairsDescending(nullptr, sortedWorkspaceSize, keysDB, valuesDB,
                                                       batch_size * box_nums, batch_size, (const int *) nullptr,
                                                       (const int *) nullptr);
    return sortedWorkspaceSize;
}

__global__ void filter_by_score_kernel(const float *__restrict__ scoresInput, float *__restrict__ validScores,
                                       uint32_t *__restrict__ validIndices, uint32_t *__restrict__ validNums,
                                       float score_threshold, int batch_size,
                                       int box_nums) {
    uint32_t batch_id = blockIdx.y;
    uint32_t box_id_in_batch = blockDim.x * blockIdx.x + threadIdx.x;
    if (batch_id >= batch_size || box_id_in_batch >= box_nums) {
        return;
    }

    auto box_id = batch_id * box_nums + box_id_in_batch;
    auto score = scoresInput[box_id];
    if (score > score_threshold) {
        auto ind = batch_id * box_nums + atomicAdd(validNums + batch_id, 1);
        validIndices[ind] = box_id_in_batch;
        validScores[ind] = score;
    }
}

__global__ void valid_batch_segments_kernel(const uint32_t *__restrict__ validNums,
                                            uint32_t *__restrict__ validIndStart, uint32_t *__restrict__ validIndEnd,
                                            int batch_size, int box_nums) {
    auto batch_id = threadIdx.x;
    if (batch_id > batch_size) {
        return;
    }
    validIndStart[batch_id] = batch_id * box_nums;
    validIndEnd[batch_id] = batch_id * box_nums + validNums[batch_id];
}

__global__ void nms_kernel(const float *__restrict__ boxesInput, const float *__restrict__ sortedScores,
                           const uint32_t *__restrict__ sortedIndices, const uint32_t *__restrict__ validNums,
                           float *__restrict__ boxesOutput,
                           float *__restrict__ scoresOutput, uint32_t *__restrict__ numsOutput, float iou_threshold,
                           int batch_size,
                           int box_nums, int box_dims, int max_nms_num) {
    auto batch_id = blockIdx.y;
    auto tile_id = threadIdx.x;
    unsigned int tile_num = blockDim.x;
    uint32_t validBoxNum = validNums[batch_id];
    if (tile_id >= validBoxNum) {
        return;
    }
    __shared__ int blockState;
    __shared__ unsigned int output_num;
    if (tile_id == 0) {
        blockState = NMS_STATE_READY;
        output_num = 0;
    }

    int thread_state[NMS_TILES];
    uint32_t thread_box_id_in_batch[NMS_TILES];
    float thread_box_scores[NMS_TILES];
    float thread_boxes[NMS_TILES][BOX_DIMS_MAX];

    uint32_t tile_size = (validBoxNum + tile_num - 1) / tile_num;
    for (int tile = 0; tile < tile_size; tile++) {
        thread_state[tile] = BOX_UNPROCESSED;
        auto sorted_box_id_in_batch = tile_id + tile * blockDim.x;
        if (sorted_box_id_in_batch < tile_num) {
            auto sorted_box_id = batch_id * box_nums + sorted_box_id_in_batch;
            auto input_box_id = batch_id * box_nums + sortedIndices[sorted_box_id];
            thread_box_id_in_batch[tile] = sorted_box_id_in_batch;
            thread_box_scores[tile] = sortedScores[sorted_box_id];
            for (int box_code_i = 0; box_code_i < box_dims; ++box_code_i) {
                thread_boxes[tile][box_code_i] = boxesInput[input_box_id * box_dims + box_code_i];
            }
        }
#ifdef DEBUG1
        printf("%f, (%f, %f, %f, %f, %f, %f, %f, %f, %f)\n", thread_box_scores[tile], thread_boxes[tile][0],
            thread_boxes[tile][1], thread_boxes[tile][2], thread_boxes[tile][3], thread_boxes[tile][4],
            thread_boxes[tile][5], thread_boxes[tile][6], thread_boxes[tile][7], thread_boxes[tile][8]);
#endif
    }

    // Iterate through all boxes to NMS against.
    for (int i = 0; i < validBoxNum; i++) // 每次循环所有线程同步一次。
    {
        int tile = i / tile_num;

        if (thread_box_id_in_batch[tile] == i) // 遇到本thread负责的5个box之一： tid,..,tid+5tile_size，不同线程互不干涉
        {
            if (thread_state[tile] == BOX_PROCESSED) {
                blockState = NMS_SKIP_ITERATION; // -1 => Signal all threads to skip iteration
            } else if (thread_state[tile] == BOX_UNPROCESSED) {
                if (output_num >= max_nms_num) {
                    blockState = NMS_EARLY_STOP; // -2 => Signal all threads to do an early loop exit.
                } else {
                    blockState
                            = NMS_NEED_IOU; // +1 => Signal all (higher index) threads to calculate IOU against this box
                    thread_state[tile]
                            = BOX_NEED_SAVE; // +1 => Mark this box's thread to be kept and written out to results
                    ++output_num;
                    auto output_box_id = batch_id * max_nms_num + output_num - 1;
                    scoresOutput[output_box_id] = thread_box_scores[tile];
                    for (int box_code_i = 0; box_code_i < box_dims; ++box_code_i) {
                        boxesOutput[output_box_id * box_dims + box_code_i] = thread_boxes[tile][box_code_i];
                    }
                    numsOutput[batch_id] = output_num;
                }
            } else {
                blockState = NMS_STATE_READY; // 0 => Signal all threads to not do any updates, nothing happens.
            }
        }

        __syncthreads();

        if (blockState == NMS_EARLY_STOP) {
            // This is the signal to exit from the loop.
            return;
        }

        if (blockState == NMS_SKIP_ITERATION) {
            // This is the signal for all threads to just skip this iteration, as no IOU's need to be checked.
            continue;
        }

        auto test_sorted_box_id = batch_id * box_nums + i;
        auto test_input_box_id = batch_id * box_nums + sortedIndices[test_sorted_box_id];
        auto test_score = sortedScores[test_sorted_box_id];
        auto test_box = boxesInput + test_input_box_id * box_dims;

        for (int tile = 0; tile < tile_size; tile++) {
            // 1. Make sure two different boxes are being tested, and that it's a higher index;
            // 2. Signal that allows IOU checks to be performed;
            // 3. Make sure this box hasn't been either dropped or kept  already;
            // 4. Make sure the sorting order of scores is as expected;
            // 5. And... IOU overlap.
            if (thread_box_id_in_batch[tile] > i && blockState == NMS_NEED_IOU && thread_state[tile] == BOX_UNPROCESSED
                && thread_box_scores[tile] < test_score && iou_bev(thread_boxes[tile], test_box) >= iou_threshold) {
                thread_state[tile] = BOX_PROCESSED; // -1 => Mark this box's thread to be dropped.
            }
        }
    }
}

void nms3DInference(const float *const boxesInput, const float *const scoresInput,
                    float *const boxesOutput, float *const scoresOutput, uint32_t *const numsOutput,
                    float score_threshold, float iou_threshold, int max_nms_num,
                    int batch_size, int box_nums, int box_dims, size_t sortTempSize,
                    uint32_t *const validIndices, float *const validScores, uint32_t *const validNums,
                    uint32_t *const validIndStart, uint32_t *const validIndEnd,
                    uint8_t *sortTempWorkspace, uint32_t *const sortedIndices, float *const sortedScores,
                    cudaStream_t stream) {
    PLUGIN_CUASSERT(cudaMemsetAsync(boxesOutput, 0x00, batch_size * max_nms_num * box_dims * sizeof(float), stream));
    PLUGIN_CUASSERT(cudaMemsetAsync(scoresOutput, 0x00, batch_size * max_nms_num * sizeof(float), stream));
    PLUGIN_CUASSERT(cudaMemsetAsync(numsOutput, 0x00, batch_size * sizeof(int), stream));

    size_t total_box_num = batch_size * box_nums;


    cudaMemsetAsync(validNums, 0x00, batch_size * sizeof(*validNums), stream);

    auto gridSize = BLOCKS2D(box_nums, batch_size), blockSize = THREADS();
    filter_by_score_kernel<<<gridSize, blockSize, 0, stream>>>(
            scoresInput, validScores, validIndices, validNums, score_threshold, batch_size, box_nums);
    valid_batch_segments_kernel<<<1, batch_size, 0, stream>>>(
            validNums, validIndStart, validIndEnd, batch_size, box_nums);

#ifdef DEBUG1
    cudaStreamSynchronize(stream);
    printf("%f\n", score_threshold);
    print((float*) scoresInput, {batch_size, box_nums, 1}, "scoresInput");
    print(validNums, {batch_size, 1}, "validNums");
#endif

    cub::DoubleBuffer<float> scoresDB(validScores, sortedScores);
    cub::DoubleBuffer<uint32_t> indicesDB(validIndices, sortedIndices);
    cub::DeviceSegmentedRadixSort::SortPairsDescending(sortTempWorkspace, sortTempSize, scoresDB, indicesDB,
                                                       total_box_num, batch_size, validIndStart, validIndEnd, 0,
                                                       sizeof(float) * 8, stream);

#ifdef DEBUG1
    cudaStreamSynchronize(stream);
    print((int*) sortedIndices, {batch_size, box_nums, 1}, "sortedIndices");
    print(sortedScores, {batch_size, box_nums, 1}, "sortedScores");
#endif

    unsigned int tile_num = box_nums / NMS_TILES;
    if (box_nums <= 512) {  // bug fix: origin is total_box_nums, which causes invalid memory access when bs > 2
        tile_num = 512;
    }
    if (box_nums <= 256) {
        tile_num = 256;
    }

    blockSize = {tile_num, 1, 1};
    gridSize = {1, static_cast<unsigned int>(batch_size), 1};

#ifdef DEBUG1
    cudaStreamSynchronize(stream);
    printf("%d\n", box_dims);
    print((float*) boxesInput, {batch_size, box_nums, 9, 1}, "boxesInput");
#endif
    nms_kernel<<<gridSize, blockSize, 0, stream>>>(
            boxesInput, scoresDB.Current(), indicesDB.Current(), validNums,
            boxesOutput, scoresOutput, numsOutput, iou_threshold, batch_size,
            box_nums, box_dims, max_nms_num);
}
