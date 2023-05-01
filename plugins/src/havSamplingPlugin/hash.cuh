//
// Created by nrsl on 23-5-1.
//

#ifndef POINT_DETECTION_HASH_CUH
#define POINT_DETECTION_HASH_CUH

#include <cmath>
#include <cinttypes>

#include <cuda_runtime_api.h>

#define THREAD_SIZE 256
#define BLOCKS1D(M) dim3(((M)+THREAD_SIZE-1)/THREAD_SIZE)
#define BLOCKS2D(M, B) dim3((((M)+THREAD_SIZE-1)/THREAD_SIZE),B)
#define THREADS() dim3(THREAD_SIZE)


constexpr uint32_t kEmpty = 0xffffffff;

__device__ __forceinline__
void atomicMin(float *address, float val) {
    int *address_as_i = (int *) address;
    int old = *address_as_i;
    int assumed;

    do {
        assumed = old;
        old = atomicCAS(address_as_i, assumed, __float_as_int(min(val, __int_as_float(assumed))));
    } while (assumed != old);  // fail to insert the min val since *address_as_i was changed before atomicCAS execution.
//    return __int_as_float(old);
}

__device__ __forceinline__
float3 operator+(const float3 a, const float3 b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__device__ __forceinline__
float3 operator-(const float3 a, const float3 b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__device__ __forceinline__
float3 operator/(const float3 a, const float3 b) {
    return make_float3(a.x / b.x, a.y / b.y, a.z / b.z);
}

__device__ __forceinline__
float3 operator/(const float3 a, const float b) {
    return make_float3(a.x / b, a.y / b, a.z / b);
}

__device__ __forceinline__
uint32_t coord_hash_32(const int x, const int y, const int z) {
    uint32_t hash = 2166136261;
    hash ^= (uint32_t) (x + 10000);
    hash *= 16777619;
    hash ^= (uint32_t) (y + 10000);
    hash *= 16777619;
    hash ^= (uint32_t) (z + 10000);
    hash *= 16777619;
    return hash;
}

__device__ __forceinline__
uint32_t coord_hash_32(const int b, const int x, const int y, const int z) {
    uint32_t hash = 2166136261;
    hash ^= (uint32_t) (b + 10000);
    hash *= 16777619;
    hash ^= (uint32_t) (x + 10000);
    hash *= 16777619;
    hash ^= (uint32_t) (y + 10000);
    hash *= 16777619;
    hash ^= (uint32_t) (z + 10000);
    hash *= 16777619;
    return hash;
}


inline auto get_table_size(int64_t nums, int64_t min = 2048) {
    auto table_size = nums * 2 > min ? nums * 2 : min;
    table_size = 2 << ((int64_t) ceilf((logf(static_cast<float>(table_size)) / logf(2.0f))) - 1);
    return table_size;
}

#endif //POINT_DETECTION_HASH_CUH
