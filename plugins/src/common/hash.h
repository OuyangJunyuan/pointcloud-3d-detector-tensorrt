//
// Created by nrsl on 23-4-13.
//

#ifndef POINT_DETECTION_HASH_H
#define POINT_DETECTION_HASH_H

#include <cuda.h>
#include <algorithm>

#define THREAD_SIZE 256
#define BLOCKS1D(M) dim3(((M) + THREAD_SIZE - 1) / THREAD_SIZE)
#define BLOCKS2D(M, B) dim3((((M) + THREAD_SIZE - 1) / THREAD_SIZE), B)
#define THREADS() dim3(THREAD_SIZE)

constexpr uint32_t kEmpty = 0xffffffff;

struct Voxel {
    float3 c;
    float3 l;
    float3 r;
};

//__device__ __forceinline__ float3 operator+(float3 a, float3 b);
//
//__device__ __forceinline__ float3 operator/(float3 a, float b);
//
//__device__ __forceinline__ void atomicMin(float *address, float val);
//
//__device__ __forceinline__ uint32_t coord_hash_32(const int x, const int y, const int z);

__device__ __forceinline__ float3 operator+(float3 a, float3 b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__device__ __forceinline__ float3 operator/(float3 a, float b) {
    return make_float3(a.x / b, a.y / b, a.z / b);
}

__device__ __forceinline__ void atomicMin(float *address, float val) {
    int *address_as_i = (int *) address;
    int old = *address_as_i;
    int assumed;

    do {
        assumed = old;
        old = atomicCAS(address_as_i, assumed, __float_as_int(min(val, __int_as_float(assumed))));
    } while (assumed != old);
}

__device__ __forceinline__ uint32_t coord_hash_32(const int x, const int y, const int z) {
    uint32_t hash = 2166136261;
    hash ^= (uint32_t) (x + 10000);
    hash *= 16777619;
    hash ^= (uint32_t) (y + 10000);
    hash *= 16777619;
    hash ^= (uint32_t) (z + 10000);
    hash *= 16777619;
    return hash;
}

inline auto get_table_size(size_t N, size_t min_size) {
    size_t table_size = std::max(min_size, N * 2);
    table_size = (2 << ((size_t) ceil((log((double) table_size) / log(2.0))) - 1));
    return table_size;
}

#endif //POINT_DETECTION_HASH_H
