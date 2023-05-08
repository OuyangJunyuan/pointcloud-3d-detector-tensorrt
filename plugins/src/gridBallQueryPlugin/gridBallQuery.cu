//
// Created by nrsl on 23-5-2.
//
#include "../havSamplingPlugin/hash.cuh"

// TODO: unroll the FOR loops
__global__
void ScatterSourceToNeighborQueries(const uint32_t num_src, const uint32_t num_qry,
                                    const uint32_t num_nei, const uint32_t num_hash,
                                    const float search_radius, const float3 (*__restrict__ voxels)[3],
                                    const uint32_t *__restrict__ hash_tables, const uint32_t *__restrict__ coord2query,
                                    const float3 *__restrict__ queries, const float3 *__restrict__ sources,
                                    uint32_t *__restrict__ queried_ids, uint32_t *__restrict__ num_queried) {

    const uint32_t bid = blockIdx.y;
    const uint32_t sid = blockDim.x * blockIdx.x + threadIdx.x;
    if (sid >= num_src) {
        return;
    }

    const auto queries_local = queries + num_qry * bid;
    const auto hash_tables_local = hash_tables + num_hash * bid;
    const auto coord2query_local = coord2query + num_hash * bid;

    const auto num_queried_local = num_queried + num_qry * bid;  // int[B,M]
    const auto queried_ids_local = queried_ids + num_qry * num_nei * bid;  // long[B,M,S]

    const auto src = sources[num_src * bid + sid];
    const auto voxel = voxels[bid][1];

    const auto coord_x = (int) roundf(src.x / voxel.x);
    const auto coord_y = (int) roundf(src.y / voxel.y);
    const auto coord_z = (int) roundf(src.z / voxel.z);
    const auto step_x = (int) ceilf(search_radius / voxel.x + 0.5f) - 1;
    const auto step_y = (int) ceilf(search_radius / voxel.y + 0.5f) - 1;
    const auto step_z = (int) ceilf(search_radius / voxel.z + 0.5f) - 1;
    const auto coord_min = int3{coord_x - step_x, coord_y - step_y, coord_z - step_z};
    const auto coord_max = int3{coord_x + step_x, coord_y + step_y, coord_z + step_z};

    const uint32_t kHashMax = num_hash - 1;
    const float r2 = search_radius * search_radius;

    for (int q_grid_z = coord_min.z; q_grid_z <= coord_max.z; ++q_grid_z) {
        for (int q_grid_y = coord_min.y; q_grid_y <= coord_max.y; ++q_grid_y) {
            for (int q_grid_x = coord_min.x; q_grid_x <= coord_max.x; ++q_grid_x) {

                const uint32_t hash_key = coord_hash_32(q_grid_x, q_grid_y, q_grid_z);
                uint32_t hash_slot = hash_key & kHashMax;

                while (true) {
                    if (hash_tables_local[hash_slot] == kEmpty) {  // empty neighbour voxel.
                        break;
                    }
                    if (hash_tables_local[hash_slot] == hash_key) {  // hit a non-empty neighbour voxel.
                        // it takes 80% the runtime of this kernel to achieve here,
                        // cause by accessing global memory hash_tables_local.
                        const auto qid = coord2query_local[hash_slot];
                        if (qid > num_qry) {
                            break;
                        }
                        const auto offset = src - queries_local[qid];
                        const auto d2 = offset.x * offset.x + offset.y * offset.y + offset.z * offset.z;
                        if (d2 > r2 or num_queried_local[qid] > num_nei) {
                            break;
                        }
                        const auto nid = atomicAdd(num_queried_local + qid, 1);
                        if (nid >= num_nei) {
                            break;
                        }
                        queried_ids_local[qid * num_nei + nid] = sid;
                        break;
                    }
                    hash_slot = (hash_slot + 1) & kHashMax;
                }
            }
        }
    }
}

__global__
void PadQueriedNeighbors(const uint32_t num_qry, const uint32_t num_nei,
                         uint32_t *__restrict__ queried_ids, uint32_t *__restrict__ num_queried) {
    const uint32_t bid = blockIdx.y;
    const uint32_t qid = blockDim.x * blockIdx.x + threadIdx.x;
    if (qid >= num_qry)
        return;

    auto &num = num_queried[num_qry * bid + qid];
    num = min(num, num_nei);

    const auto neighbors = queried_ids + num_qry * num_nei * bid + num_nei * qid;  // [B,M,S]
    if (num) {
        for (int l = 0, i = num; i < num_nei; ++l, ++i) {
            neighbors[i] = neighbors[l];
        }
    }
}

void QueryByPointHashingBatchLauncher(const uint32_t num_batch, const uint32_t num_src,
                                      const uint32_t num_qry, const uint32_t num_nei,
                                      const uint32_t num_hash, const float search_radius, const float3 (*voxels)[3],
                                      const uint32_t *hash_tables, const uint32_t *coord2query,
                                      const float3 *queries, const float3 *sources,
                                      uint32_t *queried_ids, uint32_t *num_queried,
                                      cudaStream_t stream = nullptr) {

    const auto src_blocks = BLOCKS2D(num_src, num_batch);
    const auto qry_blocks = BLOCKS2D(num_qry, num_batch);
    const auto threads = THREADS();

    ScatterSourceToNeighborQueries<<<src_blocks, threads, 0, stream>>>(
            num_src, num_qry, num_nei, num_hash, search_radius,
            voxels, hash_tables, coord2query,
            queries, sources, queried_ids, num_queried
    );
    PadQueriedNeighbors<<<qry_blocks, threads, 0, stream>>>(
            num_qry, num_nei,
            queried_ids, num_queried
    );
}