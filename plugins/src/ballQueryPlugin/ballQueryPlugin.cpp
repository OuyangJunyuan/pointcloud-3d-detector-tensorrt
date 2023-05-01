//
// Created by nrsl on 23-5-1.
//

#include "ballQueryPlugin.h"
#include "ballQuery.cuh"

namespace nvinfer1::plugin {

int32_t BallQueryPlugin::initialize() noexcept { return 0; }

void BallQueryPlugin::terminate() noexcept {}

int32_t BallQueryPlugin::enqueue(cudaStream_t stream) noexcept {

    BallQueryLauncher(
            def.num_batch, def.num_source, def.num_query,
            attr.radius, attr.num_neighbor,
            in.sources.ptr, in.queries.ptr,
            out.num_valid.ptr, out.indices.ptr,
            stream
    );
    return 0;
}

} // namespace nvinfer1::plugin
