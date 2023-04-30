#include "FPSampling.h"
#include "FPSamplingPlugin.h"

namespace nvinfer1::plugin {

int32_t FPSamplingPlugin::initialize() noexcept { return 0; }

void FPSamplingPlugin::terminate() noexcept {}

int32_t FPSamplingPlugin::enqueue(cudaStream_t stream) noexcept {

    farthest_point_sampling_kernel_launcher(
            def.batch, def.source, def.target,
            in.xyz.ptr, ws.furthest_dists.ptr, out.indices.ptr,
            stream
    );
    return 0;
}

} // namespace nvinfer1::plugin
