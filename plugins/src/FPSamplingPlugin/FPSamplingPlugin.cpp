#include "FPSampling.h"
#include "FPSamplingPlugin.h"


namespace nvinfer1::plugin {

int32_t FPSamplingPlugin::initialize() noexcept { return 0; }

void FPSamplingPlugin::terminate() noexcept {}

int32_t FPSamplingPlugin::enqueue(nvinfer1::PluginTensorDesc const *inputDesc,
                                  nvinfer1::PluginTensorDesc const *outputDesc, cudaStream_t stream) noexcept {

    const auto batch_size = inputDesc[0].dims.d[0];
    const auto num_src = inputDesc[0].dims.d[1];
    const auto num_trg = outputDesc[0].dims.d[1];

    farthest_point_sampling_kernel_launcher(
            stream, batch_size, num_src, num_trg, xyz, furthest_dists, indices
    );
    return 0;

}

} // namespace nvinfer1::plugin
