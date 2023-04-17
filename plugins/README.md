# AUTO CODES GENERATION
we implement a header `src/common/plugin_auto_declare.h` that can generate the most part of plugin codes by c++ build-in macro mechanism. 

## HOW TO USE
please refer to `src/FPSamplingPlugin`.
Briefly, we just define `TENSORRT_PLUGIN` to describe the inputs and outputs of a plugin like:
```c++
#define TENSORRT_PLUGIN                                                \
Setting(                                                               \
    Name(FPSampling),                                                  \
    Version("1"),                                                      \
    (                                                                  \
        Input(float, xyz, Dim3(num_batch, num_point, 3))               \
    ),                                                                 \
    (                                                                  \
        Output(int32_t, indices, Dim2(Input(0,0), Const(sample_num)))  \
    ),                                                                 \
    (                                                                  \
        Workspace(float, furthest_dists, Dim2(Input(0,0), Input(0,1))) \
    ),                                                                 \
    (                                                                  \
        Attribute(int32_t, sample_num, 1)                              \
    )                                                                  \
)

struct FPSamplingUser { 
    /* your own functions or members */
};

#include "common/plugin_auto_declare.h"
```
Then, only the `Enqueue` need to be implemented:
```c++
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
```

## LIMITATION
1. It is extremely hard to read ...
2. NOT implementation of half or int8 formation are supported.