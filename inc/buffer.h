//
// Created by nrsl on 23-4-5.
//

#ifndef POINT_DETECTION_BUFFER_H
#define POINT_DETECTION_BUFFER_H

#include <cassert>

#include <vector>
#include <memory>
#include <utility>

#include <NvInfer.h>
#include <cuda_runtime_api.h>

namespace PointDetection {
class BufferPair {
 public:
    BufferPair(const nvinfer1::Dims &dims, const nvinfer1::DataType &type, bool is_output = true) : is_output_(
            is_output) {
        bytes = Volume(dims) * SizeOf(type);
        assert(bytes);
        if (is_output) {
            host_ = malloc(bytes);
        }
        cudaMalloc(&device_, bytes);
    }

    inline void ToDevice(const cudaStream_t &stream) const {
        cudaMemcpyAsync(device_, host_, bytes, cudaMemcpyHostToDevice, stream);
    }

    inline void ToHost(const cudaStream_t &stream) const {
        cudaMemcpyAsync(host_, device_, bytes, cudaMemcpyDeviceToHost, stream);
    }

    inline auto &host() {
        return host_;
    }

    inline auto &device() {
        return device_;
    }

    ~BufferPair() {
        if (is_output_) {
            free(host_);
        }
        cudaFree(device_);
    }

    BufferPair(const BufferPair &) = delete;

    BufferPair &operator=(BufferPair &) = delete;


 private:
    static int SizeOf(const nvinfer1::DataType &type) {
        switch (type) {
            case nvinfer1::DataType::kINT32:
            case nvinfer1::DataType::kFLOAT:
                return 4;
            case nvinfer1::DataType::kHALF:
                return 2;
            case nvinfer1::DataType::kBOOL:
            case nvinfer1::DataType::kINT8:
                return 1;
            default:
                assert(false);
        }
    }

    static int Volume(const nvinfer1::Dims &dims) {
        int size = 1;
        for (int i = 0; i < dims.nbDims; ++i) {
            size *= dims.d[i];
        }
        return size;
    }

    int bytes{0};
    void *device_{nullptr};
    void *host_{nullptr};
    bool is_output_{false};
};

class BufferManager {
 public:
    explicit BufferManager(const std::unique_ptr<nvinfer1::ICudaEngine> &engine, int batch_size) {
        for (int i = 0; i < engine->getNbBindings(); ++i) {
            bool is_output = not engine->bindingIsInput(i);
            auto type = engine->getBindingDataType(i);
            auto dims = engine->getBindingDimensions(i);
            if (dims.d[0] == -1) {
                dims.d[0] = batch_size;
            }
            auto buffer = std::make_unique<BufferPair>(dims, type, is_output);
            bindings.emplace_back(buffer->device());
            (is_output ? outputs : inputs).emplace_back(std::move(buffer));
        }
    }

    explicit BufferManager(const nvinfer1::IExecutionContext &context, int batch_size = 1) {
        auto &engine = context.getEngine();
        for (int i = 0; i < engine.getNbBindings(); ++i) {
            bool is_output = not engine.bindingIsInput(i);
            auto type = engine.getBindingDataType(i);
            auto dims = context.getBindingDimensions(i);
            if (dims.d[0] == -1) {
                fprintf(stderr, "dims[0] == -1, please call context.setBindingDimensions\n");
                dims.d[0] = batch_size;
            }
            auto buffer = std::make_unique<BufferPair>(dims, type, is_output);
            bindings.emplace_back(buffer->device());
            (is_output ? outputs : inputs).emplace_back(std::move(buffer));
        }
    }

    void ToDevice(const cudaStream_t &stream) {
        for (auto &input: inputs) {
            input->ToDevice(stream);
        }
    }

    void ToHost(const cudaStream_t &stream) {
        for (auto &output: outputs) {
            output->ToHost(stream);
        }
    }

    template<class T, int N>
    inline const typeof(T[N]) *ReadOutput(int index) {
        return reinterpret_cast<typeof(T[N]) *>(outputs[index]->host());
    }

    inline void SetInputs(const std::vector<void *> &in_ptr) {
        for (int i = 0; i < in_ptr.size(); ++i) {
            inputs[i]->host() = in_ptr[i];
        }
    }

    inline auto *IO() {
        return bindings.data();
    }


 private:
    std::vector<std::unique_ptr<BufferPair>> inputs;
    std::vector<std::unique_ptr<BufferPair>> outputs;
    std::vector<void *> bindings;
};

}  // namespace PointDetection

#endif //POINT_DETECTION_BUFFER_H
