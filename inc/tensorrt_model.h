//
// Created by nrsl on 23-4-3.
//

#ifndef point_detection_TRT_MODELS_HPP
#define point_detection_TRT_MODELS_HPP

#include <dlfcn.h>

#include <memory>
#include <utility>
#include <memory>
#include <chrono>
#include <iostream>
#include <boost/filesystem.hpp>

#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <yaml-cpp/yaml.h>

#include "inc/buffer_helper.h"

inline auto canonical(const boost::filesystem::path &path) {
    return path.is_absolute() ? path : boost::filesystem::weakly_canonical(PROJECT_ROOT / path);
}

namespace PointDetection {
class Plugins {
 public:
    explicit Plugins(const YAML::Node &configs) {
        for (auto &&plugin: configs) {
            auto name_s = plugin.as<std::string>();
            auto name = name_s.data();
            fprintf(stderr, "loading %s ...\n", name);
            auto handle = dlopen(name, RTLD_NOW | RTLD_GLOBAL);
            if (handle) {
                handles.push_back(handle);
            } else {
                fprintf(stderr, "failed to load library %s\n", name);
            }
        }
#ifdef DEBUG1
        int num_plugins = 0;
        auto plugins = nvinfer1::getBuilderPluginRegistry(
                nvinfer1::EngineCapability::kSTANDARD)->getPluginCreatorList(
                &num_plugins);
        for (int i = 0; i < num_plugins; ++i) {
            fprintf(stderr, "Found plugin %s\n", plugins[i]->getPluginName());
        }
#endif
    }

    ~Plugins() {
        for (auto &&handle: handles) {
            dlclose(handle);
        }
    }

 private:
    std::vector<void *> handles;
};

class Logger : public nvinfer1::ILogger {
 public:
    void log(Severity severity, const char *msg) noexcept override {
        if (severity <= Severity::kWARNING)
            std::cout << msg << std::endl;
    }
} logger_;

class TRTDetector3D {
 public:
    explicit TRTDetector3D(const YAML::Node &cfgs) :
            cfgs_(cfgs),
            max_batch_size_(cfgs["max_batch_size"].as<int>()),
            max_point_num_(cfgs["max_point_num"].as<int>()),
            plugins_(new Plugins(cfgs["plugins"])) {

        LoadTensorrtEngine();
    }

    auto Infer(const std::vector<void *> &inputs) {
        buffers_->SetInputs(inputs);
        buffers_->ToDevice(stream_);
        context_->enqueueV2(buffers_->IO(), stream_, nullptr);
        buffers_->ToHost(stream_);
        cudaStreamSynchronize(stream_);

        return std::make_tuple(buffers_->ReadOutput<float, 8>(0),
                               buffers_->ReadOutput<float, 1>(1),
                               buffers_->ReadOutput<int, 1>(2));
    }

 private:

    void BuildEngineFromOnnx(const boost::filesystem::path &onnx_file,
                             const boost::filesystem::path &engine_file) const {
        auto builder = std::unique_ptr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(logger_));
        assert(builder);

        auto network = std::unique_ptr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(
                1U << static_cast<int>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH)));
        assert(network);

        auto parser = std::unique_ptr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, logger_));
        assert(parser);

        parser->parseFromFile(onnx_file.c_str(), static_cast<int32_t>(nvinfer1::ILogger::Severity::kWARNING));
        for (int32_t i = 0; i < parser->getNbErrors(); ++i) {
            fprintf(stderr, "%s\n", parser->getError(i)->desc());
        }

        auto config = std::unique_ptr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
        assert(config);

        config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1U << 31);

        auto profile = builder->createOptimizationProfile();
        auto input_points_name = network->getInput(0)->getName();
        profile->setDimensions(input_points_name, nvinfer1::OptProfileSelector::kMIN,
                               nvinfer1::Dims3{1, max_point_num_, 4});
        profile->setDimensions(input_points_name, nvinfer1::OptProfileSelector::kOPT,
                               nvinfer1::Dims3{1, max_point_num_, 4});
        profile->setDimensions(input_points_name, nvinfer1::OptProfileSelector::kMAX,
                               nvinfer1::Dims3{max_batch_size_, max_point_num_, 4});
        config->addOptimizationProfile(profile);

#if TRT_QUANTIZE == TRT_FP16
        printf("build TensorRT engine on FP16 mode\n");
        config->setFlag(nvinfer1::BuilderFlag::kFP16);
#elif TRT_QUANTIZE == TRT_INT8
        std::unique_ptr<Int8EntropyCalibrator2> calibrator(new Int8EntropyCalibrator2(
                Path("/media/nrsl/NRSL12YEARS/dataset/kitti/data/training/velodyne"),
                Path(PROJECT_ROOT "/config/calibration.cache"),
                network->getInput(0)->getDimensions(), 1
        ));
        printf("build TensorRT engine on INT8 mode\n");
        config->setFlag(BuilderFlag::kINT8);
        config->setInt8Calibrator(calibrator.get());
#else
        printf("build TensorRT engine on TF32/FP32 mode\n");
#endif

        auto serializedModel = std::unique_ptr<nvinfer1::IHostMemory>(
                builder->buildSerializedNetwork(*network, *config));

        std::ofstream p(engine_file.c_str(), std::ios::binary);
        if (!p.good()) {
            fprintf(stderr, "could not open engine output: %s\n", engine_file.c_str());
        } else {
            p.write(reinterpret_cast<const char *>(serializedModel->data()), serializedModel->size());
            p.close();
            printf("save to engine file: %s\n", engine_file.c_str());
        }
    }

    template<class T>
    static void SummarizeEngine(const T &engine) {
        std::cout << "====== model infos ======" << std::endl;
        for (int i = 0; i < engine->getNbBindings(); ++i) {
            auto flow = engine->bindingIsInput(i) ? "->" : "<-";
            std::cout << flow << " " << engine->getBindingName(i);
            std::cout << "(";
            auto shape = engine->getBindingDimensions(i);
            for (int j = 0; j < shape.nbDims; ++j) {
                std::cout << shape.d[j] << (j == shape.nbDims - 1 ? ")" : ", ");
            }
            std::cout << std::endl;
        }
        std::cout << "=========================" << std::endl;
    }

    static bool ReadEngineFile(const std::string &file_path, std::vector<char> *data_stream) {
        std::ifstream file(file_path, std::ios::binary);
        if (file.good()) {
            file.seekg(0, std::ifstream::end);
            auto size = file.tellg();
            file.seekg(0, std::ifstream::beg);
            data_stream->resize(size);
            file.read(data_stream->data(), size);
        }
        file.close();
        return file.good();
    }

    void LoadTensorrtEngine() {
        std::vector<char> engine_data;

        const auto engine_file = canonical(cfgs_["engine"].as<std::string>());
        fprintf(stdout, "try loading TensorRT engine file: %s\n", engine_file.c_str());
        auto res = ReadEngineFile(engine_file.string(), &engine_data);
        if (!res) {
            const auto onnx_file = canonical(cfgs_["onnx"].as<std::string>());
            fprintf(stderr, "invalid engine file %s\n", engine_file.c_str());
            BuildEngineFromOnnx(onnx_file, engine_file);
            ReadEngineFile(engine_file.string(), &engine_data);
        }

        auto runtime = std::unique_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(logger_));
        assert(runtime);

        engine_ = std::unique_ptr<nvinfer1::ICudaEngine>(
                runtime->deserializeCudaEngine(engine_data.data(), engine_data.size())
        );
        assert(engine_);

        SummarizeEngine(engine_);
        buffers_ = std::make_unique<BufferManager>(engine_, max_batch_size_);

        context_ = std::unique_ptr<nvinfer1::IExecutionContext>(engine_->createExecutionContext());
        assert(context_);

        context_->setBindingDimensions(0, nvinfer1::Dims3(max_batch_size_, max_point_num_, 4));
        // TODO: multi stream optimization for inference.
        cudaStreamCreate(&stream_);
        context_->setOptimizationProfileAsync(0, stream_);
    }

 private:
    std::unique_ptr<Plugins> plugins_{nullptr};
    std::unique_ptr<BufferManager> buffers_{nullptr};
    std::unique_ptr<nvinfer1::ICudaEngine> engine_{nullptr};
    std::unique_ptr<nvinfer1::IExecutionContext> context_{nullptr};
    cudaStream_t stream_{nullptr};

    YAML::Node cfgs_;
 public:
    const int max_batch_size_;
    const int max_point_num_;
};

}  // namespace PointDetection

#endif  // point_detection_TRT_MODELS_HPP
