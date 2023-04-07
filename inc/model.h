//
// Created by nrsl on 23-4-3.
//

#ifndef point_detection_TRT_MODELS_HPP
#define point_detection_TRT_MODELS_HPP

#include <dlfcn.h>

#include <memory>
#include <utility>
#include <memory>
#include <iostream>
#include <boost/filesystem.hpp>

#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <yaml-cpp/yaml.h>

#include "inc/buffer.h"
#include "inc/helper.h"
#include "inc/calibrator.h"

namespace PointDetection {
inline std::map<std::string, nvinfer1::ILogger::Severity>
        str2severity = {{"INTERNAL_ERROR ", nvinfer1::ILogger::Severity::kINTERNAL_ERROR},
                        {"ERROR",           nvinfer1::ILogger::Severity::kERROR},
                        {"WARNING",         nvinfer1::ILogger::Severity::kWARNING},
                        {"INFO",            nvinfer1::ILogger::Severity::kINFO},
                        {"VERBOSE",         nvinfer1::ILogger::Severity::kVERBOSE}};

class Plugins {
 public:
    explicit Plugins(const YAML::Node &configs) {
        for (auto &&plugin: configs) {
            auto name = canonical(plugin.as<std::string>()).string();
            fprintf(stderr, "loading %s ...\n", name.c_str());
            auto handle = dlopen(name.c_str(), RTLD_NOW | RTLD_GLOBAL);
            if (handle) {
                handles.push_back(handle);
            } else {
                fprintf(stderr, "failed to load library %s\n", name.c_str());
            }
        }
#ifdef DEBUG
        int num_plugins = 0;
        auto plugins = nvinfer1::getPluginRegistry(
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

inline class Logger : public nvinfer1::ILogger {
 public:
    Severity severity{Severity::kWARNING};

    void log(Severity severity, const char *msg) noexcept override {
        if (severity <= this->severity)
            std::cout << msg << std::endl;
    }
} logger_;

class TRTDetector3D {
 public:
    explicit TRTDetector3D(const YAML::Node &cfgs) :
            cfgs_(cfgs),
            max_batch_size_(cfgs["max_batch_size"].as<int>()),
            plugins_(new Plugins(cfgs["plugins"])) {

        LoadTensorrtEngine();
    }

    auto max_point() {
        return max_point_num_;
    }
    auto max_batch() {
        return max_batch_size_;
    }
    auto operator()(const std::vector<void *> &inputs) {
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

    void BuildEngineFromOnnx(const boost::filesystem::path &engine_file) {
        const auto onnx_file = canonical(cfgs_["build"]["onnx"].as<std::string>());
        fprintf(stderr, "build engine from onnx file: %s\n", onnx_file.c_str());

        auto builder = std::unique_ptr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(logger_));
        assert(builder);

        int explicit_batch = 1U << static_cast<int>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
        auto network = std::unique_ptr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicit_batch));
        assert(network);

        auto parser = std::unique_ptr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, logger_));
        assert(parser);


        auto verbosity = static_cast<int32_t>(logger_.severity);
        parser->parseFromFile(onnx_file.c_str(), verbosity);
        for (int32_t i = 0; i < parser->getNbErrors(); ++i) {
            fprintf(stderr, "%s\n", parser->getError(i)->desc());
        }
        max_point_num_ = network->getInput(0)->getDimensions().d[1];

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

        std::unique_ptr<Int8EntropyCalibrator2> calibrator{nullptr};
        auto quan = cfgs_["build"]["quan"].as<std::string>();
        if (quan == "fp32") {
            fprintf(stderr, "build TensorRT engine on FP32 mode\n");
        }
        if (quan == "fp16") {
            fprintf(stderr, "build TensorRT engine on FP16 mode\n");
            config->setFlag(nvinfer1::BuilderFlag::kFP16);
        } else if (quan == "int8") {
            auto calib_cfgs = cfgs_["build"]["calib"];
            if (calib_cfgs["enable"].as<bool>()) {
                fprintf(stderr, "build TensorRT engine on INT8 mode with calibration!\n");
                auto dims = network->getInput(0)->getDimensions();
                calibrator = std::make_unique<Int8EntropyCalibrator2>(calib_cfgs, onnx_file, dims, 1);
                config->setInt8Calibrator(calibrator.get());
                auto calibration_profile = builder->createOptimizationProfile();
                calibration_profile->setDimensions(input_points_name, nvinfer1::OptProfileSelector::kMIN,
                                                   nvinfer1::Dims3{1, max_point_num_, 4});
                calibration_profile->setDimensions(input_points_name, nvinfer1::OptProfileSelector::kOPT,
                                                   nvinfer1::Dims3{1, max_point_num_, 4});
                calibration_profile->setDimensions(input_points_name, nvinfer1::OptProfileSelector::kMAX,
                                                   nvinfer1::Dims3{max_batch_size_, max_point_num_, 4});
                config->setCalibrationProfile(calibration_profile);
            } else {
                fprintf(stderr, "build TensorRT engine on INT8 mode without calibration!\n");
            }
            config->setFlag(nvinfer1::BuilderFlag::kINT8);
        }

        auto serialized_model
                = std::unique_ptr<nvinfer1::IHostMemory>(builder->buildSerializedNetwork(*network, *config));

        std::ofstream p(engine_file.c_str(), std::ios::binary);
        p.write(reinterpret_cast<const char *>(serialized_model->data()), serialized_model->size());
        p.close();
        printf("save to engine file: %s\n", engine_file.c_str());
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
            fprintf(stderr, "invalid engine file: %s\n", engine_file.c_str());
            BuildEngineFromOnnx(engine_file);
            ReadEngineFile(engine_file.string(), &engine_data);
        }

        auto runtime = std::unique_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(logger_));
        assert(runtime);

        engine_ = std::unique_ptr<nvinfer1::ICudaEngine>(
                runtime->deserializeCudaEngine(engine_data.data(), engine_data.size())
        );
        assert(engine_);
        max_point_num_ = engine_->getBindingDimensions(0).d[1];
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
    int max_batch_size_;
    int max_point_num_;
};

}  // namespace PointDetection

#endif  // point_detection_TRT_MODELS_HPP
