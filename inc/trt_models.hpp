//
// Created by nrsl on 23-4-3.
//

#ifndef point_detection_TRT_MODELS_HPP
#define point_detection_TRT_MODELS_HPP

#include <dlfcn.h>

#include <memory>
#include <chrono>
#include <iostream>

#include <yaml-cpp/yaml.h>
#include <boost/filesystem.hpp>
#include <NvInfer.h>
#include <NvOnnxParser.h>

namespace PointDetection {
using namespace nvinfer1;
using Path = boost::filesystem::path;

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
        int num_plugins = 0;
        auto plugins = nvinfer1::getBuilderPluginRegistry(
                nvinfer1::EngineCapability::kSTANDARD)->getPluginCreatorList(
                &num_plugins);
        for (int i = 0; i < num_plugins; ++i) {
#ifdef DEBUG
            fprintf(stderr, "Found plugin %s\n", plugins[i]->getPluginName());
#endif
        }
    }

    ~Plugins() {
        for (auto &&handle: handles) {
            dlclose(handle);
        }
    }

 private:
    std::vector<void *> handles;
};

class BufferPair {
 public:
    BufferPair(const nvinfer1::Dims &dims, const nvinfer1::DataType &type, bool is_input = false) {
        elem_size = getSize(dims);
        elem_byte = getBytes(type);
        bytes = elem_size * elem_byte;
        assert(bytes);
        if (not is_input) {
            free_host = true;
            host = malloc(bytes);
        }
        cudaMalloc(&device, bytes);
    };

    ~BufferPair() {
        if (free_host) {
            free(host);
        }
        std::cout << "free BufferPair" << std::endl;
        cudaFree(device);
    }

    void moveToDevice(const cudaStream_t &stream) const {
        cudaMemcpyAsync(device, host, bytes, cudaMemcpyHostToDevice, stream);
    }

    void moveToHost(const cudaStream_t &stream) const {
        cudaMemcpyAsync(host, device, bytes, cudaMemcpyDeviceToHost, stream);
    }

 private:
    static int getBytes(const nvinfer1::DataType &type) {
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

    static int getSize(const nvinfer1::Dims &dims) {
        int size = 1;
        for (int i = 0; i < dims.nbDims; ++i) {
            size *= dims.d[i];
        }
        return size;
    }

 public:
    int elem_byte;
    int elem_size;
    int bytes;
    bool free_host{false};
    void *device{nullptr};
    void *host{nullptr};
};

class BufferManager {
 public:
    explicit BufferManager(std::unique_ptr<nvinfer1::ICudaEngine> &engine,
                           const nvinfer1::IExecutionContext *context = nullptr,
                           int batch_size = 0) {
        assert(context || batch_size);

        for (int i = 0; i < engine->getNbBindings(); ++i) {
            auto is_input = engine->bindingIsInput(i);
            auto type = engine->getBindingDataType(i);
            auto dims = context ? context->getBindingDimensions(i) : engine->getBindingDimensions(i);
            dims.d[0] = dims.d[0] == -1 ? batch_size : dims.d[0];

            auto buffers = std::make_unique<BufferPair>(dims, type, is_input);
            bindings.emplace_back(buffers->device);
            if (is_input) {
                inputs.emplace_back(std::move(buffers));
            } else {
                outputs.emplace_back(std::move(buffers));
            }
        }
    }

    void connectInputs(const std::vector<void *> &ptrs) {
        for (int i = 0; i < inputs.size(); ++i) {
            inputs[i]->host = ptrs[i];
        }
    }

    void moveToDevice(const cudaStream_t &stream) {
        for (auto &input: inputs) {
            input->moveToDevice(stream);
        }
    }

    void moveToHost(const cudaStream_t &stream) {
        for (auto &output: outputs) {
            output->moveToHost(stream);
        }
    }

    template<class T, int N>
    auto readOutput(int index) {
        using pT = typeof(T[N]) *;
        return static_cast<pT> ((void *) outputs[index]->host);
    }

 private:
    std::vector<std::unique_ptr<BufferPair>> inputs;
    std::vector<std::unique_ptr<BufferPair>> outputs;
 public:
    std::vector<void *> bindings;
};

class TRTDetector3D {
 public:
    explicit TRTDetector3D(const YAML::Node &configs) {
        load_trt_engine(configs["engine"]);
    }


    auto inference(void *points) {
        auto t1 = std::chrono::steady_clock::now();
        mBuffers->connectInputs({points});
        mBuffers->moveToDevice(stream);
        mContext->enqueueV2(mBuffers->bindings.data(), stream, nullptr);
        mBuffers->moveToHost(stream);
        cudaStreamSynchronize(stream);
        auto t2 = std::chrono::steady_clock::now();
        auto time = std::chrono::duration_cast<std::chrono::milliseconds>((t2 - t1)).count();

        std::cout << "======== outputs ========" << std::endl;
        std::cout << "runtime: " << time << "ms" << std::endl;
//        auto boxes = mBuffers->readOutput<float, 8>(0);
//        auto scores = mBuffers->readOutput<float, 1>(1);
//        auto nums = mBuffers->readOutput<int, 1>(2);
//        for (int i = 0; i < nums[0][0]; ++i) {
//            auto &box = boxes[i];
//            printf("box_%d(%d,%f) [%.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f]\n",
//                   i, int(box[7]), scores[i][0],
//                   box[0], box[1], box[2], box[3], box[4], box[5], box[6]
//            );
//        }
        return std::make_tuple(mBuffers->readOutput<float, 8>(0),
                               mBuffers->readOutput<float, 1>(1),
                               mBuffers->readOutput<int, 1>(2));
    }

 private:

    static bool load_file(const boost::filesystem::path &file_path, std::vector<char> *data_stream) {
        std::ifstream file(file_path.string(), std::ios::binary);
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

    template<class T>
    static void summary_engine(const T &engine) {
        std::cout << "====== model infos ======" << std::endl;
        for (int i = 0; i < engine->getNbBindings(); ++i) {
            auto flow = engine->bindingIsInput(0) ? "->" : "<-";
            std::cout << flow << " " << engine->getBindingName(i);
            std::cout << "(";
            auto shape = engine->getBindingDimensions(0);
            for (int j = 0; j < shape.nbDims; ++j) {
                std::cout << shape.d[j] << (j == shape.nbDims - 1 ? ")" : ", ");
            }
            std::cout << std::endl;
        }
        std::cout << "=========================" << std::endl;
    }

 private:

    void build_engine(const boost::filesystem::path &onnx_file, const boost::filesystem::path &engine_file) {
        auto builder = std::unique_ptr<IBuilder>(createInferBuilder(logger));
        assert(builder);

        auto network = std::unique_ptr<INetworkDefinition>(builder->createNetworkV2(
                1U << static_cast<int>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH)));
        assert(network);

        auto parser = std::unique_ptr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, logger));
        assert(parser);

        parser->parseFromFile(onnx_file.c_str(), static_cast<int32_t>(ILogger::Severity::kWARNING));
        for (int32_t i = 0; i < parser->getNbErrors(); ++i) {
            std::cout << parser->getError(i)->desc() << std::endl;
        }

        auto config = std::unique_ptr<IBuilderConfig>(builder->createBuilderConfig());
        assert(config);

        config->setMemoryPoolLimit(MemoryPoolType::kWORKSPACE, 1U << 31);

        auto profile = builder->createOptimizationProfile();
        auto input_points_name = network->getInput(0)->getName();
        profile->setDimensions(input_points_name, OptProfileSelector::kMIN, Dims3{1, 16384, 4});
        profile->setDimensions(input_points_name, OptProfileSelector::kOPT, Dims3{1, 16384, 4});
        profile->setDimensions(input_points_name, OptProfileSelector::kMAX, Dims3{1, 16384, 4});
        config->addOptimizationProfile(profile);

#if TRT_QUANTIZE == TRT_FP16
        std::cout << "build on FP16 mode" << std::endl;
        config->setFlag(nvinfer1::BuilderFlag::kFP16);
#elif TRT_QUANTIZE == TRT_INT8
        std::unique_ptr<Int8EntropyCalibrator2> calibrator(new Int8EntropyCalibrator2(
                Path("/media/nrsl/NRSL12YEARS/dataset/kitti/data/training/velodyne"),
                Path(PROJECT_ROOT "/config/calibration.cache"),
                network->getInput(0)->getDimensions(), 1
        ));
        std::cout << "build on INT8 mode" << std::endl;
        config->setFlag(BuilderFlag::kINT8);
        config->setInt8Calibrator(calibrator.get());
#endif

        auto serializedModel = std::unique_ptr<nvinfer1::IHostMemory>(
                builder->buildSerializedNetwork(*network, *config));

        std::ofstream p(engine_file.c_str(), std::ios::binary);
        if (!p) { std::cerr << "Could not open plan output file" << std::endl; }
        p.write(reinterpret_cast<const char *>(serializedModel->data()), serializedModel->size());
        p.close();
        std::cout << "output engine file: " << engine_file << std::endl;
    }


    void load_trt_engine(const YAML::Node &engine_cfg) {
        const auto engine_file = boost::filesystem::path{PROJECT_ROOT"/" + engine_cfg.as<std::string>()};

        mRuntime = std::unique_ptr<IRuntime>(nvinfer1::createInferRuntime(logger));
        assert(mRuntime);

        std::cout << "try loading trt engine from " << engine_file << std::endl;

        std::vector<char> fsteam;
        auto res = load_file(engine_file.string(), &fsteam);
        if (not load_file(engine_file.string(), &fsteam)) {
            std::cerr << "no engine file exists, now compiling ..." << std::endl;

            auto onnx_file = engine_file;
            onnx_file.replace_extension(".onnx");
            build_engine(onnx_file, engine_file);
            load_file(engine_file, &fsteam);
        }


        mEngine = std::unique_ptr<ICudaEngine>(mRuntime->deserializeCudaEngine(fsteam.data(), fsteam.size()));
        assert(mEngine);

        summary_engine(mEngine);
        mBuffers = std::make_unique<BufferManager>(mEngine, nullptr, 1);

        mContext = std::unique_ptr<nvinfer1::IExecutionContext>(mEngine->createExecutionContext());
        assert(mContext);

        cudaStreamCreate(&stream);
        mContext->setOptimizationProfileAsync(0, stream);
        mContext->setBindingDimensions(0, Dims3(1, 16384, 4));
    }

 private:
    class Logger : public nvinfer1::ILogger {
     public:
        void log(Severity severity, const char *msg) noexcept override {
            if (severity <= Severity::kWARNING)
                std::cout << msg << std::endl;
        }
    } logger;

    cudaStream_t stream;
    std::unique_ptr<BufferManager> mBuffers{nullptr};
    std::unique_ptr<nvinfer1::IRuntime> mRuntime{nullptr};
    std::unique_ptr<nvinfer1::ICudaEngine> mEngine{nullptr};
    std::unique_ptr<nvinfer1::IExecutionContext> mContext{nullptr};
};

std::vector<float> load_data(const boost::filesystem::path &path, size_t size = 16384 * 4) {
    std::vector<float> data(size, 0);
    std::ifstream file(path.string(), std::ios::in | std::ios::binary);
    if (file) {
        unsigned int len = 0;
        file.seekg(0, std::ifstream::end);
        len = file.tellg();
        file.seekg(0, std::ifstream::beg);
        data.resize(len / sizeof(float), 0);
        file.read(static_cast<char *>((void *) data.data()), len);
    }
    if (data.size() > size)
        data.resize(size);
#ifdef DEBUG
    int cnt = 0;
    for (auto &&x: data) {
        cnt++;
        std::cout << x << " ";
        if ((cnt % 4) == 0) {
            std::cout << std::endl;
        }
    }
#endif
    return data;
}

}  // namespace PointDetection

#endif //point_detection_TRT_MODELS_HPP
