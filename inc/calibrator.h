//
// Created by nrsl on 23-4-7.
//

#ifndef POINT_DETECTION_CALIBRATOR_H
#define POINT_DETECTION_CALIBRATOR_H

#include <NvInfer.h>

#include "inc/helper.h"

namespace PointDetection {
class Int8EntropyCalibrator2 : public nvinfer1::IInt8EntropyCalibrator2 {
 public:
    Int8EntropyCalibrator2(const YAML::Node &cfgs,
                           const boost::filesystem::path name,
                           const nvinfer1::Dims &dims,
                           int batch_size) :
            batch_size(batch_size),
            use_cache(cfgs["cache"].as<bool>()),
            data_dir(canonical(cfgs["data"].as<std::string>()).string()),
            cache_file(canonical(change_extension(name, ".cache").string()).string()) {

        for (auto &&file: boost::filesystem::recursive_directory_iterator(data_dir)) {
            data_files.emplace_back(file);
        }

        nbytes_per_sample_ = getBytesPerSample(dims);
        cudaMalloc(&mDeviceInput, batch_size * nbytes_per_sample_);

        last_iter = cfgs["max_iters"].as<int>();

        step_iter = data_files.size() / (last_iter * batch_size);
        step_iter = step_iter < 1 ? 1 : step_iter;
    }

    int getBytesPerSample(const nvinfer1::Dims &dims) {
        int bytes = 1;
        for (int i = 1; i < dims.nbDims; ++i) {
            bytes *= dims.d[i];
        }
        bytes *= sizeof(float);
        return bytes;
    }

    int getBatchSize() const noexcept override {
        return batch_size;
    }

    bool getBatch(void *bindings[], const char *names[], int nbBindings) noexcept override {
        if (cur_iter >= last_iter) {
            return false;
        }
        for (int i = 0; i < batch_size; ++i) {
            auto points = LoadBinData(data_files[batch_size * cur_iter * step_iter + i]);
            std::vector<float> host_data;
            ReadAndPreprocess(points.data(), points.size(), 4 * sizeof(float), &host_data);
            cudaMemcpy((float *) mDeviceInput + i * nbytes_per_sample_, host_data.data(),
                       nbytes_per_sample_, cudaMemcpyHostToDevice);
        }
        cur_iter += 1;
        bindings[0] = mDeviceInput;
        fprintf(stderr, "batch(%d/%d)\n", cur_iter, last_iter);
        return true;
    }

    const void *readCalibrationCache(size_t &length) noexcept override {
        cache_data.clear();
        std::ifstream input(cache_file, std::ios::binary);
        input >> std::noskipws;
        if (use_cache && input.good()) {
            std::copy(std::istream_iterator<char>(input), std::istream_iterator<char>(),
                      std::back_inserter(cache_data));
        }
        length = cache_data.size();
        return length ? cache_data.data() : nullptr;
    }

    void writeCalibrationCache(const void *cache, size_t length) noexcept override {
        std::ofstream output(cache_file, std::ios::binary);
        output.write(reinterpret_cast<const char *>(cache), length);
    }

 private:
    const std::string data_dir;
    std::vector<boost::filesystem::path> data_files;
    bool use_cache;
    const std::string cache_file;
    std::vector<char> cache_data;

    int batch_size{1};
    int nbytes_per_sample_{1};
    void *mDeviceInput{nullptr};

    int cur_iter{0};
    int last_iter{1};
    int step_iter{0};
};

}

#endif //POINT_DETECTION_CALIBRATOR_H
