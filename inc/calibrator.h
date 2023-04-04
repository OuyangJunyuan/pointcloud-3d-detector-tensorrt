//
// Created by nrsl on 23-4-3.
//

#ifndef APP_CALIBRATOR_H
#define APP_CALIBRATOR_H

#include <NvInfer.h>
#include <boost/filesystem.hpp>

#include "helper.h"

using Path = boost::filesystem::path;

class Int8EntropyCalibrator2 : public nvinfer1::IInt8EntropyCalibrator2 {
 public:
    Int8EntropyCalibrator2(const Path &data_dir,
                           const Path &cache_file,
                           const nvinfer1::Dims &dims,
                           int batch_size,
                           bool use_cache = true) :
            data_dir(data_dir),
            cache_file(cache_file),
            batch_size(batch_size),
            use_cache(use_cache) {

        for (auto &&file: boost::filesystem::recursive_directory_iterator(data_dir)) {
            data_files.emplace_back(file);
        }
        for (int i = 1; i < dims.nbDims; ++i) {
            nbytes_per_sample *= dims.d[i];
        }
        nbytes_per_sample *= sizeof(float);
        last_iter = data_files.size() / batch_size;
        cudaMalloc(&mDeviceInput, batch_size * nbytes_per_sample);
    }

    int getBatchSize() const noexcept override {
        return batch_size;
    }

    bool getBatch(void *bindings[], const char *names[], int nbBindings) noexcept override {
        if (cur_iter >= last_iter) {
            return false;
        }
//        for (int i = 0; i < batch_size; ++i) {
//            auto host_data = load_data(data_files[batch_size * cur_iter + i]);
//            cudaMemcpy((float *) mDeviceInput + i * nbytes_per_sample, host_data.data(),
//                       nbytes_per_sample, cudaMemcpyHostToDevice);
//        }
        cur_iter += 1;
        bindings[0] = mDeviceInput;
        return true;
    }

    const void *readCalibrationCache(size_t &length) noexcept override {
        data_cache.clear();
        std::ifstream input(cache_file.string(), std::ios::binary);
        input >> std::noskipws;
        if (use_cache && input.good()) {
            std::copy(std::istream_iterator<char>(input), std::istream_iterator<char>(),
                      std::back_inserter(data_cache));
        }
        length = data_cache.size();
        return length ? data_cache.data() : nullptr;
    }

    void writeCalibrationCache(const void *cache, size_t length) noexcept override {
        std::ofstream output(cache_file.string(), std::ios::binary);
        output.write(reinterpret_cast<const char *>(cache), length);
    }

 private:
    const Path data_dir;
    const Path cache_file;
    bool use_cache;
    int batch_size{1};
    int cur_iter{0};
    int last_iter{1};
    int nbytes_per_sample{1};
    std::vector<Path> data_files;
    std::vector<char> data_cache;
    void *mDeviceInput{nullptr};
};

#endif //APP_CALIBRATOR_H
