//
// Created by nrsl on 23-4-7.
//

#ifndef POINT_DETECTION_UTILS_H
#define POINT_DETECTION_UTILS_H

#include <vector>
#include <chrono>

#include <boost/filesystem.hpp>

inline auto canonical(const boost::filesystem::path &path) {
    return path.is_absolute() ? path : boost::filesystem::weakly_canonical(PROJECT_ROOT / path);
}

inline auto time_to(const std::chrono::steady_clock::time_point &t1) {
    return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t1).count();
}

auto LoadBinData(const boost::filesystem::path &path) {
    std::vector<unsigned char> data(0);
    std::ifstream file(path.string(), std::ios::in | std::ios::binary);
    if (file) {
        unsigned int len = 0;
        file.seekg(0, std::ifstream::end);
        len = file.tellg();
        file.seekg(0, std::ifstream::beg);
        data.resize(len, 0);
        file.read(static_cast<char *>((void *) data.data()), len);
    }
    return data;
}

inline void ReadAndPreprocess(const unsigned char *const src_ptr,
                              const size_t src_bytes,
                              const size_t src_point_bytes,
                              std::vector<float> *dest) {
    const static float x_mim = 0, x_max = 70, y_min = -45, y_max = 45, r2_min = 16;
    const static auto dest_point_feats = 4;

    const auto src_point_num = src_bytes / src_point_bytes;
    dest->reserve(src_point_num * dest_point_feats);

    auto *const src_end = src_ptr + src_bytes;
    auto src_cur = src_ptr;
    auto dest_cur = dest->data();
    for (; src_cur <= src_end; src_cur += src_point_bytes) {
        auto *p = reinterpret_cast<const float *>(src_cur);
        auto x = p[0], y = p[1], z = p[2], intensity = p[3];
        auto r2 = x * x + y * y + z * z;
        if (/*(x_mim < x and x < x_max) and (y_min < y and y < y_max) and*/ r2_min < r2) {
            dest_cur[0] = x;
            dest_cur[1] = y;
            dest_cur[2] = z;
            dest_cur[3] = intensity;
            dest_cur += dest_point_feats;
        }
    }
    std::random_shuffle(reinterpret_cast<typeof(float[dest_point_feats]) *>((void *) dest->data()),
                        reinterpret_cast<typeof(float[dest_point_feats]) *>(dest_cur));
}

#endif //POINT_DETECTION_UTILS_H
