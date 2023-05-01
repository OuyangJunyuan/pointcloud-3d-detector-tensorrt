//
// Created by nrsl on 23-4-30.
//

#ifndef POINT_DETECTION_PLUGIN_AUTO_DECLARE_HELPER_H
#define POINT_DETECTION_PLUGIN_AUTO_DECLARE_HELPER_H

#include <cassert>
#include <sstream>

#include <NvInferRuntime.h>
#include <type_traits>

namespace {
#ifdef TENSORRT_PLUGIN_DEBUG
#define dbg(...) fprintf(stderr,"[%s]: ",__FUNCTION__); fprintf(stderr,__VA_ARGS__)
#else
#define dbg(...)
#endif
#define BOOST_PP_FWD(...) __VA_ARGS__
#define TRT_PLUGIN_ITEM0(z, n, X) BOOST_PP_FWD(BOOST_PP_TUPLE_ELEM(0,X) BOOST_PP_TUPLE_PUSH_FRONT(BOOST_PP_TUPLE_ELEM(n, BOOST_PP_TUPLE_ELEM(1,X)),n))
#define TRT_ENUM(X, MACRO) BOOST_PP_REPEAT(BOOST_PP_TUPLE_SIZE((TRT_PLUGIN_##X)), TRT_PLUGIN_ITEM0,(MACRO, (TRT_PLUGIN_##X)) )
#define TRT_SIZE(X) BOOST_PP_TUPLE_SIZE((TRT_PLUGIN_##X))

#define TRT_PLUGIN_name(...) __VA_ARGS__
#define TRT_PLUGIN_version(...) __VA_ARGS__
#define TRT_PLUGIN_attribute(...) __VA_ARGS__
#define TRT_PLUGIN_define(...) __VA_ARGS__
#define TRT_PLUGIN_input(...) __VA_ARGS__
#define TRT_PLUGIN_output(...) __VA_ARGS__
#define TRT_PLUGIN_workspace(...) __VA_ARGS__
#define TRT_PLUGIN_dim(...) __VA_ARGS__
#define TRT_PLUGIN_NAME BOOST_PP_CAT(TRT_PLUGIN_,BOOST_PP_TUPLE_ELEM(0, TENSORRT_PLUGIN_SETTING))
#define TRT_PLUGIN_VERSION BOOST_PP_CAT(TRT_PLUGIN_,BOOST_PP_TUPLE_ELEM(1, TENSORRT_PLUGIN_SETTING))
#define TRT_PLUGIN_ATTRIBUTE BOOST_PP_CAT(TRT_PLUGIN_,BOOST_PP_TUPLE_ELEM(2, TENSORRT_PLUGIN_SETTING))
#define TRT_PLUGIN_DEFINE BOOST_PP_CAT(TRT_PLUGIN_,BOOST_PP_TUPLE_ELEM(3, TENSORRT_PLUGIN_SETTING))
#define TRT_PLUGIN_INPUT BOOST_PP_CAT(TRT_PLUGIN_,BOOST_PP_TUPLE_ELEM(4, TENSORRT_PLUGIN_SETTING))
#define TRT_PLUGIN_OUTPUT BOOST_PP_CAT(TRT_PLUGIN_,BOOST_PP_TUPLE_ELEM(5, TENSORRT_PLUGIN_SETTING))
#define TRT_PLUGIN_WORKSPACE BOOST_PP_CAT(TRT_PLUGIN_,BOOST_PP_TUPLE_ELEM(6, TENSORRT_PLUGIN_SETTING))
#define TRT_PLUGIN_DIM(...) BOOST_PP_CAT(TRT_PLUGIN_, __VA_ARGS__)
}

namespace nvinfer1::plugin {
struct FakeBase {
};

template<typename T, typename = void>
struct InheritIfComplete : FakeBase {
};

template<typename T>
struct InheritIfComplete<T, decltype(void(sizeof(T)))> : T {
};

inline auto TryExpr(int32_t t, nvinfer1::IExprBuilder &exprBuilder) {
    return exprBuilder.constant(t);
}

inline auto TryExpr(const IDimensionExpr *const &t, nvinfer1::IExprBuilder &exprBuilder) {
    return t;
}


template<typename C>
inline constexpr PluginFieldType type2field() {
    if constexpr(std::is_same_v<double, C>) {
        return PluginFieldType::kFLOAT64;
    } else if constexpr(std::is_same_v<float, C>) {
        return PluginFieldType::kFLOAT32;
    } else if constexpr(std::is_same_v<int, C> or std::is_same_v<unsigned int, C>) {
        return PluginFieldType::kINT32;
    } else if constexpr(std::is_same_v<unsigned char, C>) {
        return PluginFieldType::kINT8;
    } else if constexpr(std::is_same_v<char, C>) {
        return PluginFieldType::kCHAR;
    } else {
        assert(false);
        return PluginFieldType::kUNKNOWN;
    }
}

template<typename C>
inline constexpr DataType type2data() {
    if constexpr(std::is_same_v<float, C>) {
        return DataType::kFLOAT;
    } else if constexpr(std::is_same_v<int, C> or std::is_same_v<unsigned int, C>) {
        return DataType::kINT32;
    } else if constexpr(std::is_same_v<char, C> or std::is_same_v<unsigned char, C>) {
        return DataType::kINT8;
    } else if constexpr(std::is_same_v<bool, C>) {
        return DataType::kBOOL;
    } else {
        assert(false);
        return DataType::kFLOAT;
    }
}

inline constexpr const char *data2str(const DataType &data) {
    switch (data) {
        case DataType::kFLOAT:
            return "float";
        case DataType::kHALF:
            return "half";
        case DataType::kINT8:
            return "int8";
        case DataType::kINT32:
            return "int32";
        case DataType::kBOOL:
            return "bool";
        default:
            return "unknown";
    }
}

inline auto to_string(const DimsExprs &dim) {
    std::stringstream ss;
    for (int i = 0; i < dim.nbDims;) {
        ss << dim.d[i]->getConstantValue() << (++i < dim.nbDims ? ", " : "");
    }
    return ss.str();
}

inline auto to_string(const Dims &dim) {
    std::stringstream ss;
    for (int i = 0; i < dim.nbDims;) {
        ss << dim.d[i] << (++i < dim.nbDims ? ", " : "");
    }
    return ss.str();
}

template<typename T>
struct TypeInfo {
    using type = T;
    static constexpr int len = 1;
    static constexpr auto field_type = type2field<std::remove_cv_t<T>>();
    static constexpr auto data_type = type2data<std::remove_cv_t<T>>();

    static void DeepCopy(T &val, const T &data) {
        val = data;
        #ifdef TENSORRT_PLUGIN_DEBUG
        std::stringstream ss;
        ss << val << " = " << data;
        dbg("%s\n", ss.str().c_str());
        #endif
    }

    static void ReadBuffer(T &val, const void *const data) {
        val = static_cast<T const *>(data)[0];
        #ifdef TENSORRT_PLUGIN_DEBUG
        std::stringstream ss;
        ss << val << " = " << static_cast<T const *>(data)[0];
        dbg("%s\n", ss.str().c_str());
        #endif
    }

    template<typename buffer_type>
    static void WriteBuffer(const T &val, buffer_type *&data) {
        #ifdef TENSORRT_PLUGIN_DEBUG
        std::stringstream ss;
        ss << reinterpret_cast<T *>(data)[0] << " = " << val;
        dbg("%s\n", ss.str().c_str());
        #endif
        reinterpret_cast<T *>(data)[0] = val;
        data = reinterpret_cast<buffer_type *>(reinterpret_cast<T *>(data) + 1);
    }
};

template<typename T, int N>
struct TypeInfo<T[N]> {
    using type = T;
    using array_type = T[N];
    static constexpr int len = N;
    static constexpr auto field_type = type2field<std::remove_cv_t<T>>();
    static constexpr auto data_type = type2data<std::remove_cv_t<T>>();

    static void DeepCopy(array_type &val, const array_type &data) {
        for (int i = 0; i < N; ++i) {
            TypeInfo<T>::DeepCopy(val[i], data[i]);
        }
    }

    static void ReadBuffer(array_type &val, const void *const data) {
        auto *ptr = static_cast<T const *>(data);
        for (int i = 0; i < N; ++i) {
            TypeInfo<T>::ReadBuffer(val[i], &ptr[i]);
        }
    }

    template<typename buffer_type>
    static void WriteBuffer(const array_type &val, buffer_type *&data) {
        for (int i = 0; i < N; ++i) {
            TypeInfo<T>::WriteBuffer(val[i], data);
        }
    }
};
}
#endif //POINT_DETECTION_PLUGIN_AUTO_DECLARE_HELPER_H
