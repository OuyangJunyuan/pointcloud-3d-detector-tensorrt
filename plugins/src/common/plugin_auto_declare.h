//
// Created by nrsl on 23-4-15.
//

#ifndef POINT_DETECTION_PLUGIN_AUTO_DECLARE_H
#define POINT_DETECTION_PLUGIN_AUTO_DECLARE_H

#include <cstring>
#include <cassert>

#include <sstream>
#include <vector>
#include <iostream>

#include <NvInferRuntime.h>
#include <boost/preprocessor.hpp>

namespace {
#ifdef TENSORRT_PLUGIN_DEBUG
#define TrTPrintf(...) fprintf(stderr,"[%s]: ",__FUNCTION__); fprintf(stderr,__VA_ARGS__)
#elif
#define TrTPrintf(...)
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

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
namespace nvinfer1::plugin {
struct Dummy {
};

template<typename T, typename = void>
struct as_base_if_it_is_complete_type : Dummy {
};

template<typename T>
struct as_base_if_it_is_complete_type<T, decltype(void(sizeof(T)))> : T {
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
        TrTPrintf("%s\n", ss.str().c_str());
        #endif
    }

    static void ReadBuffer(T &val, const void *const data) {
        val = static_cast<T const *>(data)[0];
        #ifdef TENSORRT_PLUGIN_DEBUG
        std::stringstream ss;
        ss << val << " = " << static_cast<T const *>(data)[0];
        TrTPrintf("%s\n", ss.str().c_str());
        #endif
    }

    template<typename buffer_type>
    static void WriteBuffer(const T &val, buffer_type *&data) {
        #ifdef TENSORRT_PLUGIN_DEBUG
        std::stringstream ss;
        ss << reinterpret_cast<T *>(data)[0] << " = " << val;
        TrTPrintf("%s\n", ss.str().c_str());
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

inline size_t SizeAlign256(size_t size) {
    constexpr size_t align = 256;
    return size + (size % align ? align - (size % align) : 0);
}

template<typename T>
inline T *GetOneWorkspace(void *const workspace, size_t size, size_t &offset) {
    auto buffer = reinterpret_cast<size_t>( workspace) + offset;
    offset += SizeAlign256(size);
    return reinterpret_cast<T *>( buffer);
}

}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#define TRT_PLUGIN_PLUGIN_NAME BOOST_PP_CAT(TRT_PLUGIN_NAME, Plugin)
#define TRT_PLUGIN_PLUGIN_CREATOR_NAME BOOST_PP_CAT(TRT_PLUGIN_NAME,PluginCreator)
#define TRT_PLUGIN_PLUGIN_USER BOOST_PP_CAT(TRT_PLUGIN_NAME, User)
struct TRT_PLUGIN_PLUGIN_USER;
namespace nvinfer1::plugin {
class TRT_PLUGIN_PLUGIN_NAME
        : public nvinfer1::IPluginV2DynamicExt,
          public as_base_if_it_is_complete_type<TRT_PLUGIN_PLUGIN_USER> {

    #define TRT_PLUGIN_ATTRIBUTE_DEFINE_TYPE(I, TYPE, NAME, ...)                                                        \
    using NAME##_t = TYPE;
    TRT_ENUM(ATTRIBUTE, TRT_PLUGIN_ATTRIBUTE_DEFINE_TYPE)

    template<typename T, int N>
    struct BufferManager {
        using Type = T;
        static constexpr int index = N;
        static constexpr int dsize = sizeof(T);
        T *ptr{nullptr};
        int32_t bytes{0};
        int32_t elems{0};
    };
 private:

/*    auto GetWorkSpaceElems(PluginTensorDesc const *inputs, PluginTensorDesc const *outputs, size_t *elems) const {
        #define Input(nin, ndim, ...)  inputs[nin].dims.d[ndim]
        #define Output(nin, ndim, ...)  outputs[nin].dims.d[ndim]
        #define Attr(name, ...)  attr.name
        #define Dimension(...) * __VA_ARGS__
        #define Define(name, ...)                                                                                       \
        const auto name=__VA_ARGS__;

        TENSORRT_PLUGIN_SETTING_DEFINE

        #define Workspace(t, n, ...)                                                                                    \
        sizes.n.nbytes = sizeof(t) TRT_CAT(TENSORRT_PLUGIN_,__VA_ARGS__);

                TENSORRT_PLUGIN_SETTING_WORKSPACE
        #undef Input
        #undef Output
        #undef Attr
        #undef Dimension
        #undef Define
        #undef Workspace

        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        #ifdef TENSORRT_PLUGIN_DEBUG
        #define Dimension(...) <<__VA_ARGS__<<" x "
        #define Workspace(t, n, ...)                                                                                    \
        {                                                                                                               \
            std::stringstream ss;                                                                                       \
            ss TRT_CAT(TENSORRT_PLUGIN_,__VA_ARGS__) << sizeof(t);                                                      \
            TrTPrintf("%s = %s\n", #n, ss.str().c_str());                                                \
        }
        TENSORRT_PLUGIN_SETTING_WORKSPACE
        #undef Dimension
        #undef Workspace
        #endif
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////

        return sizes;
    }

    inline void SetIOPointer(void const *const *inputs, void *const *outputs) {
        int ni = 0;
        #define Input(t, n, ...)                                                                                        \
        input.n = static_cast<const t* >(inputs[ni++]);                                                                 \
        TrTPrintf("%p(%s)\n", input.n,#n);
        TENSORRT_PLUGIN_SETTING_INPUT
        #undef Input

        int no = 0;
        #define Output(t, n, ...)                                                                                       \
        output.n = static_cast<t* >(outputs[no++]);                                                                     \
        TrTPrintf("%p(%s)\n", output.n,#n);
        TENSORRT_PLUGIN_SETTING_OUTPUT
        #undef Output
    }

    inline void SetWorkspacePtr(void *workspace) {
        size_t offset = 0;
        #define Workspace(type, name, ...)                                                                              \
        ws.name = GetOneWorkspace<type>(workspace, ws.size.name, offset);                                               \
        TrTPrintf("%p(%s)\n", ws.name, #name);
        TENSORRT_PLUGIN_SETTING_WORKSPACE
        #undef Workspace
    }

    inline void SetDefine(PluginTensorDesc const *inputs, PluginTensorDesc const *outputs) {
        #define Attr(name, ...)  attr.name
        #define Output(nin, ndim, ...)  int(0)
        #define Input(nin, ndim, ...)  inputs[nin].dims.d[ndim]
        #define Define(name, ...)                                                                                       \
        auto name = def.name =__VA_ARGS__;
        TENSORRT_PLUGIN_SETTING_DEFINE
        #undef Output
        #undef Input
        #undef Define
    }

    inline void SetWorkSpaceSize(nvinfer1::PluginTensorDesc const *inputs, nvinfer1::PluginTensorDesc const *outputs) {
        ws.size = GetWorkSpaceSizes(inputs, outputs);
        #define Workspace(t, n, ...)                                                                                    \
        TrTPrintf("%s = %lu\n", #n, ws.size.n );
        TENSORRT_PLUGIN_SETTING_WORKSPACE
        #undef Workspace
    }

    void InitAllShapeAndPointer(PluginTensorDesc const *inputDesc,
                                PluginTensorDesc const *outputDesc,
                                void const *const *inputs,
                                void *const *outputs,
                                void *workspace) {
        SetDefine(inputDesc, outputDesc);
        SetWorkSpaceSize(inputDesc, outputDesc);
        SetIOPointer(inputs, outputs);
        SetWorkspacePtr(workspace);
    }*/

 public:

    void serialize(void *buffer) const noexcept override {
        assert(buffer != nullptr);
        auto *ptr = reinterpret_cast<uint8_t *>(buffer);
        auto *begin = ptr;

        #define TRT_PLUGIN_ATTRIBUTE_WRITE_BUFFER(I, TYPE, NAME, ...)                                                   \
        TypeInfo<TYPE>::WriteBuffer(attr.NAME, ptr);
        TRT_ENUM(ATTRIBUTE, TRT_PLUGIN_ATTRIBUTE_WRITE_BUFFER)

        TrTPrintf("%lu = %lu + %lu ?\n", (size_t) ptr, (size_t) begin, getSerializationSize());
        assert(ptr == begin + getSerializationSize());
    }

    void setPluginNamespace(char const *pluginNamespace) noexcept override {
        assert(pluginNamespace != nullptr);
        namespace_ = pluginNamespace;
        TrTPrintf("%s\n", namespace_.c_str());
    }

    [[nodiscard]] size_t getSerializationSize() const noexcept override {
        #define TRT_PLUGIN_ATTRIBUTE_SIZE_SUM(I, TYPE, NAME, ...) + sizeof(attr.NAME)
        constexpr size_t serialization_size{0 TRT_ENUM(ATTRIBUTE, TRT_PLUGIN_ATTRIBUTE_SIZE_SUM)};
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        #define TRT_PLUGIN_ATTRIBUTE_SIZE_PRINT_STR(I, TYPE, NAME, ...) " + %lu"  BOOST_PP_STRINGIZE((NAME))
        TrTPrintf("%zu =" TRT_ENUM(ATTRIBUTE, TRT_PLUGIN_ATTRIBUTE_SIZE_PRINT_STR),
                  #define TRT_PLUGIN_ATTRIBUTE_SIZE_PRINT_SIZE(I, TYPE, NAME, ...) , sizeof(attr.NAME)
                          serialization_size TRT_ENUM(ATTRIBUTE, TRT_PLUGIN_ATTRIBUTE_SIZE_PRINT_SIZE));
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        return serialization_size;
    }

    static int32_t getNbInputs() noexcept {
        constexpr int32_t n = IN::N;
        TrTPrintf("%d\n", n);
        return n;
    }

    [[nodiscard]] int32_t getNbOutputs() const noexcept override {
        constexpr int32_t n = OUT::N;
        TrTPrintf("%d\n", n);
        return n;
    }

    [[nodiscard]] char const *getPluginNamespace() const noexcept override {
        TrTPrintf("%s\n", namespace_.c_str());
        return namespace_.c_str();
    }

    [[nodiscard]] char const *getPluginVersion() const noexcept override {
        TrTPrintf("%s\n", TRT_PLUGIN_VERSION);
        return TRT_PLUGIN_VERSION;
    }

    [[nodiscard]] char const *getPluginType() const noexcept override {
        TrTPrintf("%s\n", BOOST_PP_STRINGIZE(TRT_PLUGIN_PLUGIN_NAME));
        return BOOST_PP_STRINGIZE(TRT_PLUGIN_PLUGIN_NAME);
    }

    void destroy() noexcept override {
        delete this;
    }

    void terminate() noexcept override;

    int32_t initialize() noexcept override;

 public:

    DimsExprs getOutputDimensions(int32_t outputIndex,
                                  DimsExprs const *inputs,
                                  int32_t nbInputs,
                                  IExprBuilder &exprBuilder) noexcept override {
        assert(0 <= outputIndex && outputIndex < this->getNbOutputs());

        #define Output(nin, ndim, ...)  int(0)
        #define Input(nin, ndim, ...)  inputs[nin].d[ndim]->getConstantValue()
        #define Attr(name, ...)  attr.name
        #define TRT_PLUGIN_DEFINE_OUTPUT_DIMENSION(I, NAME, ...)                                                       \
        const auto NAME = __VA_ARGS__;
        TRT_ENUM(DEFINE, TRT_PLUGIN_DEFINE_OUTPUT_DIMENSION)
        #undef Output
        #undef Input
        #undef Define

        nvinfer1::DimsExprs dim{};
        #define dim(...) (__VA_ARGS__)
        #define TRT_PLUGIN_OUTPUT_SET_ONE_DIMENSION_PRINT_FORMAT(Z, N, X)  X
        #define TRT_PLUGIN_OUTPUT_SET_ONE_DIMENSION_PRINT_DATA(Z, N, X)  ,BOOST_PP_TUPLE_ELEM(N,X)
        #define TRT_PLUGIN_OUTPUT_SET_ONE_DIMENSION(Z, N, X)  dim.d[N] = TryExpr(BOOST_PP_TUPLE_ELEM(N,X),exprBuilder);
        #define TRT_PLUGIN_OUTPUT_SET_DIMENSION(I, TYPE, NAME, ...)                                                     \
        if ( out.NAME.index == outputIndex) {                                                                           \
            BOOST_PP_REPEAT(BOOST_PP_TUPLE_SIZE(__VA_ARGS__),TRT_PLUGIN_OUTPUT_SET_ONE_DIMENSION,__VA_ARGS__)           \
            dim.nbDims = BOOST_PP_TUPLE_SIZE(__VA_ARGS__);                                                              \
            TrTPrintf(#NAME " " #TYPE BOOST_PP_STRINGIZE(                                                               \
            (BOOST_PP_ENUM(BOOST_PP_TUPLE_SIZE(__VA_ARGS__),TRT_PLUGIN_OUTPUT_SET_ONE_DIMENSION_PRINT_FORMAT,%d))       \
            ) BOOST_PP_REPEAT(BOOST_PP_TUPLE_SIZE(__VA_ARGS__),                                                         \
            TRT_PLUGIN_OUTPUT_SET_ONE_DIMENSION_PRINT_DATA,__VA_ARGS__) );                                              \
        } else
        TRT_ENUM(OUTPUT, TRT_PLUGIN_OUTPUT_SET_DIMENSION) {
            assert(false);
        }
        #undef dim
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        return dim;
    }

    nvinfer1::DataType getOutputDataType(int32_t outputIndex,
                                         DataType const *inputTypes,
                                         int32_t nbInputs) const noexcept override {
        assert(inputTypes != nullptr);

        #define TRT_PLUGIN_OUTPUT_DATA_TYPE(I, TYPE, NAME, ...)                                                         \
        if (out.NAME.index == outputIndex) {                                                                            \
            TrTPrintf("%s %s\n", #NAME, data2str(TypeInfo<TYPE>::data_type));                                           \
            return TypeInfo<TYPE>::data_type;                                                                           \
        } else
        TRT_ENUM(OUTPUT, TRT_PLUGIN_OUTPUT_DATA_TYPE) {
            assert(false);
            return nvinfer1::DataType{};
        }
    }

    bool supportsFormatCombination(int32_t index,
                                   PluginTensorDesc const *inOut,
                                   int32_t nbInputs,
                                   int32_t nbOutputs) noexcept override {
        assert(inOut != nullptr);
        assert(nbInputs == getNbInputs());
        assert(nbOutputs == getNbOutputs());

        PluginTensorDesc const &io = inOut[index];

        #define TRT_PLUGIN_INPUT_SUPPORTS_FORMAT(I, TYPE, NAME, ...)                                                    \
        if (index == in.NAME.index) {                                                                                   \
            TrTPrintf(#NAME ", %s\n", data2str(TypeInfo<TYPE>::data_type));                                             \
            return (TypeInfo<TYPE>::data_type == io.type) && (io.format == TensorFormat::kLINEAR);}                     \
        else
        #define TRT_PLUGIN_OUTPUT_SUPPORTS_FORMAT(I, TYPE, NAME, ...)                                                   \
        if (index == in.N + out.NAME.index) {                                                                           \
            TrTPrintf(#NAME ", %s\n", data2str(TypeInfo<TYPE>::data_type));                                             \
            return (TypeInfo<TYPE>::data_type == io.type) && (io.format == TensorFormat::kLINEAR);}                     \
        else
        TRT_ENUM(INPUT, TRT_PLUGIN_INPUT_SUPPORTS_FORMAT) TRT_ENUM(OUTPUT, TRT_PLUGIN_OUTPUT_SUPPORTS_FORMAT) {
            assert(false);
            return false;
        }

    }

    void configurePlugin(nvinfer1::DynamicPluginTensorDesc const *ins, int32_t nbInputs,
                         nvinfer1::DynamicPluginTensorDesc const *outs, int32_t nbOutputs) noexcept override {
        assert(ins != nullptr);
        assert(outs != nullptr);
        assert(nbInputs == getNbInputs());
        assert(nbOutputs == getNbOutputs());
    }

    size_t getWorkspaceSize(nvinfer1::PluginTensorDesc const *inputs, int32_t nbInputs,
                            nvinfer1::PluginTensorDesc const *outputs, int32_t nbOutputs) const noexcept override {
        auto sizes = GetWorkSpaceSizes(inputs, outputs);
        size_t total = 0;
#define Workspace(data_type, data_name, ...)                                                                    \
        total += SizeAlign256(sizes.data_name);                                                                       \
        TrTPrintf("%zu(total)\n", total);
        TENSORRT_PLUGIN_SETTING_WORKSPACE
#undef Workspace
        return
                total;
    };

    int32_t enqueue(PluginTensorDesc const *inputDesc,
                    PluginTensorDesc const *outputDesc,
                    void const *const *inputs,
                    void *const *outputs,
                    void *workspace,
                    cudaStream_t stream) noexcept override {
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        #ifdef TENSORRT_PLUGIN_DEBUG
        TrTPrintf("enqueue\n");
        int n_in = 0;
        #define PRINT(desc, t, n, ...)                                                                                  \
          {                                                                                                               \
              std::stringstream ss;                                                                                       \
              for (int i = 0; i < desc[n_in].dims.nbDims; i++) {                                                          \
                  ss<<desc[n_in].dims.d[i]<<(i == desc[n_in].dims.nbDims - 1 ? "" : ",");                                 \
              }                                                                                                           \
              TrTPrintf("%s %s[%s] format: %d\n",                                                                         \
                  #n, data2str(desc[n_in].type),ss.str().c_str(),(int)desc[n_in].format);                                 \
          }
        #define Input(t, n, ...) PRINT(inputDesc,t,n)
        #define Output(t, n, ...) PRINT(outputDesc,t,n)
        TENSORRT_PLUGIN_SETTING_INPUT
                TENSORRT_PLUGIN_SETTING_OUTPUT
        #undef Input
        #undef Output
        #undef PRINT
        #endif
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        InitAllShapeAndPointer(inputDesc, outputDesc, inputs, outputs, workspace);
        return enqueue(stream);
    }

    int32_t enqueue(cudaStream_t stream) noexcept;

 public:

    TRT_PLUGIN_PLUGIN_NAME(void const *data, size_t length) {
        assert(data != nullptr);
        auto const *d = reinterpret_cast<uint8_t const *>(data);
        auto const *a = d;

        #define TRT_PLUGIN_ATTRIBUTE_CONSTRUCTION(I, TYPE, NAME, ...) \
        TypeInfo<TYPE>::ReadBuffer(attr.NAME, d);                     \
        d+=sizeof(attr.NAME);
        TRT_ENUM(ATTRIBUTE, TRT_PLUGIN_ATTRIBUTE_CONSTRUCTION)
        TrTPrintf("deserialization\n");
        assert(d == a + length);
    }

    #define TRT_PLUGIN_ATTRIBUTE_CONSTRUCTION_ARGS(I, TYPE, NAME, ...) , const NAME##_t &_##NAME

    TRT_PLUGIN_PLUGIN_NAME(void *TRT_ENUM(ATTRIBUTE, TRT_PLUGIN_ATTRIBUTE_CONSTRUCTION_ARGS)) {

        #define TRT_PLUGIN_ATTRIBUTE_CONSTRUCTION_ARGS_2(I, TYPE, NAME, ...) \
        TypeInfo<TYPE>::DeepCopy(attr.NAME,_##NAME);
        TRT_ENUM(ATTRIBUTE, TRT_PLUGIN_ATTRIBUTE_CONSTRUCTION_ARGS_2)
    }

    [[nodiscard]] nvinfer1::IPluginV2DynamicExt *clone() const noexcept override {
        #define TRT_PLUGIN_ATTRIBUTE_CONSTRUCTION_ARGS3(I, TYPE, NAME, ...) , attr.NAME
        auto *plugin = new TRT_PLUGIN_PLUGIN_NAME(nullptr TRT_ENUM(ATTRIBUTE, TRT_PLUGIN_ATTRIBUTE_CONSTRUCTION_ARGS3));
        plugin->setPluginNamespace(namespace_.c_str());
        return plugin;
    }

    TRT_PLUGIN_PLUGIN_NAME() = delete;

 private:
    struct DEF {
        #define TRT_PLUGIN_DEFINE_DEFINE(I, NAME, ...)                                                                  \
        int NAME;
        TRT_ENUM(DEFINE, TRT_PLUGIN_DEFINE_DEFINE)
        static constexpr int N{TRT_SIZE(DEFINE)};
    } def;
    struct ATTR {
        #define TRT_PLUGIN_ATTRIBUTE_DEFINE(I, TYPE, NAME, ...)                                                         \
        NAME##_t NAME=__VA_ARGS__;
        TRT_ENUM(ATTRIBUTE, TRT_PLUGIN_ATTRIBUTE_DEFINE)
        static constexpr int N{TRT_SIZE(ATTRIBUTE)};
    } attr;
    struct IN {
        #define TRT_PLUGIN_INPUT_DEFINE(I, TYPE, NAME, ...)                                                             \
        BufferManager<TYPE,I> NAME;
        TRT_ENUM(INPUT, TRT_PLUGIN_INPUT_DEFINE)
        static constexpr int N{TRT_SIZE(INPUT)};
    } in;
    struct OUT {
        #define TRT_PLUGIN_OUTPUT_DEFINE(I, TYPE, NAME, ...)                                                            \
        BufferManager<TYPE,I> NAME;
        TRT_ENUM(OUTPUT, TRT_PLUGIN_OUTPUT_DEFINE)
        static constexpr int N{TRT_SIZE(OUTPUT)};
    } out;
    struct WS {
        #define TRT_PLUGIN_WORKSPACE_DEFINE(I, TYPE, NAME, ...)                                                         \
        BufferManager<TYPE, I> NAME;
        TRT_ENUM(WORKSPACE, TRT_PLUGIN_WORKSPACE_DEFINE)
        static constexpr int N{TRT_SIZE(WORKSPACE)};
    } ws;
    std::string namespace_;

    friend class TENSORRT_PLUGIN_CREATOR_NAME;
};
/*
class TENSORRT_PLUGIN_CREATOR_NAME : public nvinfer1::IPluginCreator {
    decltype(TENSORRT_PLUGIN_NAME::attr
    )
            attr;

 public:
    TENSORRT_PLUGIN_CREATOR_NAME() {
        plugin_attributes_.

                clear();

#define Attribute(type, name, ...)                                                                              \
        plugin_attributes_.emplace_back(                                                                                \
            PluginField(TRT_STR(name), nullptr,                                                                         \
            TypeInfo<decltype(attr.name)>::field_type,                                                                  \
            TypeInfo<decltype(attr.name)>::len)                                                                         \
        );
        TENSORRT_PLUGIN_SETTING_ATTR
#undef Attribute
        field_collection_
                .
                        nbFields = plugin_attributes_.size();
        field_collection_.
                fields = plugin_attributes_.data();
    }

    nvinfer1::IPluginV2 *createPlugin(char const *name, nvinfer1::PluginFieldCollection const *fc) noexcept

    override {
        assert(fc != nullptr);

#define Attribute(type, data_name, ...)                                                                         \
        decltype(attr.data_name) data_name;
        TENSORRT_PLUGIN_SETTING_ATTR
#undef Attribute

                int32_t
        num_fields = fc->nbFields;
        for (
                auto i = 0;
                i < num_fields;
                ++i) {
            auto &field = fc->fields[i];
#define Attribute(type, data_name, ...)                                                                     \
            if(!strcmp(field.name, #data_name)){                                                                        \
                TypeInfo<decltype(data_name)>::ReadBuffer(data_name, field.data);                                       \
            }else

            TENSORRT_PLUGIN_SETTING_ATTR{
                    assert(0);
            }

#undef Attribute
        }
#define Attribute(type, name, ...)  , name
        IPluginV2 * plugin = new TENSORRT_PLUGIN_NAME
                (nullptr
        TENSORRT_PLUGIN_SETTING_ATTR
        );
#undef Attribute
        return
                plugin;
    }

    [[nodiscard]] char const *getPluginName() const noexcept

    override {
        return
                TRT_STR(TENSORRT_PLUGIN_SETTING_NAME);
    }

    [[nodiscard]] char const *getPluginVersion() const noexcept

    override {
        return
                TENSORRT_PLUGIN_SETTING_VERSION;
    }

    nvinfer1::PluginFieldCollection const *getFieldNames() noexcept

    override {
        return &
                field_collection_;
    }

    nvinfer1::IPluginV2 *deserializePlugin(char const *name, void const *data, size_t length) noexcept

    override {
        return new
                TENSORRT_PLUGIN_NAME(data, length
        );
    }

    void setPluginNamespace(char const *pluginNamespace) noexcept

    override {
        assert(pluginNamespace != nullptr);
        namespace_ = pluginNamespace;
    }

    [[nodiscard]] char const *getPluginNamespace() const noexcept

    override {
        return namespace_.

                c_str();

    }

 private:
    static nvinfer1::PluginFieldCollection field_collection_;
    static std::vector<nvinfer1::PluginField> plugin_attributes_;
    std::string namespace_;
};

PluginFieldCollection TENSORRT_PLUGIN_CREATOR_NAME
        ::field_collection_{
        };
std::vector<PluginField> TENSORRT_PLUGIN_CREATOR_NAME
        ::plugin_attributes_;
REGISTER_TENSORRT_PLUGIN(TENSORRT_PLUGIN_CREATOR_NAME);*/
}

#endif //POINT_DETECTION_PLUGIN_AUTO_DECLARE_H
