//
// Created by nrsl on 23-4-15.
//

#ifndef POINT_DETECTION_PLUGIN_AUTO_DECLARE_H
#define POINT_DETECTION_PLUGIN_AUTO_DECLARE_H

#include "plugin_auto_declare_helper.h"

#include <cstring>
#include <cassert>

#include <sstream>
#include <vector>
#include <iostream>

#include <NvInferPlugin.h>
#include <boost/preprocessor.hpp>

#include "common/common.h"


#define TrTPluginBase BOOST_PP_CAT(TRT_PLUGIN_NAME, Base)
#define TrTPluginUser BOOST_PP_CAT(TRT_PLUGIN_NAME, User)
#define TrTPluginImpl BOOST_PP_CAT(TRT_PLUGIN_NAME, Plugin)
#define TrTPluginCreator BOOST_PP_CAT(TRT_PLUGIN_NAME, PluginCreator)


namespace nvinfer1::plugin {
struct TrTPluginUser;


struct TrTPluginBase : public IPluginV2DynamicExt,
                       public InheritIfComplete<TrTPluginUser> {

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

    void setPluginNamespace() noexcept {
        setPluginNamespace(namespace_.c_str());
    }

    void setPluginNamespace(char const *pluginNamespace) noexcept override {
        assert(pluginNamespace != nullptr);
        namespace_ = pluginNamespace;
        dbg("%s\n", namespace_.c_str());
    }

    [[nodiscard]] char const *getPluginNamespace() const noexcept override {
        dbg("%s\n", namespace_.c_str());
        return namespace_.c_str();
    }

    [[nodiscard]] char const *getPluginVersion() const noexcept override {
        dbg("%s\n", TRT_PLUGIN_VERSION);
        return TRT_PLUGIN_VERSION;
    }

    [[nodiscard]] char const *getPluginType() const noexcept override {
        dbg("%s\n", BOOST_PP_STRINGIZE(TRT_PLUGIN_NAME));
        return BOOST_PP_STRINGIZE(TRT_PLUGIN_NAME);
    }

    static int32_t getNbInputs() noexcept {
        constexpr int32_t n = IN::N;
        dbg("%d\n", n);
        return n;
    }

    [[nodiscard]] int32_t getNbOutputs() const noexcept override {
        constexpr int32_t n = OUT::N;
        dbg("%d\n", n);
        return n;
    }

    void configurePlugin(DynamicPluginTensorDesc const *ins, int32_t nbInputs,
                         DynamicPluginTensorDesc const *outs, int32_t nbOutputs) noexcept override {
        assert(ins != nullptr);
        assert(outs != nullptr);
        assert(nbInputs == getNbInputs());
        assert(nbOutputs == getNbOutputs());
    }

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
};

class TrTPluginImpl : public TrTPluginBase {
    inline void GetWorkSpaceElems(PluginTensorDesc const *inputs,
                                  PluginTensorDesc const *outputs,
                                  size_t (&elems)[WS::N]) const {
        #define outputs(INDEX) outputs[INDEX] TRT_PLUGIN_DEFINE_GET_WS_SPACE_DIM_
        #define inputs(INDEX)  inputs[INDEX] TRT_PLUGIN_DEFINE_GET_WS_SPACE_DIM_
        #define TRT_PLUGIN_DEFINE_GET_WS_SPACE_DIM_(NDIM) .dims.d[NDIM]
        #define TRT_PLUGIN_DEFINE_GET_WS_SPACE_DIMENSION(I, NAME, ...)                                                  \
        const auto NAME = __VA_ARGS__;
        TRT_ENUM(DEFINE, TRT_PLUGIN_DEFINE_GET_WS_SPACE_DIMENSION)
        #undef outputs
        #undef inputs

        #define dim(...) (__VA_ARGS__)
        #define TRT_PLUGIN_DEFINE_GET_WS_SPACE_BYTES_(Z, N, X) *BOOST_PP_TUPLE_ELEM(N,X)
        #define TRT_PLUGIN_DEFINE_GET_WS_SPACE_BYTES(I, TYPE, NAME, ...)                                                \
        elems[I] = 1 BOOST_PP_REPEAT(BOOST_PP_TUPLE_SIZE(__VA_ARGS__),TRT_PLUGIN_DEFINE_GET_WS_SPACE_BYTES_,__VA_ARGS__);
        TRT_ENUM(WORKSPACE, TRT_PLUGIN_DEFINE_GET_WS_SPACE_BYTES)
        #undef dim
    }

    inline void SetWorkSpaceSize(PluginTensorDesc const *inputs,
                                 PluginTensorDesc const *outputs) {
        #define TRT_PLUGIN_SET_WORKSPACE_SIZE(I, TYPE, NAME, ...)                                                       \
        ws.NAME.elems = elems[I];                                                                                       \
        ws.NAME.bytes = ws.NAME.elems * ws.NAME.dsize;                                                                  \
        dbg("%s %d\n", #NAME, ws.NAME.bytes);
        size_t elems[WS::N];
        GetWorkSpaceElems(inputs, outputs, elems);
        TRT_ENUM(WORKSPACE, TRT_PLUGIN_SET_WORKSPACE_SIZE)
    }

    inline void SetIOPointer(void const *const *inputs,
                             void *const *outputs) {
        #define TRT_PLUGIN_DEFINE_SET_INPUT_POINTER(I, TYPE, NAME, ...)                                                 \
        in.NAME.ptr = static_cast<decltype(in.NAME)::Type* >((void *)inputs[in.NAME.index]);                            \
        dbg("%s.ptr %p\n",#NAME, in.NAME.ptr);
        TRT_ENUM(INPUT, TRT_PLUGIN_DEFINE_SET_INPUT_POINTER)

        #define TRT_PLUGIN_DEFINE_SET_OUTPUT_POINTER(I, TYPE, NAME, ...)                                                \
        out.NAME.ptr = static_cast<decltype(out.NAME)::Type* >((void *)outputs[out.NAME.index]);                        \
        dbg("%s.ptr %p\n",#NAME, out.NAME.ptr);
        TRT_ENUM(OUTPUT, TRT_PLUGIN_DEFINE_SET_OUTPUT_POINTER)
    }

    inline void SetWorkspacePtr(PluginTensorDesc const *inputs,
                                PluginTensorDesc const *outputs, void *workspace) {
        SetWorkSpaceSize(inputs, outputs);
        size_t offset = 0;
        #define TRT_PLUGIN_DEFINE_SET_WORKSPACE_POINTER(I, TYPE, NAME, ...)                                             \
        ws.NAME.ptr = GetOneWorkspace<TYPE>(workspace, ws.NAME.bytes, offset);                                          \
        dbg("%s.ptr %p\n",#NAME, ws.NAME.ptr);

        TRT_ENUM(WORKSPACE, TRT_PLUGIN_DEFINE_SET_WORKSPACE_POINTER)
    }

    inline void SetDefinition(PluginTensorDesc const *inputs, PluginTensorDesc const *outputs) {
        #define outputs(INDEX) outputs[INDEX] TRT_PLUGIN_DEFINE_GET_DIMENSION_
        #define inputs(INDEX)  inputs[INDEX] TRT_PLUGIN_DEFINE_GET_DIMENSION_
        #define TRT_PLUGIN_DEFINE_GET_DIMENSION_(NDIM) .dims.d[NDIM]
        #define TRT_PLUGIN_DEFINE_GET_DIMENSION(I, NAME, ...)                                                           \
        def.NAME = __VA_ARGS__;                                                                                         \
        dbg("%s %d\n",#NAME,def.NAME);
        TRT_ENUM(DEFINE, TRT_PLUGIN_DEFINE_GET_DIMENSION)
        #undef outputs
        #undef inputs
    }

    void InitAllShapeAndPointer(PluginTensorDesc const *inputDesc, PluginTensorDesc const *outputDesc,
                                void const *const *inputs, void *const *outputs, void *workspace) {
        SetDefinition(inputDesc, outputDesc);
        SetIOPointer(inputs, outputs);
        SetWorkspacePtr(inputDesc, outputDesc, workspace);
    }

 public:
    bool supportsFormatCombination(int32_t io_index, PluginTensorDesc const *io,
                                   int32_t num_inputs, int32_t num_outputs) noexcept override {
        assert(io != nullptr);
        assert(num_inputs == getNbInputs());
        assert(num_outputs == getNbOutputs());

        PluginTensorDesc const &desc = io[io_index];
        bool res = true;
        dbg("io index %d\n",io_index);
        switch (io_index) {
            #define TRT_PLUGIN_INPUT_SUPPORTS_FORMAT(I, TYPE, NAME, ...)                                                \
            case decltype(in.NAME)::index: {                                                                            \
                dbg(#NAME " type = %s, test with type = %s, format = %d\n",                                             \
                data2str(TypeInfo<TYPE>::data_type),data2str(desc.type),(int)desc.format);                              \
                res = (TypeInfo<TYPE>::data_type == desc.type) && (desc.format == TensorFormat::kLINEAR);               \
                break;                                                                                                  \
            }
            TRT_ENUM(INPUT, TRT_PLUGIN_INPUT_SUPPORTS_FORMAT)

            #define TRT_PLUGIN_OUTPUT_SUPPORTS_FORMAT(I, TYPE, NAME, ...)                                               \
            case IN::N + decltype(out.NAME)::index: {                                                                   \
                dbg(#NAME " type = %s, test with type = %s, format = %d\n",                                             \
                data2str(TypeInfo<TYPE>::data_type),data2str(desc.type),(int)desc.format);                              \
                res = (TypeInfo<TYPE>::data_type == desc.type) && (desc.format == TensorFormat::kLINEAR);               \
                break;                                                                                                  \
            }
            TRT_ENUM(OUTPUT, TRT_PLUGIN_OUTPUT_SUPPORTS_FORMAT)

            default: {
                assert(0);
            }
        }
        return res;
    }

    DimsExprs getOutputDimensions(int32_t output_index, DimsExprs const *inputs,
                                  int32_t nbInputs, IExprBuilder &exprBuilder) noexcept override {
        #define outputs(nin, ndim, ...)
        #define inputs(INDEX)  inputs[INDEX] TRT_PLUGIN_DEFINE_INPUT_DIM
        #define TRT_PLUGIN_DEFINE_INPUT_DIM(NDIM) .d[NDIM]
        #define TRT_PLUGIN_DEFINE_OUTPUT_DIMENSION(I, NAME, ...)                                                       \
        const auto NAME = __VA_ARGS__;
        TRT_ENUM(DEFINE, TRT_PLUGIN_DEFINE_OUTPUT_DIMENSION)
        #undef outputs
        #undef inputs

        assert(0 <= output_index && output_index < this->getNbOutputs());
        nvinfer1::DimsExprs odim{};

        switch (output_index) {
            #define dim(...) (__VA_ARGS__)
            #define TRT_PLUGIN_OUTPUT_DIMENSION_(Z, N, X) odim.d[N] = TryExpr(BOOST_PP_TUPLE_ELEM(N,X),exprBuilder);
            #define TRT_PLUGIN_OUTPUT_DIMENSION(I, TYPE, NAME, ...)                                                     \
            case decltype(out.NAME)::index: {                                                                           \
                BOOST_PP_REPEAT(BOOST_PP_TUPLE_SIZE(__VA_ARGS__),TRT_PLUGIN_OUTPUT_DIMENSION_,__VA_ARGS__)              \
                odim.nbDims = BOOST_PP_TUPLE_SIZE(__VA_ARGS__);                                                         \
                dbg("%s shape(%s)\n",#NAME,to_string(odim).c_str());                                                    \
                break;                                                                                                  \
            }
            TRT_ENUM(OUTPUT, TRT_PLUGIN_OUTPUT_DIMENSION)
            default: {
                assert(false);
            }
        }
        #undef dim
        return odim;
    }

    DataType getOutputDataType(int32_t output_index, DataType const *input_types,
                               int32_t num_inputs) const noexcept override {
        assert(input_types != nullptr);

        DataType output_type;
        switch (output_index) {
            #define TRT_PLUGIN_OUTPUT_DATA_TYPE(I, TYPE, NAME, ...)                                                     \
            case decltype(out.NAME)::index: {                                                                           \
                output_type =  TypeInfo<TYPE>::data_type;                                                               \
                dbg("%s %s\n", #NAME, data2str(output_type));                                                           \
                break;                                                                                                  \
            }
            TRT_ENUM(OUTPUT, TRT_PLUGIN_OUTPUT_DATA_TYPE)
            default: {
                assert(0);
            }
        }
        return output_type;
    }

    size_t getWorkspaceSize(PluginTensorDesc const *inputs, int32_t num_inputs,
                            PluginTensorDesc const *outputs, int32_t num_outputs) const noexcept override {
        #define TRT_PLUGIN_GET_WORKSPACE_SIZE(I, TYPE, NAME, ...)                                                       \
        elems[I] *= ws.NAME.dsize;                                                                                      \
        total += SizeAlign256(elems[I]);                                                                                \
        dbg("%s %ld total(%ld)\n", #NAME, elems[I], total);

        size_t total = 0;
        size_t elems[WS::N];
        GetWorkSpaceElems(inputs, outputs, elems);
        TRT_ENUM(WORKSPACE, TRT_PLUGIN_GET_WORKSPACE_SIZE)
        return total;
    };

    int32_t enqueue(PluginTensorDesc const *inputDesc, PluginTensorDesc const *outputDesc,
                    void const *const *inputs, void *const *outputs, void *workspace,
                    cudaStream_t stream) noexcept override {

        #define TRT_PLUGIN_ENQUEUE_IO_INFORMATION1(I, TYPE, NAME, ...)                                                  \
        dbg("inputs(" #I ") - %s dim(%s) format(%d)\n",                                                                 \
        #NAME,to_string(inputDesc[I].dims).c_str(),(int)inputDesc[I].format);
        #define TRT_PLUGIN_ENQUEUE_IO_INFORMATION2(I, TYPE, NAME, ...)                                                  \
        dbg("output(" #I ") - %s dim(%s) format(%d)\n",                                                                 \
        #NAME,to_string(outputDesc[I].dims).c_str(),(int)outputDesc[I].format);

        TRT_ENUM(INPUT, TRT_PLUGIN_ENQUEUE_IO_INFORMATION1)
        TRT_ENUM(OUTPUT, TRT_PLUGIN_ENQUEUE_IO_INFORMATION2)
        InitAllShapeAndPointer(inputDesc, outputDesc, inputs, outputs, workspace);
        return enqueue(stream);
    }

    int32_t enqueue(cudaStream_t stream) noexcept;

 public:
    void serialize(void *buffer) const noexcept override {
        assert(buffer != nullptr);
        auto *ptr = reinterpret_cast<uint8_t *>(buffer);
        auto *begin = ptr;

        #define TRT_PLUGIN_ATTRIBUTE_WRITE_BUFFER(I, TYPE, NAME, ...)                                                   \
        TypeInfo<TYPE>::WriteBuffer(attr.NAME, ptr);
        TRT_ENUM(ATTRIBUTE, TRT_PLUGIN_ATTRIBUTE_WRITE_BUFFER)

        dbg("%lu = %lu + %lu ?\n", (size_t) ptr, (size_t) begin, getSerializationSize());
        assert(ptr == begin + getSerializationSize());
    }

#define TRT_PLUGIN_ATTRIBUTE_SIZE_SUM(I, TYPE, NAME, ...) + sizeof(attr.NAME)

    [[nodiscard]] size_t getSerializationSize() const noexcept override {
        constexpr size_t serialization_size{0 TRT_ENUM(ATTRIBUTE, TRT_PLUGIN_ATTRIBUTE_SIZE_SUM)};

        #define TRT_PLUGIN_ATTRIBUTE_SIZE_PRINT_STR(I, TYPE, NAME, ...) " + %lu"  BOOST_PP_STRINGIZE((NAME))
        #define TRT_PLUGIN_ATTRIBUTE_SIZE_PRINT_SIZE(I, TYPE, NAME, ...) , sizeof(attr.NAME)
        dbg("%zu =" TRT_ENUM(ATTRIBUTE, TRT_PLUGIN_ATTRIBUTE_SIZE_PRINT_STR)"\n",
            serialization_size TRT_ENUM(ATTRIBUTE, TRT_PLUGIN_ATTRIBUTE_SIZE_PRINT_SIZE));
        return serialization_size;
    }

 public:
    TrTPluginImpl() = delete;

    TrTPluginImpl(void const *data, size_t length) {
        assert(data != nullptr);
        auto const *d = reinterpret_cast<uint8_t const *>(data);
        auto const *a = d;

        #define TRT_PLUGIN_ATTRIBUTE_CONSTRUCTION(I, TYPE, NAME, ...)                                                   \
        TypeInfo<TYPE>::ReadBuffer(attr.NAME, d);                                                                       \
        d+=sizeof(attr.NAME);
        TRT_ENUM(ATTRIBUTE, TRT_PLUGIN_ATTRIBUTE_CONSTRUCTION)
        dbg("deserialization\n");
        assert(d == a + length);
    }

#define TRT_PLUGIN_ATTRIBUTE_CONSTRUCTION_ARGS(I, TYPE, NAME, ...) , const NAME##_t &_##NAME


    TrTPluginImpl(void *TRT_ENUM(ATTRIBUTE, TRT_PLUGIN_ATTRIBUTE_CONSTRUCTION_ARGS)) {
        #define TRT_PLUGIN_ATTRIBUTE_CONSTRUCTION_ARGS_2(I, TYPE, NAME, ...)                                            \
        TypeInfo<TYPE>::DeepCopy(attr.NAME,_##NAME);
        TRT_ENUM(ATTRIBUTE, TRT_PLUGIN_ATTRIBUTE_CONSTRUCTION_ARGS_2)
    }

    [[nodiscard]] IPluginV2DynamicExt *clone() const noexcept override {
        #define TRT_PLUGIN_ATTRIBUTE_CONSTRUCTION_ARGS3(I, TYPE, NAME, ...) , attr.NAME
        auto *plugin = new TrTPluginImpl(nullptr TRT_ENUM(ATTRIBUTE, TRT_PLUGIN_ATTRIBUTE_CONSTRUCTION_ARGS3));
        plugin->setPluginNamespace();
        return plugin;
    }

    int32_t initialize() noexcept override;

    void terminate() noexcept override;

    void destroy() noexcept override {
        delete this;
    }

    friend class TrTPluginBase;
};

class TrTPluginCreator : public IPluginCreator {
    TrTPluginBase::ATTR attr;

 public:
    [[nodiscard]] char const *getPluginName() const noexcept override {
        return BOOST_PP_STRINGIZE(TRT_PLUGIN_NAME);
    }

    [[nodiscard]] char const *getPluginVersion() const noexcept override {
        return TRT_PLUGIN_VERSION;
    }

    PluginFieldCollection const *getFieldNames() noexcept override {
        return &field_collection_;
    }

    IPluginV2 *deserializePlugin(char const *name, void const *data, size_t length) noexcept override {
        return new TrTPluginImpl(data, length);
    }

    void setPluginNamespace(char const *pluginNamespace) noexcept override {
        assert(pluginNamespace != nullptr);
        namespace_ = pluginNamespace;
    }

    [[nodiscard]] char const *getPluginNamespace() const noexcept override {
        return namespace_.c_str();
    }

 public:
    TrTPluginCreator() {
        #define TRT_PLUGIN_CREATOR_SET_ATTR(I, TYPE, NAME, ...)                                                         \
        plugin_attributes_.emplace_back(                                                                                \
            PluginField(#NAME, nullptr,                                                                                 \
            TypeInfo<decltype(attr.NAME)>::field_type,                                                                  \
            TypeInfo<decltype(attr.NAME)>::len)                                                                         \
        );
        plugin_attributes_.clear();
        TRT_ENUM(ATTRIBUTE, TRT_PLUGIN_CREATOR_SET_ATTR)
        field_collection_.nbFields = plugin_attributes_.size();
        field_collection_.fields = plugin_attributes_.data();
    }

    IPluginV2 *createPlugin(char const *name, PluginFieldCollection const *fc) noexcept override {
        #define TRT_PLUGIN_CREATOR_CREATE_PLUGIN_1(I, TYPE, NAME, ...)                                                  \
        decltype(attr.NAME) NAME;
        TRT_ENUM(ATTRIBUTE, TRT_PLUGIN_CREATOR_CREATE_PLUGIN_1)

        assert(fc != nullptr);
        int32_t num_fields = fc->nbFields;
        for (auto i = 0; i < num_fields; ++i) {
            auto &field = fc->fields[i];
            #define TRT_PLUGIN_CREATOR_CREATE_PLUGIN_2(I, TYPE, NAME, ...)                                              \
            if(!strcmp(field.name, #NAME)){                                                                             \
                TypeInfo<decltype(NAME)>::ReadBuffer(NAME, field.data);                                                 \
            }else
            TRT_ENUM(ATTRIBUTE, TRT_PLUGIN_CREATOR_CREATE_PLUGIN_2) {
                assert(0);
            }
        }
        #define TRT_PLUGIN_CREATOR_CREATE_PLUGIN_3(I, TYPE, NAME, ...) , NAME
        IPluginV2 *plugin = new TrTPluginImpl(nullptr TRT_ENUM(ATTRIBUTE, TRT_PLUGIN_CREATOR_CREATE_PLUGIN_3));
        return plugin;
    }

 private:
    static nvinfer1::PluginFieldCollection field_collection_;
    static std::vector<nvinfer1::PluginField> plugin_attributes_;
    std::string namespace_;
};

PluginFieldCollection TrTPluginCreator::field_collection_{};
std::vector<PluginField> TrTPluginCreator::plugin_attributes_;
REGISTER_TENSORRT_PLUGIN(TrTPluginCreator);
}

#endif //POINT_DETECTION_PLUGIN_AUTO_DECLARE_H
