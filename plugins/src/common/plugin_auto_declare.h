//
// Created by nrsl on 23-4-15.
//

#ifndef POINT_DETECTION_PLUGIN_AUTO_DECLARE_H
#define POINT_DETECTION_PLUGIN_AUTO_DECLARE_H

#include <cstring>
#include <cassert>

#include <string>
#include <vector>

#include <NvInferRuntime.h>
#include "common.h"

#define TRT__CAT(a, b) a##b
#define TRT_CAT(a, b) TRT__CAT(a,b)
#define TRT__STR(a) #a
#define TRT_STR(a) TRT__STR(a)
#define TRT_FWD(...) __VA_ARGS__

#define TENSORRT_PLUGIN__Setting(...) (__VA_ARGS__)
#define TENSORRT_PLUGIN_SETTING  TRT_CAT(TENSORRT_PLUGIN__,TENSORRT_PLUGIN)

#define TENSORRT_PLUGIN___Name(n) n
#define TENSORRT_PLUGIN__NAME(n, ...) TRT_CAT(TENSORRT_PLUGIN___,n)
#define TENSORRT_PLUGIN_SETTING_NAME TRT_FWD(TENSORRT_PLUGIN__NAME TENSORRT_PLUGIN_SETTING)

#define TENSORRT_PLUGIN___Version(v) v
#define TENSORRT_PLUGIN__VERSION(_, v, ...) TRT_CAT(TENSORRT_PLUGIN___,v)
#define TENSORRT_PLUGIN_SETTING_VERSION TRT_FWD(TENSORRT_PLUGIN__VERSION TENSORRT_PLUGIN_SETTING)

#define TENSORRT_PLUGIN___ATTR(...) __VA_ARGS__
#define TENSORRT_PLUGIN__ATTR(name, version, inputs, outputs, workspace, attributes) TENSORRT_PLUGIN___ATTR attributes
#define TENSORRT_PLUGIN_SETTING_ATTR TRT_FWD(TENSORRT_PLUGIN__ATTR TENSORRT_PLUGIN_SETTING)

#define TENSORRT_PLUGIN___OUTPUT(...) __VA_ARGS__
#define TENSORRT_PLUGIN__OUTPUT(name, version, inputs, outputs, workspace, attributes) TENSORRT_PLUGIN___OUTPUT outputs
#define TENSORRT_PLUGIN_SETTING_OUTPUT TRT_FWD(TENSORRT_PLUGIN__OUTPUT TENSORRT_PLUGIN_SETTING)

#define TENSORRT_PLUGIN___WORKSPACE(...) __VA_ARGS__
#define TENSORRT_PLUGIN__WORKSPACE(name, version, inputs, outputs, workspace, attributes) TENSORRT_PLUGIN___WORKSPACE workspace
#define TENSORRT_PLUGIN_SETTING_WORKSPACE TRT_FWD(TENSORRT_PLUGIN__WORKSPACE TENSORRT_PLUGIN_SETTING)

#define TENSORRT_PLUGIN___INPUT(...) __VA_ARGS__
#define TENSORRT_PLUGIN__INPUT(name, version, inputs, outputs, workspace, attributes) TENSORRT_PLUGIN___INPUT inputs
#define TENSORRT_PLUGIN_SETTING_INPUT TRT_FWD(TENSORRT_PLUGIN__INPUT TENSORRT_PLUGIN_SETTING)

#define TENSORRT_PLUGIN_Dim1(_1)       Dimension(_1)
#define TENSORRT_PLUGIN_Dim2(_1, _2)    TENSORRT_PLUGIN_Dim1(_1) Dimension(_2)
#define TENSORRT_PLUGIN_Dim3(_1, _2, _3) TENSORRT_PLUGIN_Dim2(_1,_2) Dimension(_3)
#define TENSORRT_PLUGIN_Dim4(_1, _2, _3, _4) TENSORRT_PLUGIN_Dim3(_1,_2,_3) Dimension(_4)
#define TENSORRT_PLUGIN_Dim5(_1, _2, _3, _4, _5) TENSORRT_DIM4(_1,_2,_3,_4) Dimension(_5)

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#define TENSORRT_PLUGIN_NAME TRT_CAT(TENSORRT_PLUGIN_SETTING_NAME,Plugin)

namespace nvinfer1::plugin {

template<typename C>
constexpr PluginFieldType type2field() {
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
constexpr DataType type2data() {
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

template<typename T>
struct TypeInfo {
    using type = T;
    static constexpr int len = 1;
    static constexpr auto field_type = type2field<std::remove_cv_t<T>>();
    static constexpr auto data_type = type2data<std::remove_cv_t<T>>();

    static void DeepCopy(T &val, const T &data) {
        val = data;
    }

    static void ReadBuffer(T &val, const void *const data) {
        val = static_cast<T const *>(data)[0];
    }

    template<typename buffer_type>
    static void WriteBuffer(const T &val, buffer_type *&data) {
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
        auto *ptr = reinterpret_cast<T *>( data);
        for (int i = 0; i < N; ++i) {
            TypeInfo<T>::WriteBuffer(val[i], ptr);
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
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class TENSORRT_PLUGIN_NAME : public nvinfer1::IPluginV2DynamicExt, public TRT_CAT(TENSORRT_PLUGIN_SETTING_NAME, User) {
 private:
    std::string namespace_;
#define Attribute(t, n, ...)  using n##_t=t; n##_t n=__VA_ARGS__;
    TENSORRT_PLUGIN_SETTING_ATTR
#undef Attribute
#define Workspace(t, n, ...)  + 1
    mutable size_t workspace_size_[TENSORRT_PLUGIN_SETTING_WORKSPACE]
#undef Workspace
#define Workspace(t, n, ...)  0,
            = {TENSORRT_PLUGIN_SETTING_WORKSPACE};
#undef Workspace
#define Workspace(t, n, ...)  t* n{nullptr};
    TENSORRT_PLUGIN_SETTING_WORKSPACE;
#undef Workspace
#define Input(t, n, ...) const t * n{nullptr};
    TENSORRT_PLUGIN_SETTING_INPUT
#undef Input
#define Output(t, n, ...) t * n{nullptr};
    TENSORRT_PLUGIN_SETTING_OUTPUT
#undef Output
 public:
    TENSORRT_PLUGIN_NAME() = delete;

    TENSORRT_PLUGIN_NAME(void const *data, size_t length) {
        assert(data != nullptr);
        uint8_t const *d = reinterpret_cast<uint8_t const *>(data);
        auto const *a = d;

#define Attribute(t, n, ...) TypeInfo<n##_t>::ReadBuffer(n, d); d+=sizeof(n##_t);
        TENSORRT_PLUGIN_SETTING_ATTR
#undef Attribute
        assert(d == a + length);
    }

#define Attribute(t, n, ...) ,const n##_t &_##n

    TENSORRT_PLUGIN_NAME(void *TENSORRT_PLUGIN_SETTING_ATTR) {
#undef Attribute
#define Attribute(t, n, ...)  TypeInfo<t>::DeepCopy(n,_##n);
        TENSORRT_PLUGIN_SETTING_ATTR
#undef Attribute
    }

    nvinfer1::IPluginV2DynamicExt *clone() const noexcept override {

#define Attribute(t, n, ...) , n
        auto *plugin = new TENSORRT_PLUGIN_NAME(nullptr TENSORRT_PLUGIN_SETTING_ATTR);
#undef Attribute
        plugin->setPluginNamespace(namespace_.c_str());
        return plugin;
    }

    nvinfer1::DimsExprs getOutputDimensions(int32_t outputIndex, nvinfer1::DimsExprs const *inputs, int32_t nbInputs,
                                            nvinfer1::IExprBuilder &exprBuilder) noexcept override {
        assert(0 <= outputIndex && outputIndex < this->getNbOutputs());

#define Input(nin, ndim, ...)  inputs[nin].d[ndim]
#define Dimension(...)  dim.d[n_dim++]=__VA_ARGS__;
#define Const(x) exprBuilder.constant(x)
#define Output(t, n, ...) if (outputIndex==(n_out++)) { \
    int n_dim=0;                                        \
    TRT_CAT(TENSORRT_PLUGIN_,__VA_ARGS__)               \
    dim.nbDims = n_dim;                                 \
    } else

        int n_out = 0;
        nvinfer1::DimsExprs dim{};
        TENSORRT_PLUGIN_SETTING_OUTPUT {
            assert(false);
        }

#undef Dimension
#undef Input
#undef Const
#undef Output

        return dim;
    }

    void configurePlugin(nvinfer1::DynamicPluginTensorDesc const *in, int32_t nbInputs,
                         nvinfer1::DynamicPluginTensorDesc const *out, int32_t nbOutputs) noexcept override {
        assert(in != nullptr);
        assert(out != nullptr);
        assert(nbInputs == getNbInputs());
        assert(nbOutputs == getNbOutputs());
    }

    void InitAllPointers(PluginTensorDesc const *inputDesc,
                         PluginTensorDesc const *outputDesc,
                         void const *const *inputs,
                         void *const *outputs,
                         void *workspace) {
        setIOPointer(inputs, outputs);
        setWorkspacePtr(workspace, inputDesc, outputDesc);
    }

    void setIOPointer(void const *const *inputs, void *const *outputs) {
#define Input(t, n, ...) this->n = static_cast<const t* >(inputs[ni++]);
        int ni = 0;
        TENSORRT_PLUGIN_SETTING_INPUT
#undef Input
#define Output(t, n, ...) this->n = static_cast<t* >(outputs[no++]);
        int no = 0;
        TENSORRT_PLUGIN_SETTING_OUTPUT
#undef Output
    }

    void setWorkSpaceSize(nvinfer1::PluginTensorDesc const *inputs, nvinfer1::PluginTensorDesc const *outputs) const {
        int num_workspace = 0;
#define Dimension(...) * __VA_ARGS__
#define Input(nin, ndim, ...)  inputs[nin].dims.d[ndim]
#define Workspace(t, n, ...)  workspace_size_[num_workspace++]=sizeof(t) TRT_CAT(TENSORRT_PLUGIN_,__VA_ARGS__);
        TENSORRT_PLUGIN_SETTING_WORKSPACE
#undef Input
#undef Workspace
#undef Dimension
    }

    void setWorkspacePtr(void *workspace) {
        int num_workspace = 0;
        size_t offset = 0;
#define Workspace(t, n, ...)  n = GetOneWorkspace<t>(workspace, workspace_size_[num_workspace++], offset);
        TENSORRT_PLUGIN_SETTING_WORKSPACE
#undef Workspace
    }

    void setWorkspacePtr(void *workspace,
                         nvinfer1::PluginTensorDesc const *inputs,
                         nvinfer1::PluginTensorDesc const *outputs) {
        setWorkSpaceSize(inputs, outputs);
        setWorkspacePtr(workspace);
    }

    size_t getWorkspaceSize(nvinfer1::PluginTensorDesc const *inputs, int32_t nbInputs,
                            nvinfer1::PluginTensorDesc const *outputs, int32_t nbOutputs) const noexcept override {
        setWorkSpaceSize(inputs, outputs);
        size_t total = 0;
        for (auto &&size: workspace_size_) { total += SizeAlign256(size); }
        return total;
    };

    int32_t enqueue(nvinfer1::PluginTensorDesc const *inputDesc, nvinfer1::PluginTensorDesc const *outputDesc,
                    void const *const *inputs, void *const *outputs, void *workspace,
                    cudaStream_t stream) noexcept override {

        InitAllPointers(inputDesc, outputDesc, inputs, outputs, workspace);
        return enqueue(inputDesc, outputDesc, stream);
    }

    int32_t enqueue(nvinfer1::PluginTensorDesc const *inputDesc,
                    nvinfer1::PluginTensorDesc const *outputDesc, cudaStream_t stream) noexcept;

    nvinfer1::DataType getOutputDataType(
            int32_t index, nvinfer1::DataType const *inputTypes, int32_t nbInputs) const noexcept override {
        assert(inputTypes != nullptr);
        int i = 0;
#define Output(t, n, ...) if (index==(i++)){ /* fprintf(stderr,"getOutputDataType(%d): %d\n",index, TypeInfo<t>::data_type); */ return TypeInfo<int>::data_type; } else
        TENSORRT_PLUGIN_SETTING_OUTPUT {
            assert(false);
            return nvinfer1::DataType{};
        }
#undef Output
    }

    void serialize(void *buffer) const noexcept override {
        assert(buffer != nullptr);
        uint8_t *d = reinterpret_cast<uint8_t *>(buffer);
        auto *a = d;
#define Attribute(t, v, ...) TypeInfo<t>::WriteBuffer(v,d);
        TENSORRT_PLUGIN_SETTING_ATTR
#undef Attribute
        assert(d == a + getSerializationSize());
    }

    void setPluginNamespace(char const *pluginNamespace) noexcept override {
        assert(pluginNamespace != nullptr);
        namespace_ = pluginNamespace;
    }

    size_t getSerializationSize() const noexcept override {
#define Attribute(t, v, ...) *sizeof(t)
        static constexpr size_t serialization_size{1 TENSORRT_PLUGIN_SETTING_ATTR};
#undef Attribute
        return serialization_size;
    }

    int32_t getNbInputs() const noexcept {
#define Input(t, n, ...) (+1)
        return (0 + TENSORRT_PLUGIN_SETTING_INPUT);
#undef Input
    }

    int32_t getNbOutputs() const noexcept override {
#define Output(t, n, ...) (+1)
        return (0 + TENSORRT_PLUGIN_SETTING_OUTPUT);
#undef Output
    }

    char const *getPluginNamespace() const noexcept override {
        return namespace_.c_str();
    }

    char const *getPluginVersion() const noexcept override {
        return TENSORRT_PLUGIN_SETTING_VERSION;
    }

    char const *getPluginType() const noexcept override {
        return TRT_STR(TENSORRT_PLUGIN_SETTING_NAME);
    }

    void destroy() noexcept override {
        delete this;
    }

    void terminate() noexcept override;

    int32_t initialize() noexcept override;

    bool supportsFormatCombination(
            int32_t pos, nvinfer1::PluginTensorDesc const *inOut, int32_t nbInputs,
            int32_t nbOutputs) noexcept override {

        assert(inOut != nullptr);
        assert(nbInputs == getNbInputs());
        assert(nbOutputs == getNbOutputs());

        PluginTensorDesc const &io = inOut[pos];
        int i = 0;
#define Input(t, n, ...) if (pos==(i++)){ return (TypeInfo<t>::data_type == io.type)&&(io.format == TensorFormat::kLINEAR);} else
        TENSORRT_PLUGIN_SETTING_INPUT
#undef Input
#define Output(t, n, ...) if (pos==(i++)){ return (TypeInfo<t>::data_type == io.type)&&(io.format == TensorFormat::kLINEAR);} else
        TENSORRT_PLUGIN_SETTING_OUTPUT { assert(false); }
#undef Output
        return false;
    }
};

#define TENSORRT_PLUGIN_CREATOR_NAME TRT_CAT(TENSORRT_PLUGIN_SETTING_NAME,PluginCreator)

class TENSORRT_PLUGIN_CREATOR_NAME : public nvinfer1::IPluginCreator {
#define Attribute(t, n, ...)  using n##_t=t;
    TENSORRT_PLUGIN_SETTING_ATTR
#undef Attribute
 public:
    TENSORRT_PLUGIN_CREATOR_NAME() {
        plugin_attributes_.clear();
#define Attribute(t, v, ...)  plugin_attributes_.emplace_back(PluginField(TRT_STR(v), nullptr, TypeInfo<t>::field_type, TypeInfo<t>::len));
        TENSORRT_PLUGIN_SETTING_ATTR
#undef Attribute
        field_collection_.nbFields = plugin_attributes_.size();
        field_collection_.fields = plugin_attributes_.data();
    }

    char const *getPluginName() const noexcept override {
        return TRT_STR(TENSORRT_PLUGIN_SETTING_NAME);
    }

    char const *getPluginVersion() const noexcept override {
        return TENSORRT_PLUGIN_SETTING_VERSION;
    }

    nvinfer1::PluginFieldCollection const *getFieldNames() noexcept override {
        return &field_collection_;
    }

    nvinfer1::IPluginV2 *createPlugin(char const *name, nvinfer1::PluginFieldCollection const *fc) noexcept override {
        assert(fc != nullptr);

#define Attribute(t, n, ...)  n##_t n;
        TENSORRT_PLUGIN_SETTING_ATTR
#undef Attribute
        PluginField const *fields = fc->fields;
        int32_t num_fields = fc->nbFields;
        for (auto i = 0; i < num_fields; ++i) {
            auto &field = fields[i];
#define Attribute(t, n, ...)                                \
            if(!strcmp(field.name,#n)){                     \
                TypeInfo<t>::ReadBuffer(n, field.data);     \
            }else
            TENSORRT_PLUGIN_SETTING_ATTR {
                assert(0);
            }
#undef Attribute
        }
#define Attribute(t, n, ...)  , n
        IPluginV2 *plugin = new TENSORRT_PLUGIN_NAME(nullptr TENSORRT_PLUGIN_SETTING_ATTR);
#undef Attribute
        return plugin;
    }

    nvinfer1::IPluginV2 *
    deserializePlugin(char const *name, void const *serialData, size_t serialLength) noexcept override {
        return new TENSORRT_PLUGIN_NAME(serialData, serialLength);
    }

    void setPluginNamespace(char const *pluginNamespace) noexcept override {
        assert(pluginNamespace != nullptr);
        namespace_ = pluginNamespace;
    }

    char const *getPluginNamespace() const noexcept override {
        return namespace_.c_str();
    }

 private:
    static nvinfer1::PluginFieldCollection field_collection_;
    static std::vector<nvinfer1::PluginField> plugin_attributes_;
    std::string namespace_;
};

PluginFieldCollection TENSORRT_PLUGIN_CREATOR_NAME::field_collection_{};
std::vector<PluginField> TENSORRT_PLUGIN_CREATOR_NAME::plugin_attributes_;
REGISTER_TENSORRT_PLUGIN(TENSORRT_PLUGIN_CREATOR_NAME);
}

#endif //POINT_DETECTION_PLUGIN_AUTO_DECLARE_H
