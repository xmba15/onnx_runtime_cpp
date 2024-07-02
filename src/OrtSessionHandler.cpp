/**
 * @file    OrtSessionHandler.cpp
 *
 * @author  btran
 *
 */


#include <../onnxruntime/include/onnxruntime/core/session/onnxruntime_c_api.h>
#include <../onnxruntime/include/onnxruntime/core/session/onnxruntime_cxx_api.h>


#if ENABLE_TENSORRT
#include <onnxruntime/core/providers/tensorrt/tensorrt_provider_factory.h>
#endif

#include <ort_utility/ort_utility.hpp>

#include <algorithm>
#include <cassert>
#include <numeric>
#include <sstream>

namespace
{
std::string toString(const ONNXTensorElementDataType dataType)
{
    switch (dataType) {
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT: {
            return "float";
        }
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8: {
            return "uint8_t";
        }
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8: {
            return "int8_t";
        }
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16: {
            return "uint16_t";
        }
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16: {
            return "int16_t";
        }
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32: {
            return "int32_t";
        }
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64: {
            return "int64_t";
        }
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING: {
            return "string";
        }
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL: {
            return "bool";
        }
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16: {
            return "float16";
        }
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE: {
            return "double";
        }
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32: {
            return "uint32_t";
        }
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64: {
            return "uint64_t";
        }
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64: {
            return "complex with float32 real and imaginary components";
        }
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128: {
            return "complex with float64 real and imaginary components";
        }
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16: {
            return "complex with float64 real and imaginary components";
        }
        default:
            return "undefined";
    }
}
}  // namespace

namespace Ort
{
//-----------------------------------------------------------------------------//
// OrtSessionHandlerIml Definition
//-----------------------------------------------------------------------------//

class OrtSessionHandler::OrtSessionHandlerIml
{
 public:
    OrtSessionHandlerIml(const std::string& modelPath,         //
                         const std::optional<size_t>& gpuIdx,  //
                         const std::optional<std::vector<std::vector<int64_t>>>& inputShapes);
    ~OrtSessionHandlerIml();

    std::vector<DataOutputType> operator()(const std::vector<float*>& inputData) const;

    void updateInputShapes(const std::vector<std::vector<int64_t>>& inputShapes)
    {
        if (inputShapes.size() != m_numInputs) {
            DEBUG_LOG("inputShapes must be of size: %d", m_numInputs);
            return;
        }
        m_inputShapes = inputShapes;

        for (int i = 0; i < m_numInputs; i++) {
            const auto& curInputShape = m_inputShapes[i];
            m_inputTensorSizes[i] =
                std::accumulate(std::begin(curInputShape), std::end(curInputShape), 1, std::multiplies<int64_t>());
        }
    }

 private:
    void initSession();
    void initModelInfo();

 private:
    std::string m_modelPath;

    mutable Ort::Session m_session;
    Ort::Env m_env;
    Ort::AllocatorWithDefaultOptions m_ortAllocator;

    std::optional<size_t> m_gpuIdx;

    std::vector<std::vector<int64_t>> m_inputShapes;
    std::vector<std::vector<int64_t>> m_outputShapes;

    std::vector<int64_t> m_inputTensorSizes;
    std::vector<int64_t> m_outputTensorSizes;

    uint8_t m_numInputs;
    uint8_t m_numOutputs;

    std::vector<char*> m_inputNodeNames;
    std::vector<char*> m_outputNodeNames;

    bool m_inputShapesProvided = false;
};

//-----------------------------------------------------------------------------//
// OrtSessionHandler
//-----------------------------------------------------------------------------//

OrtSessionHandler::OrtSessionHandler(const std::string& modelPath,         //
                                     const std::optional<size_t>& gpuIdx,  //
                                     const std::optional<std::vector<std::vector<int64_t>>>& inputShapes)
    : m_piml(std::make_unique<OrtSessionHandlerIml>(modelPath,  //
                                                    gpuIdx,     //
                                                    inputShapes))
{
}

OrtSessionHandler::~OrtSessionHandler() = default;

std::vector<OrtSessionHandler::DataOutputType>
OrtSessionHandler::operator()(const std::vector<float*>& inputImgData) const
{
    return this->m_piml->operator()(inputImgData);
}

//-----------------------------------------------------------------------------//
// piml class implementation
//-----------------------------------------------------------------------------//

OrtSessionHandler::OrtSessionHandlerIml::OrtSessionHandlerIml(
    const std::string& modelPath,         //
    const std::optional<size_t>& gpuIdx,  //
    const std::optional<std::vector<std::vector<int64_t>>>& inputShapes)
    : m_modelPath(modelPath)
    , m_session(nullptr)
    , m_env(nullptr)
    , m_ortAllocator()
    , m_gpuIdx(gpuIdx)
    , m_inputShapes()
    , m_outputShapes()
    , m_numInputs(0)
    , m_numOutputs(0)
    , m_inputNodeNames()
    , m_outputNodeNames()
{
    this->initSession();

    if (inputShapes.has_value()) {
        m_inputShapesProvided = true;
        m_inputShapes = inputShapes.value();
    }

    this->initModelInfo();
}

OrtSessionHandler::OrtSessionHandlerIml::~OrtSessionHandlerIml()
{
    for (auto& elem : this->m_inputNodeNames) {
        free(elem);
        elem = nullptr;
    }
    this->m_inputNodeNames.clear();

    for (auto& elem : this->m_outputNodeNames) {
        free(elem);
        elem = nullptr;
    }
    this->m_outputNodeNames.clear();
}

void OrtSessionHandler::OrtSessionHandlerIml::initSession()
{
#if ENABLE_DEBUG
    m_env = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "test");
#else
    m_env = Ort::Env(ORT_LOGGING_LEVEL_ERROR, "test");
#endif
    Ort::SessionOptions sessionOptions;

    sessionOptions.SetIntraOpNumThreads(1);
    // tensorrt options can be customized into sessionOptions
    // https://onnxruntime.ai/docs/execution-providers/TensorRT-ExecutionProvider.html

#if ENABLE_GPU
    if (m_gpuIdx.has_value()) {
        Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_CUDA(sessionOptions, m_gpuIdx.value()));
#if ENABLE_TENSORRT
        Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_Tensorrt(sessionOptions, m_gpuIdx.value()));
#endif
    }
#endif

    sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    m_session = Ort::Session(m_env, m_modelPath.c_str(), sessionOptions);
    m_numInputs = m_session.GetInputCount();
    DEBUG_LOG("Model number of inputs: %d\n", m_numInputs);

    m_inputNodeNames.reserve(m_numInputs);
    m_inputTensorSizes.reserve(m_numInputs);

    m_numOutputs = m_session.GetOutputCount();
    DEBUG_LOG("Model number of outputs: %d\n", m_numOutputs);

    m_outputNodeNames.reserve(m_numOutputs);
    m_outputTensorSizes.reserve(m_numOutputs);
}

void OrtSessionHandler::OrtSessionHandlerIml::initModelInfo()
{
    for (int i = 0; i < m_numInputs; i++) {
        if (!m_inputShapesProvided) {
            Ort::TypeInfo typeInfo = m_session.GetInputTypeInfo(i);
            auto tensorInfo = typeInfo.GetTensorTypeAndShapeInfo();

            m_inputShapes.emplace_back(tensorInfo.GetShape());
        }

        const auto& curInputShape = m_inputShapes[i];

        m_inputTensorSizes.emplace_back(
            std::accumulate(std::begin(curInputShape), std::end(curInputShape), 1, std::multiplies<int64_t>()));

#if ORT_API_VERSION > 12
        m_inputNodeNames.emplace_back(strdup(m_session.GetInputNameAllocated(i, m_ortAllocator).get()));
#else
        char* inputName = m_session.GetInputName(i, m_ortAllocator);
        m_inputNodeNames.emplace_back(strdup(inputName));
        m_ortAllocator.Free(inputName);
#endif
    }

    {
#if ENABLE_DEBUG
        std::stringstream ssInputs;
        ssInputs << "Model input shapes: ";
        ssInputs << m_inputShapes << std::endl;
        ssInputs << "Model input node names: ";
        ssInputs << m_inputNodeNames << std::endl;
        DEBUG_LOG("%s\n", ssInputs.str().c_str());
#endif
    }

    for (int i = 0; i < m_numOutputs; ++i) {
        Ort::TypeInfo typeInfo = m_session.GetOutputTypeInfo(i);
        auto tensorInfo = typeInfo.GetTensorTypeAndShapeInfo();

        m_outputShapes.emplace_back(tensorInfo.GetShape());

#if ORT_API_VERSION > 12
        m_outputNodeNames.emplace_back(strdup(m_session.GetOutputNameAllocated(i, m_ortAllocator).get()));
#else
        char* outputName = m_session.GetOutputName(i, m_ortAllocator);
        m_outputNodeNames.emplace_back(strdup(outputName));
        m_ortAllocator.Free(outputName);
#endif
    }

    {
#if ENABLE_DEBUG
        std::stringstream ssOutputs;
        ssOutputs << "Model output shapes: ";
        ssOutputs << m_outputShapes << std::endl;
        ssOutputs << "Model output node names: ";
        ssOutputs << m_outputNodeNames << std::endl;
        DEBUG_LOG("%s\n", ssOutputs.str().c_str());
#endif
    }
}

std::vector<OrtSessionHandler::DataOutputType>
OrtSessionHandler::OrtSessionHandlerIml::operator()(const std::vector<float*>& inputData) const
{
    if (m_numInputs != inputData.size()) {
        DEBUG_LOG("m_numInputs:%d, input size:%ld", m_numInputs, inputData.size());
        throw std::runtime_error("Mismatch size of input data");
    }

    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    std::vector<Ort::Value> inputTensors;
    inputTensors.reserve(m_numInputs);

    for (int i = 0; i < m_numInputs; ++i) {
        inputTensors.emplace_back(std::move(
            Ort::Value::CreateTensor<float>(memoryInfo, const_cast<float*>(inputData[i]), m_inputTensorSizes[i],
                                            m_inputShapes[i].data(), m_inputShapes[i].size())));
    }

    auto outputTensors = m_session.Run(Ort::RunOptions{nullptr}, m_inputNodeNames.data(), inputTensors.data(),
                                       m_numInputs, m_outputNodeNames.data(), m_numOutputs);

    assert(outputTensors.size() == m_numOutputs);
    std::vector<DataOutputType> outputData;
    outputData.reserve(m_numOutputs);

    int count = 1;
    for (auto& elem : outputTensors) {
        DEBUG_LOG("type of input %d: %s", count++, toString(elem.GetTensorTypeAndShapeInfo().GetElementType()).c_str());
        outputData.emplace_back(
            std::make_pair(std::move(elem.GetTensorMutableData<float>()), elem.GetTensorTypeAndShapeInfo().GetShape()));
    }

    return outputData;
}

void OrtSessionHandler::updateInputShapes(const std::vector<std::vector<int64_t>>& inputShapes)
{
    m_piml->updateInputShapes(inputShapes);
}
}  // namespace Ort
