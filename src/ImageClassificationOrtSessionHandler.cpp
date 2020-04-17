/**
 * @file    ImageClassificationOrtSessionHandler.cpp
 *
 * @author  btran
 *
 * Copyright (c) organization
 *
 */

#include <cassert>
#include <cstring>
#include <sstream>

#include "ort_utility/ort_utility.hpp"

namespace Ort
{
ImageClassificationOrtSessionHandler::ImageClassificationOrtSessionHandler(
    const uint16_t numClasses,     //
    const std::string& modelPath,  //
    const std::optional<size_t>& gpuIdx, const std::optional<std::vector<std::vector<int64_t>>>& inputShapes)
    : ImageRecognitionOrtSessionHandlerBase(numClasses, modelPath, gpuIdx, inputShapes)
{
}

ImageClassificationOrtSessionHandler::~ImageClassificationOrtSessionHandler()
{
}

std::vector<std::pair<int, float>>
ImageClassificationOrtSessionHandler::topK(const std::vector<float*>& inferenceOutput,  //
                                           const uint16_t k,                            //
                                           const bool useSoftmax) const
{
    const uint16_t realK = std::max(std::min(k, m_numClasses), static_cast<uint16_t>(1));

    assert(inferenceOutput.size() == 1);
    float* processData = inferenceOutput[0];
    if (useSoftmax) {
        softmax(processData, m_numClasses);
    }

    std::vector<std::pair<int, float>> ps;
    ps.reserve(m_numClasses);

    for (int i = 0; i < m_numClasses; ++i) {
        ps.emplace_back(std::make_pair(i, processData[i]));
    }

    std::sort(ps.begin(), ps.end(), [](const auto& elem1, const auto& elem2) { return elem1.second > elem2.second; });

    return std::vector<std::pair<int, float>>(ps.begin(), ps.begin() + realK);
}

std::string ImageClassificationOrtSessionHandler::topKToString(const std::vector<float*>& inferenceOutput,  //
                                                               const uint16_t k,                            //
                                                               const bool useSoftmax) const
{
    auto ps = this->topK(inferenceOutput, k, useSoftmax);

    std::stringstream ss;

    if (m_classNames.size() == 0) {
        for (const auto& elem : ps) {
            ss << elem.first << " : " << elem.second << std::endl;
        }
    } else {
        for (const auto& elem : ps) {
            ss << elem.first << " : " << m_classNames[elem.first] << " : " << elem.second << std::endl;
        }
    }

    return ss.str();
}
}  // namespace Ort
