/**
 * @file    ImageRecognitionOrtSessionHandlerBase.cpp
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
ImageRecognitionOrtSessionHandlerBase::ImageRecognitionOrtSessionHandlerBase(
    const uint16_t numClasses,     //
    const std::string& modelPath,  //
    const std::optional<size_t>& gpuIdx, const std::optional<std::vector<std::vector<int64_t>>>& inputShapes)
    : OrtSessionHandler(modelPath, gpuIdx, inputShapes)
    , m_numClasses(numClasses)
    , m_classNames()
{
    if (numClasses <= 0) {
        throw std::runtime_error("Number of classes must be more than 0\n");
    }

    m_classNames.reserve(m_numClasses);
    for (uint16_t i = 0; i < m_numClasses; ++i) {
        m_classNames.emplace_back(std::to_string(i));
    }
}

ImageRecognitionOrtSessionHandlerBase::~ImageRecognitionOrtSessionHandlerBase()
{
}

void ImageRecognitionOrtSessionHandlerBase::initClassNames(const std::vector<std::string>& classNames)
{
    if (classNames.size() != m_numClasses) {
        throw std::runtime_error("Mismatch number of classes\n");
    }

    m_classNames = classNames;
}

void ImageRecognitionOrtSessionHandlerBase::preprocess(float* dst,                         //
                                                       const unsigned char* src,           //
                                                       const int64_t targetImgWidth,       //
                                                       const int64_t targetImgHeight,      //
                                                       const int numChannels,              //
                                                       const std::vector<float>& meanVal,  //
                                                       const std::vector<float>& stdVal) const
{
    if (!meanVal.empty() && !stdVal.empty()) {
        assert(meanVal.size() == stdVal.size() && meanVal.size() == static_cast<std::size_t>(numChannels));
    }

    int64_t dataLength = targetImgHeight * targetImgWidth * numChannels;

    memcpy(dst, reinterpret_cast<const float*>(src), dataLength);

    if (!meanVal.empty() && !stdVal.empty()) {
        for (int i = 0; i < targetImgHeight; ++i) {
            for (int j = 0; j < targetImgWidth; ++j) {
                for (int c = 0; c < numChannels; ++c) {
                    dst[c * targetImgHeight * targetImgWidth + i * targetImgWidth + j] =
                        (src[i * targetImgWidth * numChannels + j * numChannels + c] / 255.0 - meanVal[c]) / stdVal[c];
                }
            }
        }
    } else {
        for (int i = 0; i < targetImgHeight; ++i) {
            for (int j = 0; j < targetImgWidth; ++j) {
                for (int c = 0; c < numChannels; ++c) {
                    dst[c * targetImgHeight * targetImgWidth + i * targetImgWidth + j] =
                        src[i * targetImgWidth * numChannels + j * numChannels + c] / 255.0;
                }
            }
        }
    }
}
}  // namespace Ort
