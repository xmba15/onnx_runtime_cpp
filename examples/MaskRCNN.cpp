/**
 * @file    MaskRCNN.cpp
 *
 * @author  btran
 *
 * @date    2020-05-18
 *
 * Copyright (c) organization
 *
 */

#include <cstring>

#include "MaskRCNN.hpp"

namespace Ort
{
MaskRCNN::MaskRCNN(const uint16_t numClasses,     //
                   const std::string& modelPath,  //
                   const std::optional<size_t>& gpuIdx,
                   const std::optional<std::vector<std::vector<int64_t>>>& inputShapes)
    : ImageRecognitionOrtSessionHandlerBase(numClasses, modelPath, gpuIdx, inputShapes)
{
}

MaskRCNN::~MaskRCNN()
{
}

void MaskRCNN::preprocess(float* dst,                     //
                          const unsigned char* src,       //
                          const int64_t targetImgWidth,   //
                          const int64_t targetImgHeight,  //
                          const int numChannels,          //
                          const int64_t offsetPadW,       //
                          const int64_t offsetPadH,       //
                          const std::vector<float>& meanVal) const
{
    if (!meanVal.empty()) {
        for (int i = 0; i < targetImgHeight; ++i) {
            for (int j = 0; j < targetImgWidth; ++j) {
                for (int c = 0; c < numChannels; ++c) {
                    dst[c * targetImgHeight * targetImgWidth + i * targetImgWidth + j] =
                        src[i * targetImgWidth * numChannels + j * numChannels + c] - meanVal[c];
                }
            }
        }
    } else {
        for (int i = 0; i < targetImgHeight; ++i) {
            for (int j = 0; j < targetImgWidth; ++j) {
                for (int c = 0; c < numChannels; ++c) {
                    dst[c * targetImgHeight * targetImgWidth + i * targetImgWidth + j] =
                        src[i * targetImgWidth * numChannels + j * numChannels + c];
                }
            }
        }
    }

    for (int i = targetImgHeight; i < targetImgHeight + offsetPadH; ++i) {
        for (int j = targetImgWidth; j < targetImgWidth + offsetPadW; ++j) {
            for (int c = 0; c < numChannels; ++c) {
              dst[c * targetImgHeight * targetImgWidth + i * targetImgWidth + j] = 0;
            }
        }
    }
}

}  // namespace Ort
