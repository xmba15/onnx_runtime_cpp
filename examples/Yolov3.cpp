/**
 * @file    Yolov3.cpp
 *
 * @author  btran
 *
 * @date    2020-05-31
 *
 * Copyright (c) organization
 *
 */

#include "Yolov3.hpp"

namespace Ort
{
Yolov3::Yolov3(const uint16_t numClasses,     //
               const std::string& modelPath,  //
               const std::optional<size_t>& gpuIdx, const std::optional<std::vector<std::vector<int64_t>>>& inputShapes)
    : ImageRecognitionOrtSessionHandlerBase(numClasses, modelPath, gpuIdx, inputShapes)
{
}

Yolov3::~Yolov3()
{
}

void Yolov3::preprocess(float* dst,                     //
                        const unsigned char* src,       //
                        const int64_t targetImgWidth,   //
                        const int64_t targetImgHeight,  //
                        const int numChannels) const
{
    for (int c = 0; c < numChannels; ++c) {
        for (int i = 0; i < targetImgHeight; ++i) {
            for (int j = 0; j < targetImgWidth; ++j) {
                dst[c * targetImgHeight * targetImgWidth + i * targetImgWidth + j] =
                    src[i * targetImgWidth * numChannels + j * numChannels + c] / 255.0;
            }
        }
    }
}
}  // namespace Ort
