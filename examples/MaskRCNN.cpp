/**
 * @file    MaskRCNN.cpp
 *
 * @author  btran
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
                          const float* src,               //
                          const int64_t targetImgWidth,   //
                          const int64_t targetImgHeight,  //
                          const int numChannels) const
{
    for (int c = 0; c < numChannels; ++c) {
        for (int i = 0; i < targetImgHeight; ++i) {
            for (int j = 0; j < targetImgWidth; ++j) {
                dst[c * targetImgHeight * targetImgWidth + i * targetImgWidth + j] =
                    src[i * targetImgWidth * numChannels + j * numChannels + c];
            }
        }
    }
}

void MaskRCNN::preprocess(float* dst,                     //
                          const cv::Mat& imgSrc,          //
                          const int64_t targetImgWidth,   //
                          const int64_t targetImgHeight,  //
                          const int numChannels) const
{
    for (int i = 0; i < targetImgHeight; ++i) {
        for (int j = 0; j < targetImgWidth; ++j) {
            for (int c = 0; c < numChannels; ++c) {
                dst[c * targetImgHeight * targetImgWidth + i * targetImgWidth + j] = imgSrc.ptr<float>(i, j)[c];
            }
        }
    }
}

}  // namespace Ort
