/**
 * @file    SemanticSegmentationPaddleSegBisenetv2.cpp
 *
 * @author  btran
 *
 */

#include "SemanticSegmentationPaddleSegBisenetv2.hpp"

namespace Ort
{
SemanticSegmentationPaddleSegBisenetv2::SemanticSegmentationPaddleSegBisenetv2(
    const uint16_t numClasses,     //
    const std::string& modelPath,  //
    const std::optional<size_t>& gpuIdx, const std::optional<std::vector<std::vector<int64_t>>>& inputShapes)
    : ImageRecognitionOrtSessionHandlerBase(numClasses, modelPath, gpuIdx, inputShapes)
{
}

void SemanticSegmentationPaddleSegBisenetv2::preprocess(float* dst,                         //
                                                        const unsigned char* src,           //
                                                        int64_t targetImgWidth,             //
                                                        int64_t targetImgHeight,            //
                                                        int numChannels,                    //
                                                        const std::vector<float>& meanVal,  //
                                                        const std::vector<float>& stdVal) const
{
    for (int i = 0; i < targetImgHeight; ++i) {
        for (int j = 0; j < targetImgWidth; ++j) {
            for (int c = 0; c < numChannels; ++c) {
                dst[c * targetImgHeight * targetImgWidth + i * targetImgWidth + j] =
                    (src[i * targetImgWidth * numChannels + j * numChannels + c] / 255.0 - meanVal[c]) / stdVal[c];
            }
        }
    }
}
}  // namespace Ort
