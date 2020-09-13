/**
 * @file    UltraLightFastGenericFaceDetector.cpp
 *
 * @author  btran
 *
 */

#include "UltraLightFastGenericFaceDetector.hpp"

namespace Ort
{
UltraLightFastGenericFaceDetector::UltraLightFastGenericFaceDetector(
    const std::string& modelPath, const std::optional<size_t>& gpuIdx,
    const std::optional<std::vector<std::vector<int64_t>>>& inputShapes)
    : ImageRecognitionOrtSessionHandlerBase(1 /* num classes */, modelPath, gpuIdx, inputShapes)
{
}

UltraLightFastGenericFaceDetector::~UltraLightFastGenericFaceDetector()
{
}

// Ref: https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB/blob/master/detect_imgs_onnx.py#L70
void UltraLightFastGenericFaceDetector::preprocess(float* dst,                     //
                                                   const unsigned char* src,       //
                                                   const int64_t targetImgWidth,   //
                                                   const int64_t targetImgHeight,  //
                                                   const int numChannels) const
{
    for (int c = 0; c < numChannels; ++c) {
        for (int i = 0; i < targetImgHeight; ++i) {
            for (int j = 0; j < targetImgWidth; ++j) {
                dst[c * targetImgHeight * targetImgWidth + i * targetImgWidth + j] =
                    (src[i * targetImgWidth * numChannels + j * numChannels + c] - 127.) / 128.;
            }
        }
    }
}
}  // namespace Ort
