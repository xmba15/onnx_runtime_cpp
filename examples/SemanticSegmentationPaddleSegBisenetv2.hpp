/**
 * @file    SemanticSegmentationPaddleSegBisenetv2.hpp
 *
 * @author  btran
 *
 */

#pragma once

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

#include <ort_utility/ort_utility.hpp>

namespace Ort
{
class SemanticSegmentationPaddleSegBisenetv2 : public ImageRecognitionOrtSessionHandlerBase
{
 public:
    static constexpr int64_t IMG_H = 1024;
    static constexpr int64_t IMG_W = 1024;
    static constexpr int64_t IMG_CHANNEL = 3;

    SemanticSegmentationPaddleSegBisenetv2(
        const uint16_t numClasses,                           //
        const std::string& modelPath,                        //
        const std::optional<size_t>& gpuIdx = std::nullopt,  //
        const std::optional<std::vector<std::vector<int64_t>>>& inputShapes = std::nullopt);

    void preprocess(float* dst,                                           //
                    const unsigned char* src,                             //
                    int64_t targetImgWidth,                               //
                    int64_t targetImgHeight,                              //
                    int numChannels,                                      //
                    const std::vector<float>& meanVal = {0.5, 0.5, 0.5},  //
                    const std::vector<float>& stdVal = {0.5, 0.5, 0.5}) const;
};
};  // namespace Ort
