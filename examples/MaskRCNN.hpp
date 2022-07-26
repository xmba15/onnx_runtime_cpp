/**
 * @file    MaskRCNN.hpp
 *
 * @author  btran
 *
 */

#pragma once

#include <optional>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include <ort_utility/ort_utility.hpp>

namespace Ort
{
class MaskRCNN : public ImageRecognitionOrtSessionHandlerBase
{
 public:
    static constexpr int64_t MIN_IMAGE_SIZE = 800;

    static constexpr int64_t IMG_CHANNEL = 3;

    MaskRCNN(const uint16_t numClasses,                           //
             const std::string& modelPath,                        //
             const std::optional<size_t>& gpuIdx = std::nullopt,  //
             const std::optional<std::vector<std::vector<int64_t>>>& inputShapes = std::nullopt);

    ~MaskRCNN();

    void preprocess(float* dst,                     //
                    const float* src,               //
                    const int64_t targetImgWidth,   //
                    const int64_t targetImgHeight,  //
                    const int numChannels) const;

    void preprocess(float* dst,                     //
                    const cv::Mat& imgSrc,          //
                    const int64_t targetImgWidth,   //
                    const int64_t targetImgHeight,  //
                    const int numChannels) const;
};
}  // namespace Ort
