/**
 * @file    YoloX.hpp
 *
 * @author  btran
 *
 */

#pragma once

#include <tuple>
#include <vector>

#include <opencv2/opencv.hpp>

#include <ort_utility/ort_utility.hpp>

namespace Ort
{
class YoloX : public ImageRecognitionOrtSessionHandlerBase
{
 public:
    static constexpr int64_t IMG_H = 640;
    static constexpr int64_t IMG_W = 640;
    static constexpr int64_t IMG_CHANNEL = 3;

    YoloX(const uint16_t numClasses,                           //
          const std::string& modelPath,                        //
          const std::optional<size_t>& gpuIdx = std::nullopt,  //
          const std::optional<std::vector<std::vector<int64_t>>>& inputShapes = std::nullopt);

    ~YoloX();

    void preprocess(float* dst,                     //
                    const unsigned char* src,       //
                    const int64_t targetImgWidth,   //
                    const int64_t targetImgHeight,  //
                    const int numChannels) const;
};
}  // namespace Ort
