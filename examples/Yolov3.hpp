/**
 * @file    Yolov3.hpp
 *
 * @author  btran
 *
 * @date    2020-05-31
 *
 * Copyright (c) organization
 *
 */

#pragma once

#include <tuple>
#include <vector>

#include <opencv2/opencv.hpp>

#include <ort_utility/ort_utility.hpp>

namespace Ort
{
class Yolov3 : public ImageRecognitionOrtSessionHandlerBase
{
 public:
    static constexpr int64_t IMG_H = 416;

    static constexpr int64_t IMG_W = 416;

    static constexpr int64_t IMG_CHANNEL = 3;

    Yolov3(const uint16_t numClasses,                           //
           const std::string& modelPath,                        //
           const std::optional<size_t>& gpuIdx = std::nullopt,  //
           const std::optional<std::vector<std::vector<int64_t>>>& inputShapes = std::nullopt);

    ~Yolov3();

    void preprocess(float* dst,                     //
                    const unsigned char* src,       //
                    const int64_t targetImgWidth,   //
                    const int64_t targetImgHeight,  //
                    const int numChannels) const;
};
}  // namespace Ort
