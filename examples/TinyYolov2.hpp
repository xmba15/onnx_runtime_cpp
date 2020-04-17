/**
 * @file    TinyYolov2.hpp
 *
 * @author  btran
 *
 * @date    2020-05-05
 *
 * Copyright (c) organization
 *
 */

#pragma once

#include <algorithm>
#include <cstring>
#include <string>
#include <tuple>
#include <vector>

#include <ort_utility/ort_utility.hpp>

namespace Ort
{
class TinyYolov2 : public ImageRecognitionOrtSessionHandlerBase
{
 public:
    static constexpr int64_t IMG_WIDTH = 416;

    static constexpr int64_t IMG_HEIGHT = 416;

    static constexpr int64_t IMG_CHANNEL = 3;

    static constexpr int64_t FEATURE_MAP_SIZE = 13 * 13;

    static constexpr int64_t NUM_BOXES = 1 * 13 * 13 * 125;

    static constexpr int64_t NUM_ANCHORS = 5;

    static constexpr float ANCHORS[10] = {1.08, 1.19, 3.42, 4.41, 6.63, 11.38, 9.42, 5.11, 16.62, 10.52};

    TinyYolov2(const uint16_t numClasses,                           //
               const std::string& modelPath,                        //
               const std::optional<size_t>& gpuIdx = std::nullopt,  //
               const std::optional<std::vector<std::vector<int64_t>>>& inputShapes = std::nullopt);

    ~TinyYolov2();

    void preprocess(float* dst,                     //
                    const unsigned char* src,       //
                    const int64_t targetImgWidth,   //
                    const int64_t targetImgHeight,  //
                    const int numChannels) const;

    std::tuple<std::vector<std::array<float, 4>>, std::vector<float>, std::vector<uint64_t>>
    postProcess(const std::vector<DataOutputType>& inferenceOutput, const float confidenceThresh = 0.5) const;
};
}  // namespace Ort
