/**
 * @file    MaskRCNN.hpp
 *
 * @author  btran
 *
 * @date    2020-05-18
 *
 * Copyright (c) organization
 *
 */

#pragma once

#include <optional>
#include <string>
#include <vector>

#include <ort_utility/ort_utility.hpp>

namespace Ort
{
class MaskRCNN : public ImageRecognitionOrtSessionHandlerBase
{
 public:
    static constexpr int64_t IMG_WIDTH = 800;

    static constexpr int64_t IMG_HEIGHT = 800;

    static constexpr int64_t MIN_IMAGE_SIZE = 800;

    static constexpr int64_t IMG_CHANNEL = 3;

    MaskRCNN(const uint16_t numClasses,                           //
             const std::string& modelPath,                        //
             const std::optional<size_t>& gpuIdx = std::nullopt,  //
             const std::optional<std::vector<std::vector<int64_t>>>& inputShapes = std::nullopt);

    ~MaskRCNN();

    void preprocess(float* dst,                     //
                    const unsigned char* src,       //
                    const int64_t targetImgWidth,   //
                    const int64_t targetImgHeight,  //
                    const int numChannels,          //
                    const int64_t offsetPadW = 0,   //
                    const int64_t offsetPadH = 0,   //
                    const std::vector<float>& meanVal = {}) const;
};
}  // namespace Ort
