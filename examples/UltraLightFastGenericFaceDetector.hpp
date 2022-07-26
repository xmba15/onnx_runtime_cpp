/**
 * @file    UltraLightFastGenericFaceDetector.hpp
 *
 * @author  btran
 *
 */

#pragma once

#include <optional>
#include <string>
#include <vector>

#include <ort_utility/ort_utility.hpp>

namespace Ort
{
class UltraLightFastGenericFaceDetector : public ImageRecognitionOrtSessionHandlerBase
{
 public:
    static constexpr int64_t IMG_H = 480;

    static constexpr int64_t IMG_W = 640;

    static constexpr int64_t IMG_CHANNEL = 3;

    explicit UltraLightFastGenericFaceDetector(
        const std::string& modelPath, const std::optional<size_t>& gpuIdx = std::nullopt,
        const std::optional<std::vector<std::vector<std::int64_t>>>& inputShapes = std::nullopt);

    ~UltraLightFastGenericFaceDetector();

    void preprocess(float* dst,                     //
                    const unsigned char* src,       //
                    const int64_t targetImgWidth,   //
                    const int64_t targetImgHeight,  //
                    const int numChannels) const;
};
}  // namespace Ort
