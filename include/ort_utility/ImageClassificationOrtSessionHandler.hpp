/**
 * @file    ImageClassificationOrtSessionHandler.hpp
 *
 * @author  btran
 *
 */

#pragma once

#include <string>
#include <utility>
#include <vector>

#include "ImageRecognitionOrtSessionHandlerBase.hpp"

namespace Ort
{
class ImageClassificationOrtSessionHandler : public ImageRecognitionOrtSessionHandlerBase
{
 public:
    ImageClassificationOrtSessionHandler(
        const uint16_t numClasses,                           //
        const std::string& modelPath,                        //
        const std::optional<size_t>& gpuIdx = std::nullopt,  //
        const std::optional<std::vector<std::vector<int64_t>>>& inputShapes = std::nullopt);

    ~ImageClassificationOrtSessionHandler();

    std::vector<std::pair<int, float>> topK(const std::vector<float*>& inferenceOutput,  //
                                            const uint16_t k = 1,                        //
                                            const bool useSoftmax = true) const;

    std::string topKToString(const std::vector<float*>& inferenceOutput,  //
                             const uint16_t k = 1,                        //
                             const bool useSoftmax = true) const;
};
}  // namespace Ort
