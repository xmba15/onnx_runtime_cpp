/**
 * @file    ObjectDetectionOrtSessionHandler.hpp
 *
 * @author  btran
 *
 */

#pragma once

#include <string>
#include <vector>

#include "ImageRecognitionOrtSessionHandlerBase.hpp"

namespace Ort
{
class ObjectDetectionOrtSessionHandler : public ImageRecognitionOrtSessionHandlerBase
{
 public:
    ObjectDetectionOrtSessionHandler(
        const uint16_t numClasses,                           //
        const std::string& modelPath,                        //
        const std::optional<size_t>& gpuIdx = std::nullopt,  //
        const std::optional<std::vector<std::vector<int64_t>>>& inputShapes = std::nullopt);

    ~ObjectDetectionOrtSessionHandler();
};
}  // namespace Ort
