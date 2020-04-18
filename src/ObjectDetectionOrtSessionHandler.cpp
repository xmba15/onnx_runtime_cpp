/**
 * @file    ObjectDetectionOrtSessionHandler.cpp
 *
 * @author  btran
 *
 * Copyright (c) organization
 *
 */

#include "ort_utility/ort_utility.hpp"

namespace Ort
{
ObjectDetectionOrtSessionHandler::ObjectDetectionOrtSessionHandler(
    const uint16_t numClasses,            //
    const std::string& modelPath,         //
    const std::optional<size_t>& gpuIdx,  //
    const std::optional<std::vector<std::vector<int64_t>>>& inputShapes)
    : ImageRecognitionOrtSessionHandlerBase(numClasses, modelPath, gpuIdx, inputShapes)
{
}

ObjectDetectionOrtSessionHandler::~ObjectDetectionOrtSessionHandler()
{
}

}  // namespace Ort
