/**
 * @file    OrtSessionHandler.hpp
 *
 * @author  btran
 *
 * @date    2020-04-19
 *
 * Copyright (c) organization
 *
 */

#pragma once

#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace Ort
{
class OrtSessionHandler
{
 public:
    // DataOutputType->(pointer to output data, shape of output data)
    using DataOutputType = std::pair<float*, std::vector<int64_t>>;

    OrtSessionHandler(const std::string& modelPath,  //
                      const std::optional<size_t>& gpuIdx = std::nullopt,
                      const std::optional<std::vector<std::vector<int64_t>>>& inputShapes = std::nullopt);
    ~OrtSessionHandler();

    // multiple inputs, multiple outputs
    std::vector<DataOutputType> operator()(const std::vector<float*>& inputImgData);

 private:
    class OrtSessionHandlerIml;
    std::unique_ptr<OrtSessionHandlerIml> m_piml;
};
}  // namespace Ort
