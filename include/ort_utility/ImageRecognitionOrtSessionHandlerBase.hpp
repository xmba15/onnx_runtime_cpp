/**
 * @file    ImageRecognitionOrtSessionHandlerBase.hpp
 *
 * @author  btran
 *
 */

#pragma once

#include <string>
#include <utility>
#include <vector>

#include "OrtSessionHandler.hpp"

namespace Ort
{
class ImageRecognitionOrtSessionHandlerBase : public OrtSessionHandler
{
 public:
    ImageRecognitionOrtSessionHandlerBase(
        const uint16_t numClasses,                           //
        const std::string& modelPath,                        //
        const std::optional<size_t>& gpuIdx = std::nullopt,  //
        const std::optional<std::vector<std::vector<int64_t>>>& inputShapes = std::nullopt);

    ~ImageRecognitionOrtSessionHandlerBase();

    void initClassNames(const std::vector<std::string>& classNames);

    virtual void preprocess(float* dst,                              //
                            const unsigned char* src,                //
                            const int64_t targetImgWidth,            //
                            const int64_t targetImgHeight,           //
                            const int numChanels,                    //
                            const std::vector<float>& meanVal = {},  //
                            const std::vector<float>& stdVal = {}) const;

    uint16_t numClasses() const
    {
        return m_numClasses;
    }

    const std::vector<std::string>& classNames() const
    {
        return m_classNames;
    }

 protected:
    const uint16_t m_numClasses;
    std::vector<std::string> m_classNames;
};
}  // namespace Ort
