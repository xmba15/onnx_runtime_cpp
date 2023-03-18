/**
 * @file    LoFTR.hpp
 *
 * @author  btran
 *
 */

#pragma once

#include <ort_utility/ort_utility.hpp>

namespace Ort
{
class LoFTR : public OrtSessionHandler
{
 public:
    static constexpr int64_t IMG_H = 480;
    static constexpr int64_t IMG_W = 640;
    static constexpr int64_t IMG_CHANNEL = 1;

    using OrtSessionHandler::OrtSessionHandler;

    void preprocess(float* dst,                     //
                    const unsigned char* src,       //
                    const int64_t targetImgWidth,   //
                    const int64_t targetImgHeight,  //
                    const int numChannels) const;
};
}  // namespace Ort
