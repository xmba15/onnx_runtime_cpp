/**
 * @file    LoFTR.cpp
 *
 * @author  btran
 *
 */

#include "LoFTR.hpp"

namespace Ort
{
void LoFTR::preprocess(float* dst, const unsigned char* src, const int64_t targetImgWidth,
                       const int64_t targetImgHeight, const int numChannels) const
{
    for (int i = 0; i < targetImgHeight; ++i) {
        for (int j = 0; j < targetImgWidth; ++j) {
            for (int c = 0; c < numChannels; ++c) {
                dst[c * targetImgHeight * targetImgWidth + i * targetImgWidth + j] =
                    (src[i * targetImgWidth * numChannels + j * numChannels + c] / 255.0);
            }
        }
    }
}
}  // namespace Ort
