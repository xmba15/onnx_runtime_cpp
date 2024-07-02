/**
 * @file    YoloX.cpp
 *
 * @author  btran
 *
 */

#include "YoloX.hpp"

namespace Ort
{
struct YoloX::GridAndStride {
    int grid0;
    int grid1;
    int stride;
};

YoloX::YoloX(const uint16_t numClasses,     //
             const std::string& modelPath,  //
             const std::optional<size_t>& gpuIdx, const std::optional<std::vector<std::vector<int64_t>>>& inputShapes)
    : ImageRecognitionOrtSessionHandlerBase(numClasses, modelPath, gpuIdx, inputShapes)
{
}

YoloX::~YoloX()
{
}

void YoloX::preprocess(float* dst,                     //
                       const unsigned char* src,       //
                       const int64_t targetImgWidth,   //
                       const int64_t targetImgHeight,  //
                       const int numChannels) const
{
    for (int c = 0; c < numChannels; ++c) {
        for (int i = 0; i < targetImgHeight; ++i) {
            for (int j = 0; j < targetImgWidth; ++j) {
                dst[c * targetImgHeight * targetImgWidth + i * targetImgWidth + j] =
                    src[i * targetImgWidth * numChannels + j * numChannels + c];
            }
        }
    }
}

std::vector<YoloX::Object> YoloX::decodeOutputs(const float* prob, float confThresh) const
{
    std::vector<GridAndStride> gridStrides = this->generateGridsAndStrides(IMG_W, m_strides);
    return this->generateYoloXProposals(prob, gridStrides, confThresh);
}

/**
 *  @brief https://github.com/Megvii-BaseDetection/YOLOX/blob/main/demo/MegEngine/cpp/yolox.cpp#L64
 */
std::vector<YoloX::GridAndStride> YoloX::generateGridsAndStrides(int targetSize, const std::vector<int>& strides) const
{
    std::vector<YoloX::GridAndStride> gridStrides;
    for (auto stride : strides) {
        int numGrid = targetSize / stride;
        for (int g1 = 0; g1 < numGrid; g1++) {
            for (int g0 = 0; g0 < numGrid; g0++) {
                gridStrides.push_back((GridAndStride){g0, g1, stride});
            }
        }
    }
    return gridStrides;
}

/**
 *  @brief https://github.com/Megvii-BaseDetection/YOLOX/blob/main/demo/MegEngine/cpp/yolox.cpp#L76
 */
std::vector<YoloX::Object>
YoloX::generateYoloXProposals(const float* prob, const std::vector<GridAndStride>& gridStrides, float confThresh) const
{
    std::vector<Object> objects;

    const int numAnchors = gridStrides.size();
    const int numClasses = m_numClasses;
    for (int anchorIdx = 0; anchorIdx < numAnchors; ++anchorIdx) {
        const int grid0 = gridStrides[anchorIdx].grid0;
        const int grid1 = gridStrides[anchorIdx].grid1;
        const int stride = gridStrides[anchorIdx].stride;

        // 4 parameters defining the bounding boxes and 1 parameter defining the confidence
        const int basicPos = anchorIdx * (numClasses + 5);
        float xCenter = (prob[basicPos + 0] + grid0) * stride;
        float yCenter = (prob[basicPos + 1] + grid1) * stride;
        float w = exp(prob[basicPos + 2]) * stride;
        float h = exp(prob[basicPos + 3]) * stride;
        float x0 = xCenter - w * 0.5f;
        float y0 = yCenter - h * 0.5f;

        float boxObjectness = prob[basicPos + 4];
        for (int classIdx = 0; classIdx < numClasses; ++classIdx) {
            float boxClsScore = prob[basicPos + 5 + classIdx];
            float boxProb = boxObjectness * boxClsScore;
            if (boxProb > confThresh) {
                Object obj;
                obj.pos.x = x0;
                obj.pos.y = y0;
                obj.pos.width = w;
                obj.pos.height = h;
                obj.label = classIdx;
                obj.prob = boxProb;
                objects.push_back(obj);
            }
        }
    }

    return objects;
}
}  // namespace Ort
