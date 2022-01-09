/**
 * @file    YoloX.hpp
 *
 * @author  btran
 *
 */

#pragma once

#include <tuple>
#include <vector>

#include <opencv2/opencv.hpp>

#include <ort_utility/ort_utility.hpp>

namespace Ort
{
class YoloX : public ImageRecognitionOrtSessionHandlerBase
{
 public:
    static constexpr int64_t IMG_H = 640;
    static constexpr int64_t IMG_W = 640;
    static constexpr int64_t IMG_CHANNEL = 3;

    struct Object {
        cv::Rect_<float> pos;
        int label;
        float prob;
    };

    YoloX(const uint16_t numClasses,                           //
          const std::string& modelPath,                        //
          const std::optional<size_t>& gpuIdx = std::nullopt,  //
          const std::optional<std::vector<std::vector<int64_t>>>& inputShapes = std::nullopt);

    ~YoloX();

    void preprocess(float* dst,                     //
                    const unsigned char* src,       //
                    const int64_t targetImgWidth,   //
                    const int64_t targetImgHeight,  //
                    const int numChannels) const;

    void decodeOutputs(const float* prob, std::vector<Object>& objects, float confThresh) const;

    /**
     *  @brief update strides
     *  this method gives users the flexibility to use other yolox models
     */
    void updateStrides(const std::vector<int>& strides)
    {
        if (strides.empty()) {
            throw std::runtime_error("cannot update empty strides");
        }
        m_strides = strides;
    }

 private:
    std::vector<int> m_strides = {8, 16, 32};

    struct GridAndStride {
        int grid0;
        int grid1;
        int stride;
    };

    void genereateGridsAndStrides(int targetSize, const std::vector<int>& strides,
                                  std::vector<GridAndStride>& gridStrides) const;

    std::vector<Object> generateYoloXProposals(const float* prob, const std::vector<GridAndStride>& gridStrides,
                                               float confThresh) const;
};
}  // namespace Ort
