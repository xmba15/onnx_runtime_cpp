/**
 * @file    TinyYolov2.cpp
 *
 * @author  btran
 *
 */

#include "TinyYolov2.hpp"

namespace Ort
{
TinyYolov2::TinyYolov2(const uint16_t numClasses,     //
                       const std::string& modelPath,  //
                       const std::optional<size_t>& gpuIdx,
                       const std::optional<std::vector<std::vector<int64_t>>>& inputShapes)
    : ImageRecognitionOrtSessionHandlerBase(numClasses, modelPath, gpuIdx, inputShapes)
{
}

TinyYolov2::~TinyYolov2()
{
}

std::tuple<std::vector<std::array<float, 4>>, std::vector<float>, std::vector<uint64_t>>
TinyYolov2::postProcess(const std::vector<DataOutputType>& inferenceOutput, const float confidenceThresh) const
{
    std::vector<float> outputData(inferenceOutput.front().first, inferenceOutput.front().first + NUM_BOXES);

    // or
    // std::vector<float> outputData(inferenceOutput.front().first,
    //                               inferenceOutput.front().first +
    //                                   std::accumulate(inferenceOutput.front().second.begin(),
    //                                                   inferenceOutput.front().second.end(), 1, std::multiplies<>()));

    std::vector<std::array<float, 4>> bboxes;
    std::vector<float> scores;
    std::vector<uint64_t> classIndices;

    std::vector<float> tmpScores(m_numClasses);

    for (uint64_t i = 0; i < FEATURE_MAP_SIZE; ++i) {
        for (uint64_t j = 0; j < NUM_ANCHORS; ++j) {
            for (uint64_t k = 0; k < m_numClasses; ++k) {
                tmpScores[k] = outputData[i + FEATURE_MAP_SIZE * ((m_numClasses + 5) * j + k + 5)];
            }
            Ort::softmax(tmpScores.data(), m_numClasses);
            uint64_t maxIdx =
                std::distance(tmpScores.begin(), std::max_element(tmpScores.begin(), tmpScores.begin() + m_numClasses));
            float probability = tmpScores[maxIdx];

            if (Ort::sigmoid(outputData[i + FEATURE_MAP_SIZE * ((m_numClasses + 5) * j + 4)]) * probability >=
                confidenceThresh) {
                float xcenter =
                    (Ort::sigmoid(outputData[i + FEATURE_MAP_SIZE * (m_numClasses + 5 * j)]) + i % 13) * 32.0;
                float ycenter =
                    (Ort::sigmoid(outputData[i + FEATURE_MAP_SIZE * ((m_numClasses + 5) * j + 1)]) + i % 13) * 32.0;

                float width =
                    expf(outputData[i + FEATURE_MAP_SIZE * ((m_numClasses + 5) * j + 2)]) * ANCHORS[2 * j] * 32.0;
                float height =
                    expf(outputData[i + FEATURE_MAP_SIZE * ((m_numClasses + 5) * j + 3)]) * ANCHORS[2 * j + 1] * 32.0;

                float xmin = std::max<float>(xcenter - width / 2, 0.0);
                float ymin = std::max<float>(ycenter - height / 2, 0.0);
                float xmax = std::min<float>(xcenter + width / 2, IMG_WIDTH);
                float ymax = std::min<float>(ycenter + height / 2, IMG_HEIGHT);

                bboxes.emplace_back(std::array<float, 4>{xmin, ymin, xmax, ymax});
                scores.emplace_back(probability);
                classIndices.emplace_back(maxIdx);
            }
        }
    }

    return std::make_tuple(bboxes, scores, classIndices);
}

void TinyYolov2::preprocess(float* dst,                     //
                            const unsigned char* src,       //
                            const int64_t targetImgWidth,   //
                            const int64_t targetImgHeight,  //
                            const int numChannels) const
{
    for (int i = 0; i < targetImgHeight; ++i) {
        for (int j = 0; j < targetImgWidth; ++j) {
            for (int c = 0; c < numChannels; ++c) {
                dst[c * targetImgHeight * targetImgWidth + i * targetImgWidth + j] =
                    src[i * targetImgWidth * numChannels + j * numChannels + c];
            }
        }
    }
}
}  // namespace Ort
