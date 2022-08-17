/**
 * @file    SuperPoint.cpp
 *
 * @author  btran
 *
 */

#include "SuperPoint.hpp"
#include "Utility.hpp"

namespace Ort
{
void SuperPoint::preprocess(float* dst, const unsigned char* src, const int64_t targetImgWidth,
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

std::vector<int> SuperPoint::nmsFast(const std::vector<cv::KeyPoint>& keyPoints, int height, int width,
                                     int distThresh) const
{
    static const int TO_PROCESS = 1;
    static const int EMPTY_OR_SUPPRESSED = 0;

    std::vector<int> sortedIndices(keyPoints.size());
    std::iota(sortedIndices.begin(), sortedIndices.end(), 0);

    // sort in descending order base on confidence
    std::stable_sort(sortedIndices.begin(), sortedIndices.end(),
                     [&keyPoints](int lidx, int ridx) { return keyPoints[lidx].response > keyPoints[ridx].response; });

    cv::Mat grid = cv::Mat(height, width, CV_8U, TO_PROCESS);
    std::vector<int> keepIndices;

    for (int idx : sortedIndices) {
        int x = keyPoints[idx].pt.x;
        int y = keyPoints[idx].pt.y;

        if (grid.at<uchar>(y, x) == TO_PROCESS) {
            for (int i = y - distThresh; i < y + distThresh; ++i) {
                if (i < 0 || i >= height) {
                    continue;
                }

                for (int j = x - distThresh; j < x + distThresh; ++j) {
                    if (j < 0 || j >= width) {
                        continue;
                    }
                    grid.at<uchar>(i, j) = EMPTY_OR_SUPPRESSED;
                }
            }
            keepIndices.emplace_back(idx);
        }
    }

    return keepIndices;
}

std::vector<cv::KeyPoint>
SuperPoint::getKeyPoints(const std::vector<Ort::OrtSessionHandler::DataOutputType>& inferenceOutput, int borderRemove,
                         float confidenceThresh) const
{
    std::vector<int> detectorShape(inferenceOutput[0].second.begin() + 1, inferenceOutput[0].second.end());

    cv::Mat detectorMat(detectorShape.size(), detectorShape.data(), CV_32F,
                        inferenceOutput[0].first);  // 65 x H/8 x W/8
    cv::Mat buffer;

    transposeNDWrapper(detectorMat, {1, 2, 0}, buffer);
    buffer.copyTo(detectorMat);  // H/8 x W/8 x 65

    for (int i = 0; i < detectorShape[1]; ++i) {
        for (int j = 0; j < detectorShape[2]; ++j) {
            Ort::softmax(detectorMat.ptr<float>(i, j), detectorShape[0]);
        }
    }
    detectorMat = detectorMat({cv::Range::all(), cv::Range::all(), cv::Range(0, detectorShape[0] - 1)})
                      .clone();                                                        // H/8 x W/8 x 64
    detectorMat = detectorMat.reshape(1, {detectorShape[1], detectorShape[2], 8, 8});  // H/8 x W/8 x 8 x 8
    transposeNDWrapper(detectorMat, {0, 2, 1, 3}, buffer);
    buffer.copyTo(detectorMat);  // H/8 x 8 x W/8 x 8

    detectorMat = detectorMat.reshape(1, {detectorShape[1] * 8, detectorShape[2] * 8});  // H x W

    std::vector<cv::KeyPoint> keyPoints;
    for (int i = borderRemove; i < detectorMat.rows - borderRemove; ++i) {
        auto rowPtr = detectorMat.ptr<float>(i);
        for (int j = borderRemove; j < detectorMat.cols - borderRemove; ++j) {
            if (rowPtr[j] > confidenceThresh) {
                cv::KeyPoint keyPoint;
                keyPoint.pt.x = j;
                keyPoint.pt.y = i;
                keyPoint.response = rowPtr[j];
                keyPoints.emplace_back(keyPoint);
            }
        }
    }

    return keyPoints;
}

cv::Mat SuperPoint::getDescriptors(const cv::Mat& coarseDescriptors, const std::vector<cv::KeyPoint>& keyPoints,
                                   int height, int width, bool alignCorners) const
{
    cv::Mat keyPointMat(keyPoints.size(), 2, CV_32F);

    for (int i = 0; i < keyPoints.size(); ++i) {
        auto rowPtr = keyPointMat.ptr<float>(i);
        rowPtr[0] = 2 * keyPoints[i].pt.y / (height - 1) - 1;
        rowPtr[1] = 2 * keyPoints[i].pt.x / (width - 1) - 1;
    }
    keyPointMat = keyPointMat.reshape(1, {1, 1, static_cast<int>(keyPoints.size()), 2});
    cv::Mat descriptors = bilinearGridSample(coarseDescriptors, keyPointMat, alignCorners);
    descriptors = descriptors.reshape(1, {coarseDescriptors.size[1], static_cast<int>(keyPoints.size())});

    cv::Mat buffer;
    transposeNDWrapper(descriptors, {1, 0}, buffer);

    return buffer;
}
}  // namespace Ort
