/**
 * @file    SuperPointApp.cpp
 *
 * @author  btran
 *
 */

#include "SuperPoint.hpp"
#include "Utility.hpp"

namespace
{
using KeyPointAndDesc = std::pair<std::vector<cv::KeyPoint>, cv::Mat>;

KeyPointAndDesc processOneFrame(const Ort::SuperPoint& osh, const cv::Mat& inputImg, float* dst, int borderRemove = 4,
                                float confidenceThresh = 0.015, bool alignCorners = true);

}  // namespace

int main(int argc, char* argv[])
{
    if (argc != 4) {
        std::cerr << "Usage: [apps] [path/to/onnx/super/point] [path/to/image1] [path/to/image2]" << std::endl;
        return EXIT_FAILURE;
    }

    const std::string ONNX_MODEL_PATH = argv[1];
    const std::vector<std::string> IMAGE_PATHS = {argv[2], argv[3]};

    Ort::SuperPoint osh(ONNX_MODEL_PATH, 0,
                        std::vector<std::vector<int64_t>>{
                            {1, Ort::SuperPoint::IMG_CHANNEL, Ort::SuperPoint::IMG_H, Ort::SuperPoint::IMG_W}});

    std::vector<cv::Mat> images;
    std::vector<cv::Mat> grays;
    std::transform(IMAGE_PATHS.begin(), IMAGE_PATHS.end(), std::back_inserter(images),
                   [](const auto& imagePath) { return cv::imread(imagePath); });
    for (int i = 0; i < 2; ++i) {
        if (images[i].empty()) {
            throw std::runtime_error("failed to open " + IMAGE_PATHS[i]);
        }
    }
    std::transform(IMAGE_PATHS.begin(), IMAGE_PATHS.end(), std::back_inserter(grays),
                   [](const auto& imagePath) { return cv::imread(imagePath, 0); });

    std::vector<float> dst(Ort::SuperPoint::IMG_CHANNEL * Ort::SuperPoint::IMG_H * Ort::SuperPoint::IMG_W);

    std::vector<KeyPointAndDesc> results;
    std::transform(grays.begin(), grays.end(), std::back_inserter(results),
                   [&osh, &dst](const auto& gray) { return processOneFrame(osh, gray, dst.data()); });

    cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
    std::vector<std::vector<cv::DMatch>> knnMatches;
    const int numMatch = 2;
    matcher->knnMatch(results[0].second, results[1].second, knnMatches, numMatch);

    std::vector<cv::DMatch> goodMatches;
    const float loweRatioThresh = 0.8;
    for (const auto& match : knnMatches) {
        if (match[0].distance < loweRatioThresh * match[1].distance) {
            goodMatches.emplace_back(match[0]);
        }
    }

    cv::Mat matchesImage;
    cv::drawMatches(images[0], results[0].first, images[1], results[1].first, goodMatches, matchesImage,
                    cv::Scalar::all(-1), cv::Scalar::all(-1), std::vector<char>(),
                    cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    cv::imwrite("good_matches.jpg", matchesImage);

    return EXIT_SUCCESS;
}

namespace
{
/**
 *  @brief detect super point
 *
 *  @return vector of detected key points
 */
std::vector<cv::KeyPoint> getKeyPoints(const std::vector<Ort::OrtSessionHandler::DataOutputType>& inferenceOutput,
                                       int borderRemove, float confidenceThresh)
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

/**
 *  @brief estimate super point's keypoint descriptor
 *
 *  @return keypoint Mat of shape [num key point x 256]
 */
cv::Mat getDescriptors(const cv::Mat& coarseDescriptors, const std::vector<cv::KeyPoint>& keyPoints, int height,
                       int width, bool alignCorners)
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

KeyPointAndDesc processOneFrame(const Ort::SuperPoint& osh, const cv::Mat& inputImg, float* dst, int borderRemove,
                                float confidenceThresh, bool alignCorners)
{
    int origW = inputImg.cols, origH = inputImg.rows;
    std::vector<float> originImageSize{static_cast<float>(origH), static_cast<float>(origW)};
    cv::Mat scaledImg;
    cv::resize(inputImg, scaledImg, cv::Size(Ort::SuperPoint::IMG_W, Ort::SuperPoint::IMG_H), 0, 0, cv::INTER_CUBIC);
    osh.preprocess(dst, scaledImg.data, Ort::SuperPoint::IMG_W, Ort::SuperPoint::IMG_H, Ort::SuperPoint::IMG_CHANNEL);
    auto inferenceOutput = osh({dst});

    std::vector<cv::KeyPoint> keyPoints = getKeyPoints(inferenceOutput, borderRemove, confidenceThresh);

    std::vector<int> descriptorShape(inferenceOutput[1].second.begin(), inferenceOutput[1].second.end());
    cv::Mat coarseDescriptorMat(descriptorShape.size(), descriptorShape.data(), CV_32F,
                                inferenceOutput[1].first);  // 1 x 256 x H/8 x W/8

    cv::Mat descriptors =
        getDescriptors(coarseDescriptorMat, keyPoints, Ort::SuperPoint::IMG_H, Ort::SuperPoint::IMG_W, alignCorners);

    for (auto& keyPoint : keyPoints) {
        keyPoint.pt.x *= static_cast<float>(origW) / Ort::SuperPoint::IMG_W;
        keyPoint.pt.y *= static_cast<float>(origH) / Ort::SuperPoint::IMG_H;
    }

    return {keyPoints, descriptors};
}
}  // namespace
