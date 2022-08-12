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
                                float confidenceThresh = 0.015, bool alignCorners = true, int distThresh = 2);

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
    cv::imwrite("super_point_good_matches.jpg", matchesImage);
    cv::imshow("super_point_good_matches", matchesImage);
    cv::waitKey();

    return EXIT_SUCCESS;
}

namespace
{
KeyPointAndDesc processOneFrame(const Ort::SuperPoint& osh, const cv::Mat& inputImg, float* dst, int borderRemove,
                                float confidenceThresh, bool alignCorners, int distThresh)
{
    int origW = inputImg.cols, origH = inputImg.rows;
    cv::Mat scaledImg;
    cv::resize(inputImg, scaledImg, cv::Size(Ort::SuperPoint::IMG_W, Ort::SuperPoint::IMG_H), 0, 0, cv::INTER_CUBIC);
    osh.preprocess(dst, scaledImg.data, Ort::SuperPoint::IMG_W, Ort::SuperPoint::IMG_H, Ort::SuperPoint::IMG_CHANNEL);
    auto inferenceOutput = osh({dst});

    std::vector<cv::KeyPoint> keyPoints = osh.getKeyPoints(inferenceOutput, borderRemove, confidenceThresh);

    std::vector<int> descriptorShape(inferenceOutput[1].second.begin(), inferenceOutput[1].second.end());
    cv::Mat coarseDescriptorMat(descriptorShape.size(), descriptorShape.data(), CV_32F,
                                inferenceOutput[1].first);  // 1 x 256 x H/8 x W/8

    std::vector<int> keepIndices = osh.nmsFast(keyPoints, Ort::SuperPoint::IMG_H, Ort::SuperPoint::IMG_W, distThresh);

    std::vector<cv::KeyPoint> keepKeyPoints;
    keepKeyPoints.reserve(keepIndices.size());
    std::transform(keepIndices.begin(), keepIndices.end(), std::back_inserter(keepKeyPoints),
                   [&keyPoints](int idx) { return keyPoints[idx]; });
    keyPoints = std::move(keepKeyPoints);

    cv::Mat descriptors = osh.getDescriptors(coarseDescriptorMat, keyPoints, Ort::SuperPoint::IMG_H,
                                             Ort::SuperPoint::IMG_W, alignCorners);

    for (auto& keyPoint : keyPoints) {
        keyPoint.pt.x *= static_cast<float>(origW) / Ort::SuperPoint::IMG_W;
        keyPoint.pt.y *= static_cast<float>(origH) / Ort::SuperPoint::IMG_H;
    }

    return {keyPoints, descriptors};
}
}  // namespace
