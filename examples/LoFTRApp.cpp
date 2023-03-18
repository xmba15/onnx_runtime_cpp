/**
 * @file    LoFTRApp.cpp
 *
 * @author  btran
 *
 */

#include "LoFTR.hpp"
#include "Utility.hpp"
#include <utility>

static constexpr float CONFIDENCE_THRESHOLD = 0.1;

namespace
{
std::pair<std::vector<cv::KeyPoint>, std::vector<cv::KeyPoint>>
processOneImagePair(const Ort::LoFTR& loftrOsh, const cv::Mat& queryImg, const cv::Mat& refImg, float* queryData,
                    float* refData, float confidenceThresh = CONFIDENCE_THRESHOLD);
}  // namespace

int main(int argc, char* argv[])
{
    if (argc != 4) {
        std::cerr << "Usage: [apps] [path/to/onnx/loftr] [path/to/image1] [path/to/image2]" << std::endl;
        return EXIT_FAILURE;
    }

    const std::string ONNX_MODEL_PATH = argv[1];
    const std::vector<std::string> IMAGE_PATHS = {argv[2], argv[3]};

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

    std::vector<float> queryData(Ort::LoFTR::IMG_CHANNEL * Ort::LoFTR::IMG_H * Ort::LoFTR::IMG_W);
    std::vector<float> refData(Ort::LoFTR::IMG_CHANNEL * Ort::LoFTR::IMG_H * Ort::LoFTR::IMG_W);

    Ort::LoFTR osh(
        ONNX_MODEL_PATH, 0,
        std::vector<std::vector<int64_t>>{{1, Ort::LoFTR::IMG_CHANNEL, Ort::LoFTR::IMG_H, Ort::LoFTR::IMG_W},
                                          {1, Ort::LoFTR::IMG_CHANNEL, Ort::LoFTR::IMG_H, Ort::LoFTR::IMG_W}});

    auto matchedKpts = processOneImagePair(osh, grays[0], grays[1], queryData.data(), refData.data());
    const std::vector<cv::KeyPoint>& queryKpts = matchedKpts.first;
    const std::vector<cv::KeyPoint>& refKpts = matchedKpts.second;
    std::vector<cv::DMatch> matches;
    for (int i = 0; i < queryKpts.size(); ++i) {
        cv::DMatch match;
        match.imgIdx = 0;
        match.queryIdx = i;
        match.trainIdx = i;
        matches.emplace_back(std::move(match));
    }
    cv::Mat matchesImage;
    cv::drawMatches(images[0], queryKpts, images[1], refKpts, matches, matchesImage, cv::Scalar::all(-1),
                    cv::Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    cv::imwrite("loftr.jpg", matchesImage);
    cv::imshow("loftr", matchesImage);
    cv::waitKey();

    return EXIT_SUCCESS;
}

namespace
{
std::pair<std::vector<cv::KeyPoint>, std::vector<cv::KeyPoint>>
processOneImagePair(const Ort::LoFTR& loftrOsh, const cv::Mat& queryImg, const cv::Mat& refImg, float* queryData,
                    float* refData, float confidenceThresh)
{
    int origQueryW = queryImg.cols, origQueryH = queryImg.rows;
    int origRefW = refImg.cols, origRefH = refImg.rows;

    cv::Mat scaledQueryImg, scaledRefImg;
    cv::resize(queryImg, scaledQueryImg, cv::Size(Ort::LoFTR::IMG_W, Ort::LoFTR::IMG_H), 0, 0, cv::INTER_CUBIC);
    cv::resize(refImg, scaledRefImg, cv::Size(Ort::LoFTR::IMG_W, Ort::LoFTR::IMG_H), 0, 0, cv::INTER_CUBIC);

    loftrOsh.preprocess(queryData, scaledQueryImg.data, Ort::LoFTR::IMG_W, Ort::LoFTR::IMG_H, Ort::LoFTR::IMG_CHANNEL);
    loftrOsh.preprocess(refData, scaledRefImg.data, Ort::LoFTR::IMG_W, Ort::LoFTR::IMG_H, Ort::LoFTR::IMG_CHANNEL);
    auto inferenceOutput = loftrOsh({queryData, refData});

    // inferenceOutput[0].second: keypoints0 of shape [num kpt x 2]
    // inferenceOutput[1].second: keypoints1 of shape [num kpt x 2]
    // inferenceOutput[2].second: confidences of shape [num kpt]

    int numKeyPoints = inferenceOutput[2].second[0];
    std::vector<cv::KeyPoint> queryKpts, refKpts;
    queryKpts.reserve(numKeyPoints);
    refKpts.reserve(numKeyPoints);

    for (int i = 0; i < numKeyPoints; ++i) {
        float confidence = inferenceOutput[2].first[i];
        if (confidence < confidenceThresh) {
            continue;
        }
        float queryX = inferenceOutput[0].first[i * 2 + 0];
        float queryY = inferenceOutput[0].first[i * 2 + 1];
        float refX = inferenceOutput[1].first[i * 2 + 0];
        float refY = inferenceOutput[1].first[i * 2 + 1];
        cv::KeyPoint queryKpt, refKpt;
        queryKpt.pt.x = queryX * origQueryW / Ort::LoFTR::IMG_W;
        queryKpt.pt.y = queryY * origQueryH / Ort::LoFTR::IMG_H;

        refKpt.pt.x = refX * origRefW / Ort::LoFTR::IMG_W;
        refKpt.pt.y = refY * origRefH / Ort::LoFTR::IMG_H;

        queryKpts.emplace_back(std::move(queryKpt));
        refKpts.emplace_back(std::move(refKpt));
    }

    return std::make_pair(queryKpts, refKpts);
}
}  // namespace
