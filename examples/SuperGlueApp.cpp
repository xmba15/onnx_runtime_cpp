/**
 * @file    SuperGlueApp.cpp
 *
 * @author  btran
 *
 */

#include "SuperPoint.hpp"
#include "Utility.hpp"

namespace
{
using KeyPointAndDesc = std::pair<std::vector<cv::KeyPoint>, cv::Mat>;

KeyPointAndDesc processOneFrameSuperPoint(const Ort::SuperPoint& superPointOsh, const cv::Mat& inputImg, float* dst,
                                          int borderRemove = 4, float confidenceThresh = 0.015,
                                          bool alignCorners = true, int distThresh = 2);

void normalizeDescriptors(cv::Mat* descriptors);
}  // namespace

int main(int argc, char* argv[])
{
    if (argc != 5) {
        std::cerr
            << "Usage: [apps] [path/to/onnx/super/point] [path/to/onnx/super/glue] [path/to/image1] [path/to/image2]"
            << std::endl;
        return EXIT_FAILURE;
    }

    const std::string ONNX_MODEL_PATH = argv[1];
    const std::string SUPERGLUE_ONNX_MODEL_PATH = argv[2];
    const std::vector<std::string> IMAGE_PATHS = {argv[3], argv[4]};

    Ort::SuperPoint superPointOsh(ONNX_MODEL_PATH, 0,
                                  std::vector<std::vector<int64_t>>{{1, Ort::SuperPoint::IMG_CHANNEL,
                                                                     Ort::SuperPoint::IMG_H, Ort::SuperPoint::IMG_W}});

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

    std::vector<KeyPointAndDesc> superPointResults;
    std::transform(grays.begin(), grays.end(), std::back_inserter(superPointResults),
                   [&superPointOsh, &dst](const auto& gray) {
                       return processOneFrameSuperPoint(superPointOsh, gray, dst.data());
                   });

    for (auto& curKeyPointAndDesc : superPointResults) {
        normalizeDescriptors(&curKeyPointAndDesc.second);
    }

    // superglue
    static const int DUMMY_NUM_KEYPOINTS = 256;
    Ort::OrtSessionHandler superGlueOsh(SUPERGLUE_ONNX_MODEL_PATH, 0,
                                        std::vector<std::vector<int64_t>>{
                                            {4},
                                            {1, DUMMY_NUM_KEYPOINTS},
                                            {1, DUMMY_NUM_KEYPOINTS, 2},
                                            {1, 256, DUMMY_NUM_KEYPOINTS},
                                            {4},
                                            {1, DUMMY_NUM_KEYPOINTS},
                                            {1, DUMMY_NUM_KEYPOINTS, 2},
                                            {1, 256, DUMMY_NUM_KEYPOINTS},
                                        });

    int numKeypoints0 = superPointResults[0].first.size();
    int numKeypoints1 = superPointResults[1].first.size();
    std::vector<std::vector<int64_t>> inputShapes = {
        {4}, {1, numKeypoints0}, {1, numKeypoints0, 2}, {1, 256, numKeypoints0},
        {4}, {1, numKeypoints1}, {1, numKeypoints1, 2}, {1, 256, numKeypoints1},
    };
    superGlueOsh.updateInputShapes(inputShapes);

    std::vector<std::vector<float>> imageShapes(2);
    std::vector<std::vector<float>> scores(2);
    std::vector<std::vector<float>> keypoints(2);
    std::vector<std::vector<float>> descriptors(2);

    cv::Mat buffer;
    for (int i = 0; i < 2; ++i) {
        imageShapes[i] = {1, 1, static_cast<float>(images[0].rows), static_cast<float>(images[0].cols)};
        std::transform(superPointResults[i].first.begin(), superPointResults[i].first.end(),
                       std::back_inserter(scores[i]), [](const cv::KeyPoint& keypoint) { return keypoint.response; });
        for (const auto& k : superPointResults[i].first) {
            keypoints[i].emplace_back(k.pt.y);
            keypoints[i].emplace_back(k.pt.x);
        }

        transposeNDWrapper(superPointResults[i].second, {1, 0}, buffer);
        std::copy(buffer.begin<float>(), buffer.end<float>(), std::back_inserter(descriptors[i]));
        buffer.release();
    }
    std::vector<Ort::OrtSessionHandler::DataOutputType> superGlueOrtOutput =
        superGlueOsh({imageShapes[0].data(), scores[0].data(), keypoints[0].data(), descriptors[0].data(),
                      imageShapes[1].data(), scores[1].data(), keypoints[1].data(), descriptors[1].data()});

    // match keypoints 0 to keypoints 1
    std::vector<int64_t> matchIndices(reinterpret_cast<int64_t*>(superGlueOrtOutput[0].first),
                                      reinterpret_cast<int64_t*>(superGlueOrtOutput[0].first) + numKeypoints0);

    std::vector<cv::DMatch> goodMatches;
    for (std::size_t i = 0; i < matchIndices.size(); ++i) {
        if (matchIndices[i] < 0) {
            continue;
        }
        cv::DMatch match;
        match.imgIdx = 0;
        match.queryIdx = i;
        match.trainIdx = matchIndices[i];
        goodMatches.emplace_back(match);
    }

    cv::Mat matchesImage;
    cv::drawMatches(images[0], superPointResults[0].first, images[1], superPointResults[1].first, goodMatches,
                    matchesImage, cv::Scalar::all(-1), cv::Scalar::all(-1), std::vector<char>(),
                    cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    cv::imwrite("super_point_super_glue_good_matches.jpg", matchesImage);
    cv::imshow("super_point_super_glue_good_matches", matchesImage);
    cv::waitKey();

    return EXIT_SUCCESS;
}

namespace
{
void normalizeDescriptors(cv::Mat* descriptors)
{
    cv::Mat rsquaredSumMat;
    cv::reduce(descriptors->mul(*descriptors), rsquaredSumMat, 1, cv::REDUCE_SUM);
    cv::sqrt(rsquaredSumMat, rsquaredSumMat);
    for (int i = 0; i < descriptors->rows; ++i) {
        float rsquaredSum = std::max<float>(rsquaredSumMat.ptr<float>()[i], 1e-12);
        descriptors->row(i) /= rsquaredSum;
    }
}

KeyPointAndDesc processOneFrameSuperPoint(const Ort::SuperPoint& superPointOsh, const cv::Mat& inputImg, float* dst,
                                          int borderRemove, float confidenceThresh, bool alignCorners, int distThresh)
{
    int origW = inputImg.cols, origH = inputImg.rows;
    cv::Mat scaledImg;
    cv::resize(inputImg, scaledImg, cv::Size(Ort::SuperPoint::IMG_W, Ort::SuperPoint::IMG_H), 0, 0, cv::INTER_CUBIC);
    superPointOsh.preprocess(dst, scaledImg.data, Ort::SuperPoint::IMG_W, Ort::SuperPoint::IMG_H,
                             Ort::SuperPoint::IMG_CHANNEL);
    auto inferenceOutput = superPointOsh({dst});

    std::vector<cv::KeyPoint> keyPoints = superPointOsh.getKeyPoints(inferenceOutput, borderRemove, confidenceThresh);

    std::vector<int> descriptorShape(inferenceOutput[1].second.begin(), inferenceOutput[1].second.end());
    cv::Mat coarseDescriptorMat(descriptorShape.size(), descriptorShape.data(), CV_32F,
                                inferenceOutput[1].first);  // 1 x 256 x H/8 x W/8

    std::vector<int> keepIndices =
        superPointOsh.nmsFast(keyPoints, Ort::SuperPoint::IMG_H, Ort::SuperPoint::IMG_W, distThresh);

    std::vector<cv::KeyPoint> keepKeyPoints;
    keepKeyPoints.reserve(keepIndices.size());
    std::transform(keepIndices.begin(), keepIndices.end(), std::back_inserter(keepKeyPoints),
                   [&keyPoints](int idx) { return keyPoints[idx]; });
    keyPoints = std::move(keepKeyPoints);

    cv::Mat descriptors = superPointOsh.getDescriptors(coarseDescriptorMat, keyPoints, Ort::SuperPoint::IMG_H,
                                                       Ort::SuperPoint::IMG_W, alignCorners);

    for (auto& keyPoint : keyPoints) {
        keyPoint.pt.x *= static_cast<float>(origW) / Ort::SuperPoint::IMG_W;
        keyPoint.pt.y *= static_cast<float>(origH) / Ort::SuperPoint::IMG_H;
    }

    return {keyPoints, descriptors};
}
}  // namespace
