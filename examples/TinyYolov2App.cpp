/**
 * @file    TinyYolov2App.cpp
 *
 * @author  btran
 *
 * Copyright (c) organization
 *
 */

#include <chrono>
#include <memory>

#include <opencv2/opencv.hpp>

#include <ort_utility/ort_utility.hpp>

#include "TinyYolov2.hpp"
#include "Utility.hpp"

static constexpr const float CONFIDENCE_THRESHOLD = 0.5;
static constexpr const float NMS_THRESHOLD = 0.6;
static const std::vector<cv::Scalar> COLORS = toCvScalarColors(Ort::VOC_COLOR_CHART);

namespace
{
cv::Mat processOneFrame(Ort::TinyYolov2& osh, const cv::Mat& inputImg, float* dst);
}  // namespace

int main(int argc, char* argv[])
{
    if (argc != 3) {
        std::cerr << "Usage: [apps] [path/to/onnx/yolov3-tiny.onnx] [path/to/image]" << std::endl;
        return EXIT_FAILURE;
    }

    const std::string ONNX_MODEL_PATH = argv[1];
    const std::string IMAGE_PATH = argv[2];

    Ort::TinyYolov2 osh(Ort::VOC_NUM_CLASSES, ONNX_MODEL_PATH, 0,
                        std::vector<std::vector<int64_t>>{{1, IMG_CHANNEL, IMG_WIDTH, IMG_HEIGHT}});

    osh.initClassNames(Ort::VOC_CLASSES);
    std::array<float, IMG_WIDTH * IMG_HEIGHT * IMG_CHANNEL> dst;

    cv::Mat img = cv::imread(IMAGE_PATH);

    if (img.empty()) {
        std::cerr << "Failed to read input image" << std::endl;
        return EXIT_FAILURE;
    }

    auto resultImg = ::processOneFrame(osh, img, dst.data());

    cv::imwrite("result.jpg", resultImg);

    return EXIT_SUCCESS;
}

namespace
{
cv::Mat processOneFrame(Ort::TinyYolov2& osh, const cv::Mat& inputImg, float* dst)
{
    cv::Mat result;
    cv::resize(inputImg, result, cv::Size(IMG_WIDTH, IMG_HEIGHT));

    osh.preprocess(dst, result.data, IMG_WIDTH, IMG_HEIGHT, IMG_CHANNEL);
    auto inferenceOutput = osh({dst});
    assert(inferenceOutput.size() == 1);

    std::vector<float> outputData(inferenceOutput.front(), inferenceOutput.front() + NUM_BOXES);

    auto processedResult = osh.postProcess(inferenceOutput, CONFIDENCE_THRESHOLD);
    std::vector<std::array<float, 4>> bboxes;
    std::vector<float> scores;
    std::vector<uint64_t> classIndices;
    std::tie(bboxes, scores, classIndices) = processedResult;

    if (bboxes.size() == 0) {
        return result;
    }

    auto afterNmsIndices = Ort::nms(bboxes, scores, NMS_THRESHOLD);

    std::vector<std::array<float, 4>> afterNmsBboxes;
    std::vector<uint64_t> afterNmsClassIndices;

    afterNmsBboxes.reserve(afterNmsIndices.size());
    afterNmsClassIndices.reserve(afterNmsIndices.size());

    for (const auto idx : afterNmsIndices) {
        afterNmsBboxes.emplace_back(bboxes[idx]);
        afterNmsClassIndices.emplace_back(classIndices[idx]);
    }

    result = ::visualizeOneImage(result, afterNmsBboxes, afterNmsClassIndices, COLORS, osh.classNames());

    return result;
}
}  // namespace
