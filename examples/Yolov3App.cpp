/**
 * @file    Yolov3App.cpp
 *
 * @author  btran
 *
 * @date    2020-05-31
 *
 * Copyright (c) organization
 *
 */

#include <ort_utility/ort_utility.hpp>

#include "Utility.hpp"
#include "Yolov3.hpp"

static const std::vector<std::string> BIRD_CLASSES = {"bird_small", "bird_medium", "bird_large"};
static constexpr int64_t BIRD_NUM_CLASSES = 3;
static const std::vector<std::array<int, 3>> BIRD_COLOR_CHART = Ort::generateColorCharts(BIRD_NUM_CLASSES);

static constexpr const float CONFIDENCE_THRESHOLD = 0.2;
static constexpr const float NMS_THRESHOLD = 0.6;
static const std::vector<cv::Scalar> COLORS = toCvScalarColors(BIRD_COLOR_CHART);

namespace
{
cv::Mat processOneFrame(Ort::Yolov3& osh, const cv::Mat& inputImg, float* dst, const float confThresh = 0.15,
                        const float nmsThresh = 0.5);
}  // namespace

int main(int argc, char* argv[])
{
    if (argc != 3) {
        std::cerr << "Usage: [apps] [path/to/onnx/yolov3.onnx] [path/to/image]" << std::endl;
        return EXIT_FAILURE;
    }

    const std::string ONNX_MODEL_PATH = argv[1];
    const std::string IMAGE_PATH = argv[2];

    cv::Mat img = cv::imread(IMAGE_PATH);

    if (img.empty()) {
        std::cerr << "Failed to read input image" << std::endl;
        return EXIT_FAILURE;
    }

    Ort::Yolov3 osh(
        BIRD_NUM_CLASSES, ONNX_MODEL_PATH, 0,
        std::vector<std::vector<int64_t>>{{1, Ort::Yolov3::IMG_CHANNEL, Ort::Yolov3::IMG_H, Ort::Yolov3::IMG_W}});

    osh.initClassNames(BIRD_CLASSES);

    std::vector<float> dst(Ort::Yolov3::IMG_CHANNEL * Ort::Yolov3::IMG_H * Ort::Yolov3::IMG_W);
    auto result = processOneFrame(osh, img, dst.data());
    cv::imwrite("result.jpg", result);

    return 0;
}

namespace
{
cv::Mat processOneFrame(Ort::Yolov3& osh, const cv::Mat& inputImg, float* dst, const float confThresh,
                        const float nmsThresh)
{
    int origH = inputImg.rows;
    int origW = inputImg.cols;
    float ratioH = origH * 1.0 / osh.IMG_H;
    float ratioW = origW * 1.0 / osh.IMG_W;

    cv::Mat processedImg;
    cv::resize(inputImg, processedImg, cv::Size(osh.IMG_W, osh.IMG_H));

    osh.preprocess(dst, processedImg.data, Ort::Yolov3::IMG_W, Ort::Yolov3::IMG_H, 3);
    auto inferenceOutput = osh({dst});
    int numAnchors = inferenceOutput[0].second[1];
    int numAttrs = inferenceOutput[0].second[2];

    std::vector<std::array<float, 4>> bboxes;
    std::vector<float> scores;
    std::vector<uint64_t> classIndices;

    for (int i = 0; i < numAnchors * numAttrs; i += numAttrs) {
        float conf = inferenceOutput[0].first[i + 4];

        if (conf >= confThresh) {
            float xcenter = inferenceOutput[0].first[i + 0];
            float ycenter = inferenceOutput[0].first[i + 1];
            float width = inferenceOutput[0].first[i + 2];
            float height = inferenceOutput[0].first[i + 3];

            float xmin = xcenter - width / 2;
            float ymin = ycenter - height / 2;
            float xmax = xcenter + width / 2;
            float ymax = ycenter + height / 2;
            xmin = std::max<float>(xmin, 0);
            ymin = std::max<float>(ymin, 0);
            xmax = std::min<float>(xmax, osh.IMG_W - 1);
            ymax = std::min<float>(ymax, osh.IMG_H - 1);

            bboxes.emplace_back(std::array<float, 4>{xmin * ratioW, ymin * ratioH, xmax * ratioW, ymax * ratioH});

            scores.emplace_back(conf);
            int maxIdx = std::max_element(inferenceOutput[0].first + i + 5,
                                          inferenceOutput[0].first + i + 5 + osh.numClasses()) -
                         (inferenceOutput[0].first + i + 5);
            classIndices.emplace_back(maxIdx);
        }
    }

    if (bboxes.size() == 0) {
        return inputImg;
    }

    auto afterNmsIndices = Ort::nms(bboxes, scores, nmsThresh);

    std::vector<std::array<float, 4>> afterNmsBboxes;
    std::vector<uint64_t> afterNmsClassIndices;

    afterNmsBboxes.reserve(afterNmsIndices.size());
    afterNmsClassIndices.reserve(afterNmsIndices.size());

    for (const auto idx : afterNmsIndices) {
        afterNmsBboxes.emplace_back(bboxes[idx]);
        afterNmsClassIndices.emplace_back(classIndices[idx]);
    }

    return visualizeOneImage(inputImg, afterNmsBboxes, afterNmsClassIndices, COLORS, osh.classNames());
}
}  // namespace
