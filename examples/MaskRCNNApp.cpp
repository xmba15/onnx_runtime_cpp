/**
 * @file    MaskRCNNApp.cpp
 *
 * @author  btran
 *
 * @date    2020-05-18
 *
 * Copyright (c) organization
 *
 */
#include <chrono>
#include <iostream>
#include <memory>

#include <opencv2/opencv.hpp>

#include <ort_utility/ort_utility.hpp>

#include "MaskRCNN.hpp"
#include "Utility.hpp"

static constexpr const float CONFIDENCE_THRESHOLD = 0.5;
static const std::vector<cv::Scalar> COLORS = toCvScalarColors(Ort::MSCOCO_COLOR_CHART);

namespace
{
cv::Mat processOneFrame(Ort::MaskRCNN& osh, const cv::Mat& inputImg, float* dst, const float confThresh);
}  // namespace

int main(int argc, char* argv[])
{
    if (argc != 3) {
        std::cerr << "Usage: [apps] [path/to/onnx/maskrcnn.onnx] [path/to/image]" << std::endl;
        return EXIT_FAILURE;
    }

    const std::string ONNX_MODEL_PATH = argv[1];
    const std::string IMAGE_PATH = argv[2];

    Ort::MaskRCNN osh(Ort::MSCOCO_NUM_CLASSES, ONNX_MODEL_PATH, 0,
                      std::vector<std::vector<int64_t>>{
                          {Ort::MaskRCNN::IMG_CHANNEL, Ort::MaskRCNN::IMG_HEIGHT, Ort::MaskRCNN::IMG_WIDTH}});

    // Ort::MaskRCNN osh(Ort::MSCOCO_NUM_CLASSES, ONNX_MODEL_PATH, 0);

    osh.initClassNames(Ort::MSCOCO_CLASSES);

    std::vector<float> dst(Ort::MaskRCNN::IMG_WIDTH * Ort::MaskRCNN::IMG_HEIGHT * Ort::MaskRCNN::IMG_CHANNEL);

    cv::Mat img = cv::imread(IMAGE_PATH);

    if (img.empty()) {
        std::cerr << "Failed to read input image" << std::endl;
        return EXIT_FAILURE;
    }

    auto resultImg = ::processOneFrame(osh, img, dst.data(), CONFIDENCE_THRESHOLD);
    cv::imwrite("result.jpg", resultImg);

    return EXIT_SUCCESS;
}

namespace
{
cv::Mat processOneFrame(Ort::MaskRCNN& osh, const cv::Mat& inputImg, float* dst, float confThresh)
{
    cv::Mat result;
    cv::resize(inputImg, result, cv::Size(Ort::MaskRCNN::IMG_WIDTH, Ort::MaskRCNN::IMG_HEIGHT));
    result -= cv::Scalar(102.9801, 115.9465, 122.7717);

    osh.preprocess(dst, result.data, Ort::MaskRCNN::IMG_WIDTH, Ort::MaskRCNN::IMG_HEIGHT, 3);

    // boxes, labels, scores, masks
    auto inferenceOutput = osh({dst});

    assert(inferenceOutput[1].second.size() == 1);
    size_t nBoxes = inferenceOutput[1].second[0];

    std::vector<std::array<float, 4>> bboxes;
    std::vector<uint64_t> classIndices;

    bboxes.reserve(nBoxes);
    classIndices.reserve(nBoxes);

    for (size_t i = 0; i < nBoxes; ++i) {
        if (inferenceOutput[2].first[i] > confThresh) {
            float xmin = inferenceOutput[0].first[i * 4 + 0];
            float ymin = inferenceOutput[0].first[i * 4 + 1];
            float xmax = inferenceOutput[0].first[i * 4 + 2];
            float ymax = inferenceOutput[0].first[i * 4 + 2];

            bboxes.emplace_back(std::array<float, 4>{xmin, ymin, xmax, ymax});
            classIndices.emplace_back(reinterpret_cast<int64_t*>(inferenceOutput[1].first)[i]);
        }
    }

    if (bboxes.size() == 0) {
        return result;
    }

    result = ::visualizeOneImage(result, bboxes, classIndices, COLORS, osh.classNames());

    return result;
}
}  // namespace
