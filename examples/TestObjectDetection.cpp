/**
 * @file    TestObjectDetection.cpp
 *
 * @author  btran
 *
 * Copyright (c) organization
 *
 */

#include <chrono>

#include <opencv2/opencv.hpp>

#include <ort_utility/ort_utility.hpp>

static constexpr int64_t IMG_WIDTH = 416;
static constexpr int64_t IMG_HEIGHT = 416;
static constexpr int64_t IMG_CHANNEL = 3;
static constexpr int64_t TEST_TIMES = 1;

int main(int argc, char* argv[])
{
    if (argc != 3) {
        std::cerr << "Usage: [apps] [path/to/onnx/model] [path/to/image]" << std::endl;
        return EXIT_FAILURE;
    }

    const std::string ONNX_MODEL_PATH = argv[1];
    const std::string IMAGE_PATH = argv[2];

    Ort::ObjectDetectionOrtSessionHandler osh(
        Ort::IMAGENET_NUM_CLASSES, ONNX_MODEL_PATH, 0,
        std::vector<std::vector<int64_t>>{{1, IMG_CHANNEL, IMG_WIDTH, IMG_HEIGHT}, {1, 2}});

    osh.initClassNames(Ort::IMAGENET_CLASSES);

    cv::Mat img = cv::imread(IMAGE_PATH);

    if (img.empty()) {
        std::cerr << "Failed to read input image" << std::endl;
        return EXIT_FAILURE;
    }

    cv::resize(img, img, cv::Size(IMG_WIDTH, IMG_HEIGHT));
    float* dst = new float[IMG_WIDTH * IMG_HEIGHT * IMG_CHANNEL];

    osh.preprocess(dst, img.data, IMG_WIDTH, IMG_HEIGHT, IMG_CHANNEL);

    std::vector<float> IMAGE_SHAPE = {IMG_WIDTH, IMG_HEIGHT};

    std::chrono::high_resolution_clock::time_point begin = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < TEST_TIMES; ++i) {
        auto inferenceOutput = osh({reinterpret_cast<float*>(dst), IMAGE_SHAPE.data()});
        std::cout << inferenceOutput.size() << "\n";
    }

    std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
    auto elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin);
    std::cout << elapsedTime.count() / 1000. << "[sec]" << std::endl;

    delete[] dst;
    return EXIT_SUCCESS;
}
