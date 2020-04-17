/**
 * @file    TestImageClassification.cpp
 *
 * @author  btran
 *
 * Copyright (c) organization
 *
 */

#include <cassert>
#include <chrono>
#include <cmath>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include <ort_utility/ort_utility.hpp>

static constexpr int64_t IMG_WIDTH = 224;
static constexpr int64_t IMG_HEIGHT = 224;
static constexpr int64_t IMG_CHANNEL = 3;
static constexpr int64_t TEST_TIMES = 1000;

int main(int argc, char* argv[])
{
    if (argc != 3) {
        std::cerr << "Usage: [apps] [path/to/onnx/model] [path/to/image]" << std::endl;
        return EXIT_FAILURE;
    }

    const std::string ONNX_MODEL_PATH = argv[1];
    const std::string IMAGE_PATH = argv[2];

    Ort::ImageClassificationOrtSessionHandler osh(Ort::IMAGENET_NUM_CLASSES, ONNX_MODEL_PATH, 0);
    osh.initClassNames(Ort::IMAGENET_CLASSES);

    cv::Mat img = cv::imread(IMAGE_PATH);

    if (img.empty()) {
        std::cerr << "Failed to read input image" << std::endl;
        return EXIT_FAILURE;
    }

    cv::resize(img, img, cv::Size(IMG_WIDTH, IMG_HEIGHT));
    float* dst = new float[IMG_WIDTH * IMG_HEIGHT * IMG_CHANNEL];
    osh.preprocess(dst, img.data, IMG_WIDTH, IMG_HEIGHT, IMG_CHANNEL, Ort::IMAGENET_MEAN, Ort::IMAGENET_STD);

    std::chrono::high_resolution_clock::time_point begin = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < TEST_TIMES; ++i) {
        auto inferenceOutput = osh({reinterpret_cast<float*>(dst)});

        const int TOP_K = 5;
        // osh.topK({inferenceOutput[0].first}, TOP_K);
        std::cout << osh.topKToString({inferenceOutput[0].first}, TOP_K) << std::endl;
    }
    std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
    auto elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin);
    std::cout << elapsedTime.count() / 1000. << "[sec]" << std::endl;

    delete[] dst;

    return 0;
}
