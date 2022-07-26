/**
 * @file    SemanticSegmentationPaddleSegBisenetv2App.cpp
 *
 * @author  btran
 *
 */

#include <ort_utility/ort_utility.hpp>

#include "SemanticSegmentationPaddleSegBisenetv2.hpp"
#include "Utility.hpp"

namespace
{
cv::Mat processOneFrame(const Ort::SemanticSegmentationPaddleSegBisenetv2& osh, const cv::Mat& inputImg, float* dst,
                        float alpha = 0.4);
static const std::vector<cv::Scalar> COLORS = toCvScalarColors(Ort::CITY_SCAPES_COLOR_CHART);
}  // namespace

int main(int argc, char* argv[])
{
    if (argc != 3) {
        std::cerr << "Usage: [apps] [path/to/onnx/semantic/segmentation] [path/to/image]" << std::endl;
        return EXIT_FAILURE;
    }

    const std::string ONNX_MODEL_PATH = argv[1];
    const std::string IMAGE_PATH = argv[2];

    cv::Mat img = cv::imread(IMAGE_PATH);

    if (img.empty()) {
        std::cerr << "Failed to read input image" << std::endl;
        return EXIT_FAILURE;
    }

    Ort::SemanticSegmentationPaddleSegBisenetv2 osh(
        Ort::CITY_SCAPES_NUM_CLASSES, ONNX_MODEL_PATH, 0,
        std::vector<std::vector<int64_t>>{{1, Ort::SemanticSegmentationPaddleSegBisenetv2::IMG_CHANNEL,
                                           Ort::SemanticSegmentationPaddleSegBisenetv2::IMG_H,
                                           Ort::SemanticSegmentationPaddleSegBisenetv2::IMG_W}});

    osh.initClassNames(Ort::CITY_SCAPES_CLASSES);
    std::vector<float> dst(Ort::SemanticSegmentationPaddleSegBisenetv2::IMG_CHANNEL *
                           Ort::SemanticSegmentationPaddleSegBisenetv2::IMG_H *
                           Ort::SemanticSegmentationPaddleSegBisenetv2::IMG_W);

    auto result = processOneFrame(osh, img, dst.data());
    cv::Mat legend = drawColorChart(Ort::CITY_SCAPES_CLASSES, COLORS);
    cv::imshow("legend", legend);
    cv::imshow("overlaid result", result);
    cv::waitKey(0);

    return EXIT_SUCCESS;
}

namespace
{
cv::Mat processOneFrame(const Ort::SemanticSegmentationPaddleSegBisenetv2& osh, const cv::Mat& inputImg, float* dst,
                        float alpha)
{
    int origW = inputImg.cols, origH = inputImg.rows;
    std::vector<float> originImageSize{static_cast<float>(origH), static_cast<float>(origW)};
    cv::Mat scaledImg = inputImg.clone();
    cv::resize(inputImg, scaledImg,
               cv::Size(Ort::SemanticSegmentationPaddleSegBisenetv2::IMG_W,
                        Ort::SemanticSegmentationPaddleSegBisenetv2::IMG_H),
               0, 0, cv::INTER_CUBIC);
    cv::cvtColor(scaledImg, scaledImg, cv::COLOR_BGR2RGB);
    osh.preprocess(dst, scaledImg.data, Ort::SemanticSegmentationPaddleSegBisenetv2::IMG_W,
                   Ort::SemanticSegmentationPaddleSegBisenetv2::IMG_H, 3);
    auto inferenceOutput = osh({dst});

    // tips: when you have done all the tricks but still get the wrong output result,
    // try checking the type of inferenceOutput
    int64_t* data = reinterpret_cast<int64_t*>(inferenceOutput[0].first);
    cv::Mat segm(Ort::SemanticSegmentationPaddleSegBisenetv2::IMG_H, Ort::SemanticSegmentationPaddleSegBisenetv2::IMG_W,
                 CV_8UC(3));
    for (int i = 0; i < Ort::SemanticSegmentationPaddleSegBisenetv2::IMG_H; ++i) {
        cv::Vec3b* ptrSegm = segm.ptr<cv::Vec3b>(i);
        for (int j = 0; j < Ort::SemanticSegmentationPaddleSegBisenetv2::IMG_W; ++j) {
            const auto& color = COLORS[data[i * Ort::SemanticSegmentationPaddleSegBisenetv2::IMG_W + j]];
            ptrSegm[j] = cv::Vec3b(color[0], color[1], color[2]);
        }
    }

    cv::resize(segm, segm, inputImg.size(), 0, 0, cv::INTER_NEAREST);
    segm = (1 - alpha) * inputImg + alpha * segm;
    return segm;
}
}  // namespace
