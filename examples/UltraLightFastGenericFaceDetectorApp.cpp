/**
 * @file    UltraLightFastGenericFaceDetectorApp.cpp
 *
 * @author  btran
 *
 */

#include "Utility.hpp"

#include "UltraLightFastGenericFaceDetector.hpp"

static const std::vector<std::string> FACE_CLASSES = {"face"};
static constexpr int64_t FACE_NUM_CLASSES = 1;
static const std::vector<std::array<int, 3>> FACE_COLOR_CHART =
    Ort::generateColorCharts(FACE_NUM_CLASSES, 2020 /* choosing awesome 2020 as seed point */);

static constexpr float CONFIDENCE_THRESHOLD = 0.7;
static constexpr float NMS_THRESHOLD = 0.3;
static const std::vector<cv::Scalar> COLORS = toCvScalarColors(FACE_COLOR_CHART);

namespace
{
cv::Mat processOneFrame(Ort::UltraLightFastGenericFaceDetector& osh, const cv::Mat& inputImg, float* dst,
                        const float confThresh = CONFIDENCE_THRESHOLD, const float nmsThresh = NMS_THRESHOLD);
}  // namespace

int main(int argc, char* argv[])
{
    if (argc != 3) {
        std::cerr << "Usage: [apps] [path/to/onnx/version-RFB-640.onnx] [path/to/image]" << std::endl;
        return EXIT_FAILURE;
    }

    const std::string ONNX_MODEL_PATH = argv[1];
    const std::string IMAGE_PATH = argv[2];

    cv::Mat img = cv::imread(IMAGE_PATH);

    if (img.empty()) {
        std::cerr << "Failed to read input image" << std::endl;
        return EXIT_FAILURE;
    }

    Ort::UltraLightFastGenericFaceDetector osh(
        ONNX_MODEL_PATH, 0,
        std::vector<std::vector<int64_t>>{{1, Ort::UltraLightFastGenericFaceDetector::IMG_CHANNEL,
                                           Ort::UltraLightFastGenericFaceDetector::IMG_H,
                                           Ort::UltraLightFastGenericFaceDetector::IMG_W}});

    osh.initClassNames(FACE_CLASSES);

    std::vector<float> dst(Ort::UltraLightFastGenericFaceDetector::IMG_CHANNEL *
                           Ort::UltraLightFastGenericFaceDetector::IMG_H *
                           Ort::UltraLightFastGenericFaceDetector::IMG_W);
    auto result = processOneFrame(osh, img, dst.data());
    cv::imwrite("result.jpg", result);

    return 0;
}

namespace
{
cv::Mat processOneFrame(Ort::UltraLightFastGenericFaceDetector& osh, const cv::Mat& inputImg, float* dst,
                        const float confThresh, const float nmsThresh)
{
    int origH = inputImg.rows;
    int origW = inputImg.cols;

    cv::Mat processedImg;
    cv::resize(inputImg, processedImg, cv::Size(osh.IMG_W, osh.IMG_H));

    osh.preprocess(dst, processedImg.data, osh.IMG_W, osh.IMG_H, 3);
    auto inferenceOutput = osh({dst});

    // output includes two tensors:
    // confidences: 1 x 17640 x 2 (2 represents 2 classes of background and face)
    // bboxes: 1 x 17640 x 4 (4 represents bbox coordinates)

    int numAnchors = inferenceOutput[0].second[1];

    std::vector<std::array<float, 4>> bboxes;
    std::vector<float> scores;
    std::vector<uint64_t> classIndices;

    for (auto indices = std::make_pair(0, 0); indices.first < numAnchors * 2 && indices.second < numAnchors * 4;
         indices.first += 2, indices.second += 4) {
        float conf = inferenceOutput[0].first[indices.first + 1];
        if (conf < confThresh) {
            continue;
        }
        float xmin = inferenceOutput[1].first[indices.second + 0] * origW;
        float ymin = inferenceOutput[1].first[indices.second + 1] * origH;
        float xmax = inferenceOutput[1].first[indices.second + 2] * origW;
        float ymax = inferenceOutput[1].first[indices.second + 3] * origH;

        bboxes.emplace_back(std::array<float, 4>{xmin, ymin, xmax, ymax});
        scores.emplace_back(conf);

        // only consider face
        classIndices.emplace_back(0);
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
