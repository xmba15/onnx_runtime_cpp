/**
 * @file    Yolov3App.cpp
 *
 * @author  btran
 *
 */

#include <ort_utility/ort_utility.hpp>

#include "Utility.hpp"
#include "Yolov3.hpp"

static const std::vector<std::string> MSCOCO_WITHOUT_BG_CLASSES(Ort::MSCOCO_CLASSES.begin() + 1,
                                                                Ort::MSCOCO_CLASSES.end());
static constexpr int64_t NUM_CLASSES = 80;
static const std::vector<std::array<int, 3>> COLOR_CHART = Ort::generateColorCharts(NUM_CLASSES);

static constexpr const float CONFIDENCE_THRESHOLD = 0.2;
static const std::vector<cv::Scalar> COLORS = toCvScalarColors(COLOR_CHART);

namespace
{
cv::Mat processOneFrame(Ort::Yolov3& osh, const cv::Mat& inputImg, float* dst, const float confThresh = 0.15);
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

    Ort::Yolov3 osh(NUM_CLASSES, ONNX_MODEL_PATH, 0,
                    std::vector<std::vector<int64_t>>{
                        {1, Ort::Yolov3::IMG_CHANNEL, Ort::Yolov3::IMG_H, Ort::Yolov3::IMG_W}, {1, 2}});

    osh.initClassNames(MSCOCO_WITHOUT_BG_CLASSES);

    std::vector<float> dst(Ort::Yolov3::IMG_CHANNEL * Ort::Yolov3::IMG_H * Ort::Yolov3::IMG_W);
    auto result = processOneFrame(osh, img, dst.data());
    cv::imwrite("result.jpg", result);

    return 0;
}

namespace
{
cv::Mat processOneFrame(Ort::Yolov3& osh, const cv::Mat& inputImg, float* dst, const float confThresh)
{
    int origW = inputImg.cols, origH = inputImg.rows;
    std::vector<float> originImageSize{static_cast<float>(origH), static_cast<float>(origW)};

    float scale = std::min<float>(1.0 * Ort::Yolov3::IMG_W / origW, 1.0 * Ort::Yolov3::IMG_H / origH);

    cv::Mat scaledImg;
    cv::resize(inputImg, scaledImg, cv::Size(), scale, scale, cv::INTER_CUBIC);
    cv::Mat processedImg(Ort::Yolov3::IMG_H, Ort::Yolov3::IMG_W, CV_8UC3, cv::Scalar(128, 128, 128));

    scaledImg.copyTo(processedImg(cv::Rect((Ort::Yolov3::IMG_W - scaledImg.cols) / 2,
                                           (Ort::Yolov3::IMG_H - scaledImg.rows) / 2, scaledImg.cols, scaledImg.rows)));

    osh.preprocess(dst, processedImg.data, Ort::Yolov3::IMG_W, Ort::Yolov3::IMG_H, 3);
    auto inferenceOutput = osh({dst, originImageSize.data()});
    int numAnchors = inferenceOutput[0].second[1];
    int numOutputBboxes = inferenceOutput[2].second[0];
    DEBUG_LOG("number anchor candidates: %d", numAnchors);
    DEBUG_LOG("number output bboxes: %d", numOutputBboxes);

    const float* candidateBboxes = inferenceOutput[0].first;
    const float* candidateScores = inferenceOutput[1].first;
    const int32_t* outputIndices = reinterpret_cast<int32_t*>(inferenceOutput[2].first);

    std::vector<std::array<float, 4>> bboxes;
    std::vector<float> scores;
    std::vector<uint64_t> classIndices;

    for (int i = 0; i < numOutputBboxes; ++i) {
        int curClassIdx = outputIndices[i * 3 + 1];
        int curCandidateIdx = outputIndices[i * 3 + 2];

        float curScore = candidateScores[curClassIdx * numAnchors + curCandidateIdx];

        if (curScore < confThresh) {
            continue;
        }

        float ymin = candidateBboxes[curCandidateIdx * 4 + 0];
        float xmin = candidateBboxes[curCandidateIdx * 4 + 1];
        float ymax = candidateBboxes[curCandidateIdx * 4 + 2];
        float xmax = candidateBboxes[curCandidateIdx * 4 + 3];

        xmin = std::max<float>(xmin, 0);
        ymin = std::max<float>(ymin, 0);
        xmax = std::min<float>(xmax, inputImg.cols - 1);
        ymax = std::min<float>(ymax, inputImg.rows - 1);

        bboxes.emplace_back(std::array<float, 4>{xmin, ymin, xmax, ymax});
        scores.emplace_back(curScore);
        classIndices.emplace_back(curClassIdx);
    }

    return bboxes.empty() ? inputImg : visualizeOneImage(inputImg, bboxes, classIndices, COLORS, osh.classNames());
}
}  // namespace
