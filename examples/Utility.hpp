/**
 * @file    Utility.hpp
 *
 * @author  btran
 *
 * @date    2020-05-04
 *
 * Copyright (c) organization
 *
 */

#pragma once

#include <algorithm>
#include <string>
#include <utility>
#include <vector>

#include <opencv2/opencv.hpp>

namespace
{
inline std::vector<cv::Scalar> toCvScalarColors(const std::vector<std::array<int, 3>>& colors)
{
    std::vector<cv::Scalar> result;
    result.reserve(colors.size());
    std::transform(std::begin(colors), std::end(colors), std::back_inserter(result),
                   [](const auto& elem) { return cv::Scalar(elem[0], elem[1], elem[2]); });

    return result;
}

inline cv::Mat visualizeOneImage(const cv::Mat& img, const std::vector<std::array<float, 4>>& bboxes,
                                 const std::vector<uint64_t>& classIndices, const std::vector<cv::Scalar>& allColors,
                                 const std::vector<std::string>& allClassNames = {})
{
    assert(bboxes.size() == classIndices.size());
    if (!allClassNames.empty()) {
        assert(allClassNames.size() > *std::max_element(classIndices.begin(), classIndices.end()));
        assert(allColors.size() == allClassNames.size());
    }

    cv::Mat result = img.clone();

    for (size_t i = 0; i < bboxes.size(); ++i) {
        const auto& curBbox = bboxes[i];
        const uint64_t classIdx = classIndices[i];
        const cv::Scalar& curColor = allColors[classIdx];
        const std::string curLabel = allClassNames.empty() ? std::to_string(classIdx) : allClassNames[classIdx];

        cv::rectangle(result, cv::Point(curBbox[0], curBbox[1]), cv::Point(curBbox[2], curBbox[3]), curColor, 2);

        int baseLine = 0;
        cv::Size labelSize = cv::getTextSize(curLabel, cv::FONT_HERSHEY_COMPLEX, 0.35, 1, &baseLine);
        cv::rectangle(result, cv::Point(curBbox[0], curBbox[1]),
                      cv::Point(curBbox[0] + labelSize.width, curBbox[1] + static_cast<int>(1.3 * labelSize.height)),
                      curColor, -1);
        cv::putText(result, curLabel, cv::Point(curBbox[0], curBbox[1] + labelSize.height), cv::FONT_HERSHEY_COMPLEX,
                    0.35, cv::Scalar(255, 255, 255));
    }

    return result;
}

inline cv::Mat visualizeOneImageWithMask(const cv::Mat& img, const std::vector<std::array<float, 4>>& bboxes,
                                         const std::vector<uint64_t>& classIndices, const std::vector<cv::Mat>& masks,
                                         const std::vector<cv::Scalar>& allColors,
                                         const std::vector<std::string>& allClassNames = {},
                                         const float maskThreshold = 0.5)
{
    assert(bboxes.size() == classIndices.size());
    if (!allClassNames.empty()) {
        assert(allClassNames.size() > *std::max_element(classIndices.begin(), classIndices.end()));
        assert(allColors.size() == allClassNames.size());
    }

    cv::Mat result = img.clone();

    for (size_t i = 0; i < bboxes.size(); ++i) {
        const auto& curBbox = bboxes[i];
        const uint64_t classIdx = classIndices[i];
        cv::Mat curMask = masks[i].clone();
        const cv::Scalar& curColor = allColors[classIdx];
        const std::string curLabel = allClassNames.empty() ? std::to_string(classIdx) : allClassNames[classIdx];

        cv::rectangle(result, cv::Point(curBbox[0], curBbox[1]), cv::Point(curBbox[2], curBbox[3]), curColor, 2);

        int baseLine = 0;
        cv::Size labelSize = cv::getTextSize(curLabel, cv::FONT_HERSHEY_COMPLEX, 0.35, 1, &baseLine);
        cv::rectangle(result, cv::Point(curBbox[0], curBbox[1]),
                      cv::Point(curBbox[0] + labelSize.width, curBbox[1] + static_cast<int>(1.3 * labelSize.height)),
                      curColor, -1);
        cv::putText(result, curLabel, cv::Point(curBbox[0], curBbox[1] + labelSize.height), cv::FONT_HERSHEY_COMPLEX,
                    0.35, cv::Scalar(255, 255, 255));

        // ---------------------------------------------------------------------//
        // Visualize masks

        const cv::Rect curBoxRect(cv::Point(curBbox[0], curBbox[1]), cv::Point(curBbox[2], curBbox[3]));

        cv::resize(curMask, curMask, curBoxRect.size());

        cv::Mat finalMask = (curMask > maskThreshold);

        cv::Mat coloredRoi = (0.3 * curColor + 0.7 * result(curBoxRect));

        coloredRoi.convertTo(coloredRoi, CV_8UC3);

        std::vector<cv::Mat> contours;
        cv::Mat hierarchy;
        finalMask.convertTo(finalMask, CV_8U);

        cv::findContours(finalMask, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);
        cv::drawContours(coloredRoi, contours, -1, curColor, 5, cv::LINE_8, hierarchy, 100);
        coloredRoi.copyTo(result(curBoxRect), finalMask);
    }

    return result;
}

inline cv::Mat drawColorChart(const std::vector<std::string>& classes, const std::vector<cv::Scalar>& colors)
{
    cv::Mat legend = cv::Mat::zeros((classes.size() * 25) + 25, 300, CV_8UC(3));
    for (std::size_t i = 0; i < classes.size(); ++i) {
        cv::putText(legend, classes[i], cv::Point(5, (i * 25) + 17), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                    cv::Scalar(0, 0, 255), 2);
        cv::rectangle(legend, cv::Point(100, (i * 25)), cv::Point(300, (i * 25) + 25), colors[i], -1);
    }

    return legend;
}

inline void transposeNDWrapper(cv::InputArray src_, const std::vector<int>& order, cv::OutputArray dst_)
{
#if (CV_MAJOR_VERSION > 4 || (CV_MAJOR_VERSION == 4 && CV_MINOR_VERSION >= 6))
    cv::transposeND(src_, order, dst_);
#else
    cv::Mat inp = src_.getMat();
    CV_Assert(inp.isContinuous());
    CV_CheckEQ(inp.channels(), 1, "Input array should be single-channel");
    CV_CheckEQ(order.size(), static_cast<size_t>(inp.dims), "Number of dimensions shouldn't change");

    auto order_ = order;
    std::sort(order_.begin(), order_.end());
    for (size_t i = 0; i < order_.size(); ++i) {
        CV_CheckEQ(static_cast<size_t>(order_[i]), i, "New order should be a valid permutation of the old one");
    }

    std::vector<int> newShape(order.size());
    for (size_t i = 0; i < order.size(); ++i) {
        newShape[i] = inp.size[order[i]];
    }

    dst_.create(static_cast<int>(newShape.size()), newShape.data(), inp.type());
    cv::Mat out = dst_.getMat();
    CV_Assert(out.isContinuous());
    CV_Assert(inp.data != out.data);

    int continuous_idx = 0;
    for (int i = static_cast<int>(order.size()) - 1; i >= 0; --i) {
        if (order[i] != i) {
            continuous_idx = i + 1;
            break;
        }
    }

    size_t continuous_size = continuous_idx == 0 ? out.total() : out.step1(continuous_idx - 1);
    size_t outer_size = out.total() / continuous_size;

    std::vector<size_t> steps(order.size());
    for (int i = 0; i < static_cast<int>(steps.size()); ++i) {
        steps[i] = inp.step1(order[i]);
    }

    auto* src = inp.ptr<const unsigned char>();
    auto* dst = out.ptr<unsigned char>();

    size_t src_offset = 0;
    size_t es = out.elemSize();
    for (size_t i = 0; i < outer_size; ++i) {
        std::memcpy(dst, src + es * src_offset, es * continuous_size);
        dst += es * continuous_size;
        for (int j = continuous_idx - 1; j >= 0; --j) {
            src_offset += steps[j];
            if ((src_offset / steps[j]) % out.size[j] != 0) {
                break;
            }
            src_offset -= steps[j] * out.size[j];
        }
    }
#endif
}

/**
 *  @brief https://mmcv.readthedocs.io/en/latest/_modules/mmcv/ops/point_sample.html
 */
inline cv::Mat bilinearGridSample(const cv::Mat& input, const cv::Mat& grid, bool alignCorners)
{
    // input: B x C x Hi x Wi
    // grid: B x Hg x Wg x 2

    if (input.size[0] != grid.size[0]) {
        throw std::runtime_error("input and grid need to have the same batch size");
    }
    int batch = input.size[0];
    int channel = input.size[1];
    int height = input.size[2];
    int width = input.size[3];

    int numKeyPoints = grid.size[2];
    cv::Mat yMat = grid({cv::Range::all(), cv::Range::all(), cv::Range::all(), cv::Range(0, 1)})
                       .clone()
                       .reshape(1, {batch, grid.size[1] * grid.size[2]});
    cv::Mat xMat = grid({cv::Range::all(), cv::Range::all(), cv::Range::all(), cv::Range(1, 2)})
                       .clone()
                       .reshape(1, {batch, grid.size[1] * grid.size[2]});

    if (alignCorners) {
        xMat = ((xMat + 1) / 2) * (width - 1);
        yMat = ((yMat + 1) / 2) * (height - 1);
    } else {
        xMat = ((xMat + 1) * width - 1) / 2;
        yMat = ((yMat + 1) * height - 1) / 2;
    }

    // floor
    cv::Mat x0Mat = xMat - 0.5;
    cv::Mat y0Mat = yMat - 0.5;
    x0Mat.convertTo(x0Mat, CV_32S);
    y0Mat.convertTo(y0Mat, CV_32S);

    x0Mat.convertTo(x0Mat, CV_32F);
    y0Mat.convertTo(y0Mat, CV_32F);

    cv::Mat x1Mat = x0Mat + 1;
    cv::Mat y1Mat = x0Mat + 1;

    std::vector<cv::Mat> weights = {(x1Mat - xMat).mul(y1Mat - yMat), (x1Mat - xMat).mul(yMat - y0Mat),
                                    (xMat - x0Mat).mul(y1Mat - yMat), (xMat - x0Mat).mul(yMat - y0Mat)};

    cv::Mat result = cv::Mat::zeros(3, std::vector<int>{batch, channel, grid.size[1] * grid.size[2]}.data(), CV_32F);

    auto isCoordSafe = [](int size, int maxSize) -> bool { return size > 0 && size < maxSize; };

    for (int b = 0; b < batch; ++b) {
        for (int i = 0; i < grid.size[1] * grid.size[2]; ++i) {
            int x0 = x0Mat.at<float>(b, i);
            int y0 = y0Mat.at<float>(b, i);
            int x1 = x1Mat.at<float>(b, i);
            int y1 = y1Mat.at<float>(b, i);

            std::vector<std::pair<int, int>> pairs = {{x0, y0}, {x0, y1}, {x1, y0}, {x1, y1}};
            std::vector<cv::Mat> Is(4, cv::Mat::zeros(channel, 1, CV_32F));

            for (int k = 0; k < 4; ++k) {
                if (isCoordSafe(pairs[k].first, width) && isCoordSafe(pairs[k].second, height)) {
                    Is[k] =
                        input({cv::Range(b, b + 1), cv::Range::all(), cv::Range(pairs[k].second, pairs[k].second + 1),
                               cv::Range(pairs[k].first, pairs[k].first + 1)})
                            .clone()
                            .reshape(1, channel);
                }
            }

            cv::Mat curDescriptor = Is[0] * weights[0].at<float>(i) + Is[1] * weights[1].at<float>(i) +
                                    Is[2] * weights[2].at<float>(i) + Is[3] * weights[3].at<float>(i);

            for (int c = 0; c < channel; ++c) {
                result.at<float>(b, c, i) = curDescriptor.at<float>(c);
            }
        }
    }

    return result.reshape(1, {batch, channel, grid.size[1], grid.size[2]});
}
}  // namespace
