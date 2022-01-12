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

cv::Mat drawColorChart(const std::vector<std::string>& classes, const std::vector<cv::Scalar>& colors)
{
    cv::Mat legend = cv::Mat::zeros((classes.size() * 25) + 25, 300, CV_8UC(3));
    for (std::size_t i = 0; i < classes.size(); ++i) {
        cv::putText(legend, classes[i], cv::Point(5, (i * 25) + 17), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                    cv::Scalar(0, 0, 255), 2);
        cv::rectangle(legend, cv::Point(100, (i * 25)), cv::Point(300, (i * 25) + 25), colors[i], -1);
    }

    return legend;
}
}  // namespace
