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
                                 const std::vector<uint64_t>& classIndices, const std::vector<cv::Scalar> allColors,
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
                      cv::Point(curBbox[0] + labelSize.width, curBbox[1] + int(1.3 * labelSize.height)), curColor, -1);
        cv::putText(result, curLabel, cv::Point(curBbox[0], curBbox[1] + labelSize.height), cv::FONT_HERSHEY_COMPLEX,
                    0.35, cv::Scalar(255, 255, 255));
    }

    return result;
}
}  // namespace
