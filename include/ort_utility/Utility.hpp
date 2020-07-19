/**
 * @file    Utility.hpp
 *
 * @author  btran
 *
 * Copyright (c) organization
 *
 */

#pragma once

#include <algorithm>
#include <cassert>
#include <cmath>
#include <deque>
#include <limits>
#include <memory>
#include <numeric>
#include <utility>
#include <vector>

#ifdef DEBUG
#define ENABLE_DEBUG 1
#include <iostream>
#else
#define ENABLE_DEBUG 0
#endif

template <typename T, template <typename, typename = std::allocator<T>> class Container>
std::ostream& operator<<(std::ostream& os, const Container<T>& container)
{
    using ContainerType = Container<T>;
    for (typename ContainerType::const_iterator it = container.begin(); it != container.end(); ++it) {
        os << *it << " ";
    }

    return os;
}

namespace
{
template <typename T> std::deque<size_t> sortIndexes(const std::vector<T>& v)
{
    std::deque<size_t> indices(v.size());
    std::iota(indices.begin(), indices.end(), 0);

    std::stable_sort(std::begin(indices), std::end(indices), [&v](size_t i1, size_t i2) { return v[i1] < v[i2]; });

    return indices;
}
}  // namespace

namespace Ort
{
#if ENABLE_DEBUG
#define DEBUG_LOG(...)                                                                                                 \
    {                                                                                                                  \
        char str[100];                                                                                                 \
        snprintf(str, sizeof(str), __VA_ARGS__);                                                                       \
        std::cout << "[" << __FILE__ << "][" << __FUNCTION__ << "][Line " << __LINE__ << "] >>> " << str << std::endl; \
    }
#else
#define DEBUG_LOG(...)
#endif

template <typename T, template <typename, typename = std::allocator<T>> class Container>
std::ostream& operator<<(std::ostream& os, const Container<T>& container)
{
    using ContainerType = Container<T>;
    for (typename ContainerType::const_iterator it = container.begin(); it != container.end(); ++it) {
        os << *it << " ";
    }

    return os;
}

inline void softmax(float* input, const size_t inputLen)
{
    const float maxVal = *std::max_element(input, input + inputLen);

    const float sum = std::accumulate(input, input + inputLen, 0.0,
                                      [&](float a, const float b) { return std::move(a) + expf(b - maxVal); });

    const float offset = maxVal + logf(sum);
    for (auto it = input; it != (input + inputLen); ++it) {
        *it = expf(*it - offset);
    }
}

inline float sigmoid(const float x)
{
    return 1.0 / (1.0 + expf(-x));
}

inline std::vector<uint64_t> nms(const std::vector<std::array<float, 4>>& bboxes,            //
                                 const std::vector<float>& scores,                           //
                                 const float overlapThresh = 0.45,                           //
                                 const uint64_t topK = std::numeric_limits<uint64_t>::max()  //
)
{
    assert(bboxes.size() > 0);
    uint64_t boxesLength = bboxes.size();
    const uint64_t realK = std::max(std::min(boxesLength, topK), static_cast<uint64_t>(1));

    std::vector<uint64_t> keepIndices;
    keepIndices.reserve(realK);

    std::deque<uint64_t> sortedIndices = ::sortIndexes(scores);

    // keep only topk bboxes
    for (uint64_t i = 0; i < boxesLength - realK; ++i) {
        sortedIndices.pop_front();
    }

    std::vector<float> areas;
    areas.reserve(boxesLength);
    std::transform(std::begin(bboxes), std::end(bboxes), std::back_inserter(areas),
                   [](const auto& elem) { return (elem[2] - elem[0]) * (elem[3] - elem[1]); });

    while (!sortedIndices.empty()) {
        uint64_t currentIdx = sortedIndices.back();
        keepIndices.emplace_back(currentIdx);

        if (sortedIndices.size() == 1) {
            break;
        }

        sortedIndices.pop_back();
        std::vector<float> ious;
        ious.reserve(sortedIndices.size());

        const auto& curBbox = bboxes[currentIdx];
        const float curArea = areas[currentIdx];

        std::deque<uint64_t> newSortedIndices;

        for (const uint64_t elem : sortedIndices) {
            const auto& bbox = bboxes[elem];
            float tmpXmin = std::max(curBbox[0], bbox[0]);
            float tmpYmin = std::max(curBbox[1], bbox[1]);
            float tmpXmax = std::min(curBbox[2], bbox[2]);
            float tmpYmax = std::min(curBbox[3], bbox[3]);

            float tmpW = std::max<float>(tmpXmax - tmpXmin, 0.0);
            float tmpH = std::max<float>(tmpYmax - tmpYmin, 0.0);

            const float intersection = tmpW * tmpH;
            const float tmpArea = areas[elem];
            const float unionArea = tmpArea + curArea - intersection;
            const float iou = intersection / unionArea;

            if (iou <= overlapThresh) {
                newSortedIndices.emplace_back(elem);
            }
        }

        sortedIndices = newSortedIndices;
    }

    return keepIndices;
}

inline std::vector<std::array<int, 3>> generateColorCharts(const uint16_t numClasses = 1000, const uint16_t seed = 255)
{
    std::srand(seed);
    std::vector<std::array<int, 3>> colors;
    colors.reserve(numClasses);
    for (uint16_t i = 0; i < numClasses; ++i) {
        colors.emplace_back(std::array<int, 3>{std::rand() % 255, std::rand() % 255, std::rand() % 255});
    }

    return colors;
}
}  // namespace Ort
