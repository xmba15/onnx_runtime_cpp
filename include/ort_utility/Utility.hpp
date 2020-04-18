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
#include <cmath>
#include <numeric>
#include <utility>

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
