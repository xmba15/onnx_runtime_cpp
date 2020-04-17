/**
 * @file    PrimitiveTest.cpp
 *
 * @author  btran
 *
 */

/**
 *   @brief primitive app to test size and name of input/output tensors
 *   DO BUILD with DEBUG mode to see the debug messages
 */

#include <iostream>

#include <ort_utility/ort_utility.hpp>

namespace
{
static constexpr int64_t DUMMY_NUM_CLASSES = 2021;
}  // namespace

int main(int argc, char* argv[])
{
    if (argc != 2) {
        std::cerr << "Usage: [apps] [path/to/onnx/model]" << std::endl;
        return EXIT_FAILURE;
    }

    const std::string ONNX_MODEL_PATH = argv[1];
    Ort::ObjectDetectionOrtSessionHandler osh(DUMMY_NUM_CLASSES, ONNX_MODEL_PATH, 0);

    return EXIT_SUCCESS;
}
