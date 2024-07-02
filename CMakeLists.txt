cmake_minimum_required(VERSION 3.10)

project(ort_utility VERSION 0.0.2)
set (CMAKE_CXX_STANDARD 17)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

set(onnxruntime_INSTALL_PREFIX /home/whoami/Documents/Hanvon/onnx_runtime_cpp/onnxruntime)
set(onnxruntime_INCLUDE_DIRS
  ${onnxruntime_INSTALL_PREFIX}/include/onnxruntime
  ${onnxruntime_INSTALL_PREFIX}/include/onnxruntime/core/session
)


find_package(CUDA QUIET)

if(CUDA_FOUND AND USE_GPU)
  add_definitions(-DENABLE_GPU=1)
  include(tensorrt)
  if (TENSORRT_FOUND)
    add_definitions(-DENABLE_TENSORRT=1)
  endif()
else()
  add_definitions(-DENABLE_GPU=0)
  add_definitions(-DENABLE_TENSORRT=0)
endif()


add_compile_options(
  "$<$<CONFIG:Debug>:-DENABLE_DEBUG=1>"
  "$<$<CONFIG:Release>:-DENABLE_DEBUG=0>"
)

add_subdirectory(src)

if(BUILD_EXAMPLES)
  add_subdirectory(examples)
endif(BUILD_EXAMPLES)
