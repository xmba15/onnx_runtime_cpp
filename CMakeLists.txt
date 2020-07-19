cmake_minimum_required(VERSION 3.10)

project(ort_utility)
set (CMAKE_CXX_STANDARD 17)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

set(onnxruntime_INSTALL_PREFIX /usr/local)
set(onnxruntime_INCLUDE_DIRS
  ${onnxruntime_INSTALL_PREFIX}/include/onnxruntime
  ${onnxruntime_INSTALL_PREFIX}/include/onnxruntime/core/session
)

find_library(onnxruntime_LIBS NAMES onnxruntime PATHS /usr/local/lib)

find_package(OpenMP QUIET)

add_compile_options(
  "$<$<CONFIG:DEBUG>:-DDEBUG>"
)

add_subdirectory(src)

if(BUILD_APPS)
  add_subdirectory(examples)
endif(BUILD_APPS)
