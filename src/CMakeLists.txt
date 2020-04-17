cmake_minimum_required(VERSION 3.10)

set(LIBRARY_NAME ${PROJECT_NAME})

file(GLOB SOURCE_FILES
  ${PROJECT_SOURCE_DIR}/src/ImageClassificationOrtSessionHandler.cpp
  ${PROJECT_SOURCE_DIR}/src/ImageRecognitionOrtSessionHandlerBase.cpp
  ${PROJECT_SOURCE_DIR}/src/ObjectDetectionOrtSessionHandler.cpp
  ${PROJECT_SOURCE_DIR}/src/OrtSessionHandler.cpp
)

add_library(${LIBRARY_NAME}
  SHARED
     ${SOURCE_FILES}
)

target_include_directories(${LIBRARY_NAME}
  PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
  PRIVATE
    ${onnxruntime_INCLUDE_DIRS}
)

target_compile_features(${LIBRARY_NAME}
  PRIVATE
     cxx_std_17
)

target_compile_options(${LIBRARY_NAME}
  PRIVATE
     $<$<CONFIG:Debug>:-O0 -g -Wall -Werror>
     $<$<CONFIG:Release>:-O3>
)

list(APPEND PRIVATE_LIBS ${onnxruntime_LIBS})

if (OpenMP_CXX_FOUND)
  list(APPEND PRIVATE_LIBS OpenMP::OpenMP_CXX)
endif()

target_link_libraries(${LIBRARY_NAME}
  PRIVATE
    ${PRIVATE_LIBS}
)
