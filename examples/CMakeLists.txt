cmake_minimum_required(VERSION 3.10)

list(APPEND EXAMPLES
  TestImageClassification
  PrimitiveTest
)

include(cmake_utility)

find_package(OpenCV REQUIRED)

list(APPEND PUBLIC_LIBS
  ${PROJECT_NAME}
  ${OpenCV_LIBS}
)

list(APPEND PRIVATE_LIBS
)

foreach(EXAMPLE ${EXAMPLES})
  __build_example(
    ${EXAMPLE}
    "${PUBLIC_LIBS}"
    "${PRIVATE_LIBS}"
  )
  target_include_directories(${EXAMPLE}
    PUBLIC
      ${OpenCV_INCLUDE_DIRS}
  )
endforeach(EXAMPLE)

# ---------------------------------------------------------

add_executable(tiny_yolo_v2
  ${CMAKE_CURRENT_LIST_DIR}/TinyYolov2.cpp
  ${CMAKE_CURRENT_LIST_DIR}/TinyYolov2.hpp
  ${CMAKE_CURRENT_LIST_DIR}/TinyYolov2App.cpp
)

target_link_libraries(tiny_yolo_v2
  PUBLIC
    ${PROJECT_NAME}
    ${OpenCV_LIBS}
)

target_include_directories(tiny_yolo_v2
  PUBLIC
    ${OpenCV_INCLUDE_DIRS}
)

# ---------------------------------------------------------

add_executable(mask_rcnn
  ${CMAKE_CURRENT_LIST_DIR}/MaskRCNN.cpp
  ${CMAKE_CURRENT_LIST_DIR}/MaskRCNN.hpp
  ${CMAKE_CURRENT_LIST_DIR}/MaskRCNNApp.cpp
)

target_link_libraries(mask_rcnn
  PUBLIC
    ${PROJECT_NAME}
    ${OpenCV_LIBS}
)

target_include_directories(mask_rcnn
  PUBLIC
    ${OpenCV_INCLUDE_DIRS}
)

# ---------------------------------------------------------

add_executable(yolov3
  ${CMAKE_CURRENT_LIST_DIR}/Yolov3.cpp
  ${CMAKE_CURRENT_LIST_DIR}/Yolov3.hpp
  ${CMAKE_CURRENT_LIST_DIR}/Yolov3App.cpp
)

target_link_libraries(yolov3
  PUBLIC
    ${PROJECT_NAME}
    ${OpenCV_LIBS}
)

target_include_directories(yolov3
  PUBLIC
    ${OpenCV_INCLUDE_DIRS}
)

# ---------------------------------------------------------

add_executable(ultra_light_face_detector
  ${CMAKE_CURRENT_LIST_DIR}/UltraLightFastGenericFaceDetector.cpp
  ${CMAKE_CURRENT_LIST_DIR}/UltraLightFastGenericFaceDetector.hpp
  ${CMAKE_CURRENT_LIST_DIR}/UltraLightFastGenericFaceDetectorApp.cpp
)

target_link_libraries(ultra_light_face_detector
  PUBLIC
    ${PROJECT_NAME}
    ${OpenCV_LIBS}
)

target_include_directories(ultra_light_face_detector
  PUBLIC
    ${OpenCV_INCLUDE_DIRS}
)
