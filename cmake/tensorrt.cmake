# Ref: //github.com/PRBonn/rangenet_lib/blob/master/cmake/tensorrt-config.cmake

find_package(CUDA)
find_library(NVINFER  NAMES nvinfer)
find_library(NVINFERPLUGIN NAMES nvinfer_plugin)
find_library(NVPARSERS NAMES nvparsers)
find_library(NVONNXPARSER NAMES nvonnxparser)

# newer tensorrt does not have nvonnxparser_runtime
# find_library(NVONNXPARSERRUNTIME NAMES nvonnxparser_runtime)

# If it is ALL there, export libraries as a single package
if(CUDA_FOUND AND NVINFER AND NVINFERPLUGIN AND NVPARSERS AND NVONNXPARSER)
  message("TensorRT available!")
  message("CUDA Libs: ${CUDA_LIBRARIES}")
  message("CUDA Headers: ${CUDA_INCLUDE_DIRS}")
  message("NVINFER: ${NVINFER}")
  message("NVINFERPLUGIN: ${NVINFERPLUGIN}")
  message("NVPARSERS: ${NVPARSERS}")
  message("NVONNXPARSER: ${NVONNXPARSER}")
  list(APPEND TENSORRT_LIBRARIES ${CUDA_LIBRARIES} nvinfer nvinfer_plugin nvparsers nvonnxparser)
  message("All togheter now (libs): ${TENSORRT_LIBRARIES}")
  list(APPEND TENSORRT_INCLUDE_DIRS ${CUDA_INCLUDE_DIRS})
  message("All togheter now (inc): ${TENSORRT_INCLUDE_DIRS}")
  set(TENSORRT_FOUND ON)
else()
  message("TensorRT NOT Available")
  set(TENSORRT_FOUND OFF)
endif()
