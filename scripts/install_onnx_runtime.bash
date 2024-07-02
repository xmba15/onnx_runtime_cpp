#!/usr/bin/env bash

# reference: https://github.com/microsoft/onnxruntime#installation

readonly CURRENT_DIR=$(dirname $(realpath $0))

sudo -l

sudo apt-get update
sudo apt install -y --no-install-recommends zlib1g-dev

readonly ONNXRUNTIME_VERSION="v1.14.1"
# git clone --recursive -b ${ONNXRUNTIME_VERSION} https://github.com/Microsoft/onnxruntime
cd onnxruntime

INSTALL_PREFIX="/usr/local"
BUILDTYPE=Release
BUILDARGS="--config ${BUILDTYPE}"
BUILDARGS="${BUILDARGS} --build_shared_lib --skip_tests"
BUILDARGS="${BUILDARGS} --parallel"
BUILDARGS="${BUILDARGS} --cmake_extra_defines CMAKE_INSTALL_PREFIX=${INSTALL_PREFIX}"

source $CURRENT_DIR/get_cuda_environment_variables.bash
if [ ! -z "$CUDA_HOME" -a ! -z "$CUDA_VERSION" -a ! -z "$CUDNN_HOME" ]; then
    BUILDARGS="${BUILDARGS} --use_cuda --cuda_version=${CUDA_VERSION} --cuda_home=${CUDA_HOME} --cudnn_home=${CUDNN_HOME}"
fi

source $CURRENT_DIR/get_tensorrt_environment_variables.bash
if [ ! -z "$TENSORRT_HOME" ]; then
    # onnxruntime v1.10.0 is compatible with tensorrt 8
    BUILDARGS="${BUILDARGS} --use_tensorrt --tensorrt_home=${TENSORRT_HOME}"
fi

./build.sh ${BUILDARGS}
cd ./build/Linux/${BUILDTYPE}
sudo make install
