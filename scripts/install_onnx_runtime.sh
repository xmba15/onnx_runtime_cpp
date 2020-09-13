#!/usr/bin/env bash

# reference: https://github.com/microsoft/onnxruntime#installation

sudo -l env "PATH=$PATH"

readonly CURRENT_DIR=$(dirname $(realpath $0))

echo "-------------------------------------------------------------------------"
echo "checking dependencies..."
echo "-------------------------------------------------------------------------"

readonly CMAKE_LOWEST_VERSION="3.13"

function command_exists
{
    type "$1" &> /dev/null;
}

if ! command_exists cmake; then
    echo "need cmake ${CMAKE_LOWEST_VERSION} or above"
    bash ${CURRENT_DIR}/install_latest_cmake.sh
fi

if ! command_exists nvcc; then
    echo "need to install cuda package"
    exit
fi

readonly CMAKE_VERSION=`cmake --version | head -n1 | cut -d" " -f3`
readonly CUDA_VERSION=`nvcc --version | grep release | awk '{print $6}' | cut -c 2-4`
function version_greater_equal
{
    printf '%s\n%s\n' "$2" "$1" | sort -V -C
}

if ! version_greater_equal "${CMAKE_VERSION}" "${CMAKE_LOWEST_VERSION}"; then
    echo "need cmake ${CMAKE_LOWEST_VERSION} or above"
    bash ${CURRENT_DIR}/install_latest_cmake.sh
fi

readonly CUDA_HOME=/usr/local/cuda
readonly CUDNN_MAIN_VERSION=7

if dpkg -L libcudnn${CUDNN_MAIN_VERSION} &> /dev/null; then
    readonly CUDNN_HOME=`dirname $(dpkg -L libcudnn${CUDNN_MAIN_VERSION} | grep libcudnn.so | head -n1)`
elif ls ${CUDA_HOME}/include/ | grep cudnn &> /dev/null; then
    readonly CUDNN_HOME=${CUDA_HOME}
else
    echo "make sure you change CUDNN_MAIN_VERSION line in this script to the version in your system or need to install cudnn"
    exit
fi

sudo apt-get install -y libgomp1
sudo apt install zlib1g-dev
sudo apt-get install -y locales
sudo apt-get install -y language-pack-en
sudo locale-gen en_US.UTF-8
sudo update-locale LANG=en_US.UTF-8

echo "-------------------------------------------------------------------------"
echo "cloning onnxruntime and starting to build..."
echo "-------------------------------------------------------------------------"

readonly ONNXRUNTIME_VERSION="v1.3.1"

git clone --recursive https://github.com/Microsoft/onnxruntime
cd onnxruntime

readonly INSTALL_PREFIX="/usr/local"
BUILDTYPE=Release
BUILDARGS="--config ${BUILDTYPE}"
BUILDARGS="${BUILDARGS} --use_cuda --cuda_version=${CUDA_VERSION} --cuda_home=${CUDA_HOME} --cudnn_home=${CUDNN_HOME}"
BUILDARGS="${BUILDARGS} --parallel"
BUILDARGS="${BUILDARGS} --update"
BUILDARGS="${BUILDARGS} --use_openmp"
# BUILDARGS="${BUILDARGS} --use_tensorrt --tensorrt_home /usr/src/tensorrt/"
BUILDARGS="${BUILDARGS} --cmake_extra_defines CMAKE_INSTALL_PREFIX=${INSTALL_PREFIX}"

# options to preserve environment path of current user
sudo env "PATH=$PATH" ./build.sh ${BUILDARGS} --build
sudo env "PATH=$PATH" ./build.sh ${BUILDARGS} --build_shared_lib

cd ./build/Linux/${BUILDTYPE}
sudo make install
echo "uninstall with cat install_manifest.txt | sudo xargs rm"
