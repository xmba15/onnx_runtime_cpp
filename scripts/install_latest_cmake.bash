#!/usr/bin/env bash

readonly CMAKE_VERSION_TO_INSTALL="3.18.0"

function command_exists
{
    type "$1" &> /dev/null;
}

function version_greater_equal
{
    printf '%s\n%s\n' "$2" "$1" | sort -V -C
}

if command_exists cmake; then
    readonly CMAKE_VERSION=`cmake --version | head -n1 | cut -d" " -f3`
    if version_greater_equal "${CMAKE_VERSION}" "${CMAKE_VERSION_TO_INSTALL}"; then
        exit 0
    fi
fi

echo "Need cmake ${CMAKE_VERSION_TO_INSTALL} or above. Install now..."

sudo -l

sudo apt-get update
sudo apt-get -y purge cmake
sudo apt-get install -y --no-install-recommends wget \
    libssl-dev

cd /tmp
wget https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION_TO_INSTALL}/cmake-${CMAKE_VERSION_TO_INSTALL}.tar.gz
tar -zxvf cmake-${CMAKE_VERSION_TO_INSTALL}.tar.gz
cd cmake-${CMAKE_VERSION_TO_INSTALL}
./bootstrap
make -j`nproc`
sudo make install
