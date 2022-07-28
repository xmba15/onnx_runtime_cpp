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
sudo apt-get -y --no-install-recommends install python3-pip
python3 -m pip install cmake==$CMAKE_VERSION_TO_INSTALL
exec bash
