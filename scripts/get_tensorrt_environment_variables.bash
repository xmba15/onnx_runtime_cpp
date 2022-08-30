#!/usr/bin/env bash

export TENSORRT_HOME="$(dirname $(whereis libnvinfer |  awk '{print $2}'))"
