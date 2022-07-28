FROM ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /build
COPY ./scripts .

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        sudo \
        gnupg2 \
        lsb-release \
        build-essential \
        software-properties-common \
        cmake \
        git \
        tmux && \
    bash install_latest_cmake.bash && \
    bash install_onnx_runtime.bash && \
    bash install_apps_dependencies.bash && \
    rm -rf /build && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

ENTRYPOINT ["/bin/bash"]
