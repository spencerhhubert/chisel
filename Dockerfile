ARG UBUNTU_VERSION=20.04
ARG CUDA_VERSION=11.6.1
ARG BASE_IMAGE=nvidia/cuda:${CUDA_VERSION}-base-ubuntu${UBUNTU_VERSION}

FROM ${BASE_IMAGE} as base

ENV TORCH_VERSION=1.13.0
ENV CUDA_VERSION_SHORT=cu116
ENV PYTHON_VERSION=3.8

SHELL ["/bin/bash", "-c"]

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential ca-certificates ccache cmake curl git libjpeg-dev libgl1-mesa-glx \
        libegl1-mesa libxrandr2 libxrandr2 libxss1 libxcursor1 libxcomposite1 libasound2 \
        libxi6 libxtst6 wget libpng-dev p7zip-rar neovim tmux && \
    rm -rf /var/lib/apt/lists/*
RUN /usr/sbin/update-ccache-symlinks

RUN apt-get update && apt-get install -y python${PYTHON_VERSION} python3-pip
RUN ln -sf /usr/bin/python3.8 /usr/local/bin/python
RUN alias pip=pip3

RUN echo "alias v=nvim" >> /root/.bashrc

RUN pip install torch==${TORCH_VERSION}+${CUDA_VERSION_SHORT} torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/${CUDA_VERSION_SHORT}
RUN pip install pyg-lib torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-${TORCH_VERSION}+${CUDA_VERSION_SHORT}.html
RUN pip install torch-geometric
RUN pip install "git+https://github.com/facebookresearch/pytorch3d.git"
RUN pip install diffusers

RUN mkdir chisel
WORKDIR chisel
