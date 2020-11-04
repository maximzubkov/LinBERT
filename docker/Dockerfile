FROM nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04

ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update \
    && apt-get install --no-install-recommends -y \
                       build-essential \
                       ca-certificates \
                       wget \
                       unzip \
                       cmake \
                       git \
                       python3-dev python3-pip python3-setuptools

RUN ln -sf $(which python3) /usr/bin/python \
    && ln -sf $(which pip3) /usr/bin/pip

ENV LD_LIBRARY_PATH /usr/local/lib:/usr/local/cuda/lib64/stubs:/usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
ENV PYTHONIOENCODING=utf-8

ENV NCCL_ROOT_DIR /usr/local/cuda
ENV TH_BINARY_BUILD 1
ENV TORCH_CUDA_ARCH_LIST "3.5;5.0+PTX;5.2;6.0;6.1;7.0;7.5"
ENV TORCH_NVCC_FLAGS "-Xfatbin -compress-all"
ENV DESIRED_CUDA 102

WORKDIR /LinBERT
COPY . /LinBERT

RUN python -m pip install --upgrade pip
RUN pip install -r requirements.txt

RUN cd LinBERT \
    && python setup.py build_ext --inplace \
    && pip install -e . --force