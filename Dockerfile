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

ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

WORKDIR /LinBERT
COPY . /LinBERT

RUN python -m pip install --upgrade pip
RUN pip install -r requirements.txt

RUN python setup.py build_ext --inplace
RUN pip install -e .
RUN sh scripts/install_pos_bias.sh