FROM nvidia/cuda:10.2-cudnn7-runtime-ubuntu18.04

RUN apt update \
    && apt install -y htop python3-dev wget

RUN /bin/bash -c "cd LinBERT \
    && pip install -r requirements.txt \
    && python setup.py build_ext --inplace \
    && pip install -e ."