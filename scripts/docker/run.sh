#!/usr/bin/env bash

sudo docker run --gpus=all --ipc=host --uts=host \
                -v /root/LinBERT:/LinBERT \
                -i -t linbert /bin/bash