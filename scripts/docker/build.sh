#!/usr/bin/env bash

key_content=$(cat "$KEY_PATH")

sudo docker build --build-arg SSH_PRIVATE_KEY="${key_content}" -t linbert .