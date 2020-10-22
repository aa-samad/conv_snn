#!/bin/bash

CURRENT_PATH=$(pwd)
IMAGE_NAME="denden047/conv_snn"

docker build -t ${IMAGE_NAME} "$CURRENT_PATH"/docker && \
docker run -it --rm \
    --gpus 0 \
    -v "$CURRENT_PATH":/workdir \
    -w /workdir \
    ${IMAGE_NAME} \
    /bin/bash
