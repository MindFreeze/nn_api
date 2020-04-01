# ARG BUILD_FROM
# FROM $BUILD_FROM
FROM python:3.7-alpine

# SHELL ["/bin/bash", "-o", "pipefail", "-c"]

WORKDIR /app
USER root

COPY ./requirements.txt /app/requirements.txt

RUN apk update && \
    apk add --no-cache \
        # Numpy dependencies
        make automake gcc g++ \
        # Pillow dependencies
        jpeg-dev zlib-dev && \
    pip3 install --upgrade pip && \
    sed -t s/torch==1.4.0/torch==1.4.0a0+f43194e/g requirements.txt && \
    sed -t s/torchvision==0.5.0/torchvision==0.5.0a0+9cdc814/g requirements.txt && \
    pip3 install https://github.com/sungjuGit/PyTorch-and-Vision-for-Raspberry-Pi-4B/raw/master/torch-1.4.0a0%2Bf43194e-cp37-cp37m-linux_armv7l.whl && \
    pip3 install https://github.com/sungjuGit/PyTorch-and-Vision-for-Raspberry-Pi-4B/raw/master/torchvision-0.5.0a0%2B9cdc814-cp37-cp37m-linux_armv7l.whl && \
    # pip3 install torch==1.4.0+cpu torchvision==0.5.0+cpu -f https://download.pytorch.org/whl/torch_stable.html && \
    pip3 install -r requirements.txt && \
    apk del automake gcc g++

COPY . /app

CMD [ "make", "run" ]
