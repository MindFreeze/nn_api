ARG BUILD_FROM
FROM $BUILD_FROM

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

WORKDIR /app
USER root

COPY ./requirements.txt /app/requirements.txt

RUN apk update && \
    apk add --no-cache python3-dev \
        # Numpy dependencies
        make automake gcc g++ \
        # Pillow dependencies
        jpeg-dev zlib-dev && \
    pip3 install --upgrade pip && \
    pip3 install torch==1.4.0+cpu torchvision==0.5.0+cpu -f https://download.pytorch.org/whl/torch_stable.html && \
    pip3 install -r requirements.txt && \
    apk del automake gcc g++

COPY . /app

CMD [ "make", "run" ]
