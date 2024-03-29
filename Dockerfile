# ARG BUILD_FROM
# FROM $BUILD_FROM
# FROM mindfreeze/pytorch-opencv-armhf
FROM bitnami/pytorch:1.3.0

# SHELL ["/bin/bash", "-o", "pipefail", "-c"]

WORKDIR /app
USER root

COPY ./pip.conf /etc/pip.conf
COPY ./requirements.txt /app/requirements.txt
# COPY ./torchvision-0.4.0a0+d31eafa-cp37-cp37m-linux_armv7l.whl /app/torchvision-0.4.0a0+d31eafa-cp37-cp37m-linux_armv7l.whl

RUN pip install --upgrade pip && \
    # sed -i s/torch==1.3.0/torch==1.3.0a0+a7de545/g requirements.txt && \
    # sed -i s/torchvision==0.2.2/torchvision==0.2.2.post3/g requirements.txt && \
    pip install -r requirements.txt
    # apt-get remove automake gcc g++

COPY . /app

# CMD [ "make", "run" ]
CMD gunicorn -b 0.0.0.0:5124 --workers=2 --worker-class=gthread app:app
