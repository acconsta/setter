FROM tensorflow/tensorflow:2.8.0-gpu

ARG DEBIAN_FRONTEND=noninteractive

# Install apt dependencies
RUN apt-get update && apt-get install -y \
    git \
    gpg-agent \
    python3-cairocffi \
    protobuf-compiler \
    python3-pil \
    python3-lxml \
    python3-tk \
    wget

# Install gcloud and gsutil commands
RUN curl -O https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-sdk-374.0.0-linux-x86_64.tar.gz && \
    tar -xf google-cloud-sdk-374.0.0-linux-x86_64.tar.gz && \
    ./google-cloud-sdk/install.sh

# Add new user to avoid running as root
RUN useradd -ms /bin/bash tensorflow
USER tensorflow
WORKDIR /home/tensorflow

# Copy this version of of the model garden into the image
COPY --chown=tensorflow models /home/tensorflow/models

# Download model checkpoint
ENV MODEL_NAME="ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8"
RUN curl -O http://download.tensorflow.org/models/object_detection/tf2/20200711/${MODEL_NAME}.tar.gz && \
    tar -xf ${MODEL_NAME}.tar.gz && \
    rm -rf models/research/object_detection/test_data/checkpoint && \
    mv ${MODEL_NAME}/checkpoint models/research/object_detection/test_data/

# Compile protobuf configs
RUN (cd /home/tensorflow/models/research/ && protoc object_detection/protos/*.proto --python_out=.)
WORKDIR /home/tensorflow/models/research/

RUN cp object_detection/packages/tf2/setup.py ./
ENV PATH="/home/tensorflow/.local/bin:${PATH}"

RUN python -m pip install -U pip
RUN python -m pip install . tensorflow==2.8 notebook

ENV TF_CPP_MIN_LOG_LEVEL 3

ENTRYPOINT python -m jupyter notebook --ip 0.0.0.0