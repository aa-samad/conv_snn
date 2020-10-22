FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-runtime

RUN apt-get update
RUN pip install --upgrade pip
RUN pip install --upgrade setuptools

# OpenCV
RUN pip install opencv-python
RUN pip install opencv-contrib-python

# pip
WORKDIR /tmp
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

# clearn
RUN rm -rf /var/lib/apt/lists/*
