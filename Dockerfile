FROM tensorflow/tensorflow:latest-gpu

RUN apt-get update; apt-get -y install python-tk libjpeg-dev python-gdal
RUN pip install pillow

WORKDIR /mnt
CMD python train.py
