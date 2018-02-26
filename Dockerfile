FROM tensorflow/tensorflow:latest-gpu

RUN apt-get update; apt-get -y install python-tk libjpeg-dev python-gdal
RUN pip install pillow

ENV LD_LIBRARY_PATH /opt/cuda/extra/extras/CUPTI/lib64:$LD_LIBRARY_PATH

WORKDIR /mnt
CMD python train.py
