sudo nvidia-docker run -it -p 6006:6006 -v /home/jgoetze/FieldClassification:/mnt field-classification \
    tensorboard --logdir $1
