from __future__ import print_function
import sys
import time
from datetime import datetime
import signal
import tensorflow as tf
from PIL import Image
import numpy as np


def getImage(i):
    image = Image.open("raw_images/IMG-%s.png" % i)
    return image

def getLabel(i):
    labels = Image.open("raw_images/LBL-%s.png" % i)

    return labels

def getExample(i):
    image = getImage(i)
    label = getLabel(i)
    example = convert_to(image, label)
    return example

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def convert_to(image, label):

    image = np.asarray(image)
    label = np.asarray(label)

    image_raw = image.tostring()
    label_raw = label.tostring()

    example = tf.train.Example(features=tf.train.Features(feature={
        'label_raw': _bytes_feature(label_raw),
        'image_raw': _bytes_feature(image_raw)
    }))
    return example


if __name__ == '__main__':


    print("Exporting Training Data")

    filename = "data/train.tfrecord"
    writer = tf.python_io.TFRecordWriter(filename)

    start = datetime.now()

    for i in xrange(5001):
        example = getExample(i)
        writer.write(example.SerializeToString())

        if i % 1 is 0:

            print("\rCompleted: %06d" % (i), end="")
            sys.stdout.flush()
    print()

