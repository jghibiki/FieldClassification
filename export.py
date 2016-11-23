from __future__ import print_function
import sys
import time
from datetime import datetime
import signal
import tensorflow as tf
from PIL import Image
import numpy as np


def getImage(i):
    image_r = Image.open("raw_images/IMG-R-%08d.png" % i)
    image_g = Image.open("raw_images/IMG-G-%08d.png" % i)
    image_b = Image.open("raw_images/IMG-B-%08d.png" % i)
    image_a = Image.open("raw_images/IMG-A-%08d.png" % i)
    image = np.array([
        np.array(image_r)[..., np.newaxis],
        np.array(image_g)[..., np.newaxis],
        np.array(image_b)[..., np.newaxis],
        np.array(image_a)[..., np.newaxis]
    ])
    image = np.concatenate(image, axis=-1)
    return image

def getLabel(i):
    labels = Image.open("raw_images/LBL-%08d.png" % i)

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

    if(not image.shape[0] is 128 or not image.shape[1] is 128):
        print("bad image")
        print(image.shape)
        exit()
    if(not label.shape[0] is 128 or not label.shape[1] is 128):
        print("bad label")
        print(label.shape)
        exit()


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

    for i in xrange(8963):
        example = getExample(i)
        writer.write(example.SerializeToString())

        if i % 1 is 0:

            print("\rCompleted: %08d" % (i), end="")
            sys.stdout.flush()
    print()

