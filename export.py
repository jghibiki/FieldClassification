from __future__ import print_function
import sys
import time
from datetime import datetime
import signal
import tensorflow as tf
from PIL import Image
import numpy as np
import calculate_labels

# set random seed
np.random.seed(1)

NUM_IMAGES = 11807

def getImage(base, i):
    image_r = Image.open("%s/IMG-R-%08d.png" % (base, i))
    image_g = Image.open("%s/IMG-G-%08d.png" % (base, i))
    image_b = Image.open("%s/IMG-B-%08d.png" % (base, i))
    image_a = Image.open("%s/IMG-A-%08d.png" % (base, i))
    image = np.array([
        np.array(image_r)[..., np.newaxis],
        np.array(image_g)[..., np.newaxis],
        np.array(image_b)[..., np.newaxis],
        np.array(image_a)[..., np.newaxis]
    ])
    image = np.concatenate(image, axis=-1)
    return image

def getLabel(base, i):
    labels = Image.open("%s/LBL-%08d.png" % (base, i))

    labels = np.asarray(labels)

    simplified_labels = [ [ calculate_labels.lookup[pixel] for pixel in y ] for y in labels ]
    simplified_labels = np.asarray(simplified_labels, np.uint8)

    return simplified_labels

def getExample(base, i):
    image = getImage(base, i)
    label = getLabel(base, i)
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

    image_list = np.arange(NUM_IMAGES)
    np.random.shuffle(image_list)
    test_size = int(image_list.shape[0]*0.1)
    test = image_list[:test_size]
    train = image_list[test_size:]

    print("Calculated Partitions: train: {0}, test: {1}".format(train.shape[0], test.shape[0]))

    print("Exporting Training Data")

    filename = "data/train.tfrecord"
    writer = tf.python_io.TFRecordWriter(filename)

    start = datetime.now()

    for i in range(train.shape[0]):
        print("\rConverting: %08d image #%08d" % (i, train[i]), end="")
        sys.stdout.flush()

        example = getExample("raw_images", i)
        writer.write(example.SerializeToString())

        print("\rCompleted: %08d" % (i), end="")
        sys.stdout.flush()
    writer.close()
    print()

    print("Exporting Training Data")

    filename = "data/test.tfrecord"
    writer = tf.python_io.TFRecordWriter(filename)

    start = datetime.now()

    for i in range(test.shape[0]):
        print("\rConverting: %08d image #%08d" % (i, test[i]), end="")
        sys.stdout.flush()

        example = getExample("raw_images", i)
        writer.write(example.SerializeToString())

        print("\rCompleted: %08d" % (i), end="")
        sys.stdout.flush()
    writer.close()
    print()
