import tensorflow as tf
import numpy as np


def read_and_decode(file_name_queue, image_size):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(file_name_queue)
    features = tf.parse_single_example(
            serialized_example,
            features={
                "image_raw": tf.FixedLenFeature([], tf.string),
                "label_raw": tf.FixedLenFeature([], tf.string)
            })

    image = tf.decode_raw(features['image_raw'], tf.uint8)
    image = tf.reshape(image, [image_size, image_size, 4])
    image.set_shape([image_size, image_size, 4])
    image = tf.cast(image, tf.float32)
    #image = tf.cast(image, tf.float32) * (1. / 255) - 0.5



    label = tf.decode_raw(features['label_raw'], tf.uint8)
    label = tf.reshape(label, [image_size, image_size])
    label.set_shape([image_size, image_size])
    label = tf.cast(label, tf.int64)


    return image, label



def train_pipeline(file_name, image_size, batch_size, num_epochs=None):
    file_name_queue = tf.train.string_input_producer([file_name],
            num_epochs=num_epochs,
            shuffle=True)

    example, label = read_and_decode(file_name_queue, image_size)

    min_after_dequeue = 10000
    capacity = min_after_dequeue + 3 * batch_size

    examples, labels= tf.train.shuffle_batch(
            [example, label],
            batch_size=batch_size,
            capacity=capacity,
            min_after_dequeue=min_after_dequeue)

    return examples, labels



