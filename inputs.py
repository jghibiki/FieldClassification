import tensorflow as tf
import numpy as np


def read_and_decode(file_name_queue):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(file_name_queue)
    features = tf.parse_single_example(
            serialized_example,
            features={
                "image_raw": tf.FixedLenFeature([], tf.string),
                "label_raw": tf.FixedLenFeature([], tf.string)
            })

    image = tf.decode_raw(features['image_raw'], tf.uint8)
    image = tf.reshape(image, [128, 128, 4])
    image.set_shape([128, 128, 4])
    image = tf.cast(image, tf.float32)
    #image = tf.cast(image, tf.float32) * (1. / 255) - 0.5



    label = tf.decode_raw(features['label_raw'], tf.uint8)
    label = tf.reshape(label, [4, 4])
    label.set_shape([4, 4])
    label = tf.cast(label, tf.int64)


    return image, label



def input_pipeline(file_name, batch_size, num_epochs=None):
    file_name_queue = tf.train.string_input_producer([file_name],
            num_epochs=num_epochs,
            shuffle=True)

    example, label = read_and_decode(file_name_queue)

    min_after_dequeue = 10000
    capacity = min_after_dequeue + 3 * batch_size

    examples, labels= tf.train.shuffle_batch(
            [example, label],
            batch_size=batch_size,
            capacity=capacity,
            min_after_dequeue=min_after_dequeue)

    return examples, labels




