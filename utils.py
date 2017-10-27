import math

import tensorflow as tf

def weight_variable(shape, name=None, initializer=None):
    if not initializer:
        initial = tf.truncated_normal(shape, stddev=0.1)
    else:
        stddev = math.sqrt(2. / (kl**2 * dl))
        initial = tf.truncated_normal_initializer(stddev=stddev)

        initial = tf.truncated_normal_initializer(shape, stddev=0.1)

    return tf.get_variable(name, initializer=initial)

def weight_variable_with_weight_decay(name, initializer, wd):

    var = tf.get_variable(name, initializer=initializer)

    if wd is not None:
        with tf.variable_scope("weigth_decay"):
            weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
        with tf.device("/cpu:0"):
            tf.summary.scalar('weight_decay/' + name, weight_decay)

    return var

def msra_initializer(shape, kl, dl):
    """
    kl for kernel size, dl for filter number
    """
    stddev = math.sqrt(2. / (kl**2 * dl))
    return tf.truncated_normal(shape, stddev=stddev)

def bias_variable(shape, name=None):
    initial = tf.constant(0.1, shape=shape)
    return tf.get_variable(name, initializer=initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def batch_norm(x):
    return tf.contrib.layers.batch_norm(x, is_training=True, center=False, updates_collections=None, fused=True)

def max_pool_2x2(x):
    return tf.nn.max_pool_with_argmax(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def variable_summaries(name, var):
    with tf.device("/cpu:0"):
        with tf.variable_scope('summaries'):
            mean = tf.reduce_mean(var)
            tf.summary.scalar('mean/' + name, mean)
            with tf.variable_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar('stddev/' + name, stddev)
            tf.summary.scalar('max/' + name, tf.reduce_max(var))
            tf.summary.scalar('min/' + name, tf.reduce_min(var))
            tf.summary.histogram(name, var)

def unravel_argmax(argmax, shape):
    output_list = []
    with tf.device("/cpu:0"):
        output_list.append(argmax // (shape[2] * shape[3]))
        output_list.append(argmax % (shape[2] * shape[3]) // shape[3])
        return tf.pack(output_list)



def unpool_layer2x2_batch(updates, mask, ksize=[1, 2, 2, 1]):
    with tf.device("/cpu:0"):
	input_shape = updates.get_shape().as_list()
	#  calculation new shape
	output_shape = (input_shape[0], input_shape[1] * ksize[1], input_shape[2] * ksize[2], input_shape[3])
	# calculation indices for batch, height, width and feature maps
	one_like_mask = tf.ones_like(mask)
	batch_range = tf.reshape(tf.range(output_shape[0], dtype=tf.int64), shape=[input_shape[0], 1, 1, 1])
	b = one_like_mask * batch_range
	y = mask // (output_shape[2] * output_shape[3])
	x = mask % (output_shape[2] * output_shape[3]) // output_shape[3]
	feature_range = tf.range(output_shape[3], dtype=tf.int64)
	f = one_like_mask * feature_range
	# transpose indices & reshape update values to one dimension
	updates_size = tf.size(updates)
	indices = tf.transpose(tf.reshape(tf.stack([b, y, x, f]), [4, updates_size]))
	values = tf.reshape(updates, [updates_size])
	ret = tf.scatter_nd(indices, values, output_shape)
	return ret
