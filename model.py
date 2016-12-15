import tensorflow as tf
import math

def weight_variable(shape, initializer=None):
    if not initializer:
        initial = tf.truncated_normal(shape, stddev=0.1)
    else:
        stddev = math.sqrt(2. / (kl**2 * dl))
        initial = tf.truncated_normal_initializer(stddev=stddev)

        initial = tf.truncated_normal_initializer(shape, stddev=0.1)

    return tf.Variable(initial)

def weight_variavle_with_weight_decay(name, shape, initializer, wd):
    var = tf.get_variable(name, shape, initializer=initializer)

    if wd is not None:
        weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.scalar_summary('weight_decay/' + name, weight_decay)

    return var

def msra_initializer(kl, dl):
    """
    kl for kernel size, dl for filter number
    """
    stddev = math.sqrt(2. / (kl**2 * dl))
    return tf.truncated_normal_initializer(stddev=stddev)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def batch_norm(x):
    return tf.contrib.layers.batch_norm(x, is_training=True, center=False, updates_collections=None)

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def variable_summaries(var, name):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.scalar_summary('mean/' + name, mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.scalar_summary('stddev/' + name, stddev)
        tf.scalar_summary('max/' + name, tf.reduce_max(var))
        tf.scalar_summary('min/' + name, tf.reduce_min(var))
        tf.histogram_summary(name, var)

class ImageClassifier:

    def __init__(self, images, labels, num_classes, image_size, batch_size=50, num_epochs=500, dropout_rate=0.5, eval=False, checkpoint_file="output/model.ckpt"):
        self.image_size = image_size
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.dropout_rate = dropout_rate
        self.checkpoint_file = checkpoint_file
        self.eval = eval

        # input and label summaries
        self.image_image = tf.slice(images, [0, 0, 0, 0], [-1, self.image_size, self.image_size, 3])
        tf.image_summary('input',
                # slice removes nir layer which is stored as alpha
                self.image_image,
                max_images=50)
        self.label_image = scaled_label = ((tf.cast(labels, tf.float32)/self.num_classes)*255)
        tf.image_summary('label',
                tf.reshape(scaled_label, [-1, self.image_size, self.image_size, 1]),
                max_images=50)

        if not eval:
            self.loss(
                self.inference(images),
                labels
            )
        else:
            self.evaluate(images, labels)

        self.saver = tf.train.Saver()
        self.run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        self.run_metadata = tf.RunMetadata()

        self.merged = tf.merge_all_summaries()


    def inference(self, x):

        self.keep_prob = tf.placeholder(tf.float32)
        lrn = self.local_response_normalization_layer(x)
        conv1 = self.conv1_layer(lrn)
        conv2 = self.conv2_layer(conv1)
        conv3 = self.conv3_layer(conv2)
        conv4 = self.conv4_layer(conv3)

        deconv4 = self.deconv4_layer(conv4)
        conv4_decode = self.conv_decode4_layer(deconv4)

        deconv3 = self.deconv3_layer(conv4_decode)
        conv3_decode = self.conv_decode3_layer(deconv3)

        deconv2 = self.deconv2_layer(conv3_decode)
        conv2_decode = self.conv_decode2_layer(deconv2)

        deconv1 = self.deconv1_layer(conv2_decode)
        conv1_decode = self.conv_decode1_layer(deconv1)

        conv_class = self.conv_class_layer(conv1_decode)


        return conv_class

    def loss(self, logits, labels):
        labels = tf.reshape(labels, [self.batch_size, self.image_size * self.image_size])

        logits, labels, loss = self.calculate_loss(logits, labels)
        self.optimize_loss(loss)
        prediction = self.predict(logits, labels)
        self.calculate_accuracy(prediction)


    def evaluate(self, images, labels):

        # build a graph that computes predictions from the inference model
        logits = self.inference(images)

        img = tf.argmax(logits, 3)
        img = tf.cast(img, tf.float32)
        self.class_image = tf.reshape(img, [-1, self.image_size, self.image_size, 1])
        tf.image_summary("classification_map", self.class_image, max_images=500)

        labels = tf.cast(labels, tf.int32)
        #labels = tf.argmax(labels, 1)

        logits = tf.reshape(logits, [-1, self.num_classes])
        labels = tf.reshape(labels, [-1])

        # calculate predictions
        self.top_k_op = tf.nn.in_top_k(logits, labels, 1)


    def local_response_normalization_layer(self, x):
        return tf.nn.local_response_normalization(x)

    def conv1_layer(self, x):

        with tf.variable_scope('conv1') as scope_conv:
            W_conv1 = weight_variable([7, 7, 4, 64])
            variable_summaries(W_conv1, "W_conv1")
            b_conv1 = bias_variable([64])
            variable_summaries(b_conv1, "b_conv1")

            x_image = tf.reshape(x, [self.batch_size, self.image_size, self.image_size, 4])

            h_conv1 = conv2d(x_image, W_conv1) + b_conv1
            h_batch_norm1 = batch_norm(h_conv1)
            h_relu1 = tf.nn.relu(h_batch_norm1)
            h_pool1 = max_pool_2x2(h_relu1)

            self.image_summary(h_relu1, 'conv1/filters')

            return h_pool1


    def conv2_layer(self, h_pool1):
        with tf.variable_scope('conv2') as scope_conv:
            W_conv2 = weight_variable([7, 7, 64, 64])
            variable_summaries(W_conv2, "W_conv2")
            b_conv2 = bias_variable([64])
            variable_summaries(b_conv2, "b_conv2")


            h_conv2 = conv2d(h_pool1, W_conv2) + b_conv2
            h_batch_norm2 = batch_norm(h_conv2)
            h_relu2 = tf.nn.relu(h_batch_norm2)
            h_pool2 = max_pool_2x2(h_relu2)
            h_dropout = tf.nn.dropout(h_pool2, self.keep_prob)

            self.image_summary(h_relu2, 'conv2/filters')

            return h_dropout


    def conv3_layer(self, h_pool1):
        with tf.variable_scope('conv3') as scope_conv:
            W_conv = weight_variable([7, 7, 64, 64])
            variable_summaries(W_conv, "W_conv3")
            b_conv = bias_variable([64])
            variable_summaries(b_conv, "b_conv3")


            h_conv = conv2d(h_pool1, W_conv) + b_conv
            h_batch_norm = batch_norm(h_conv)
            h_relu = tf.nn.relu(h_batch_norm)
            h_pool = max_pool_2x2(h_relu)
            h_dropout = tf.nn.dropout(h_pool, self.keep_prob)

            self.image_summary(h_relu, 'conv3/filters')

            return h_dropout


    def conv4_layer(self, h_pool1):
        with tf.variable_scope('conv4') as scope_conv:
            W_conv = weight_variable([7, 7, 64, 64])
            variable_summaries(W_conv, "W_conv4")
            b_conv = bias_variable([64])
            variable_summaries(b_conv, "b_conv4")


            h_conv = conv2d(h_pool1, W_conv) + b_conv
            h_batch_norm = batch_norm(h_conv)
            h_relu = tf.nn.relu(h_batch_norm)
            h_pool = max_pool_2x2(h_relu)
            h_dropout = tf.nn.dropout(h_pool, self.keep_prob)

            self.image_summary(h_relu, 'conv4/filters')

            return h_dropout

    def deconv4_layer(self, h_pool2):
        with tf.variable_scope('deconv4') as scope_conv:
            W_deconv = weight_variable([2, 2, 64, 64])
            variable_summaries(W_deconv, "W_deconv4")

            h_deconv = tf.nn.conv2d_transpose(h_pool2, W_deconv, [self.batch_size, self.image_size/8, self.image_size/8, 64], [1, 2, 2, 1])
            self.image_summary(h_deconv, 'deconv4/filters')

            return h_deconv

    def conv_decode4_layer(self, h_deconv1):
        with tf.variable_scope('conv_decode4') as scope_conv:
            W_conv = weight_variable([7, 7, 64, 64])
            variable_summaries(W_conv, "W_conv_decode4")
            b_conv = bias_variable([64])
            variable_summaries(b_conv, "b_conv_decode4")

            h_conv = tf.nn.conv2d(h_deconv1, W_conv, [1, 1, 1, 1], padding="SAME") + b_conv
            h_batch_norm = batch_norm(h_conv)
            h_relu = tf.nn.relu(h_batch_norm)
            h_dropout = tf.nn.dropout(h_relu, self.keep_prob)
            self.image_summary(h_relu, 'conv_decode4/filters')

            return h_dropout


    def deconv3_layer(self, h_pool2):
        with tf.variable_scope('deconv3') as scope_conv:
            W_deconv = weight_variable([2, 2, 64, 64])
            variable_summaries(W_deconv, "W_deconv3")

            h_deconv = tf.nn.conv2d_transpose(h_pool2, W_deconv, [self.batch_size, self.image_size/4, self.image_size/4, 64], [1, 2, 2, 1])
            self.image_summary(h_deconv, 'deconv3/filters')

            return h_deconv

    def conv_decode3_layer(self, h_deconv1):
        with tf.variable_scope('conv_decode3') as scope_conv:
            W_conv = weight_variable([7, 7, 64, 64])
            variable_summaries(W_conv, "W_conv_decode3")
            b_conv = bias_variable([64])
            variable_summaries(b_conv, "b_conv_decode3")

            h_conv = tf.nn.conv2d(h_deconv1, W_conv, [1, 1, 1, 1], padding="SAME") + b_conv
            h_batch_norm = batch_norm(h_conv)
            h_relu = tf.nn.relu(h_batch_norm)
            h_dropout = tf.nn.dropout(h_relu, self.keep_prob)
            self.image_summary(h_relu, 'conv_decode3/filters')

            return h_dropout

    def deconv2_layer(self, h_pool2):
        with tf.variable_scope('deconv2') as scope_conv:
            W_deconv2 = weight_variable([2, 2, 64, 64])
            variable_summaries(W_deconv2, "W_deconv2")

            h_deconv2 = tf.nn.conv2d_transpose(h_pool2, W_deconv2, [self.batch_size, self.image_size/2, self.image_size/2, 64], [1, 2, 2, 1])
            self.image_summary(h_deconv2, 'deconv2/filters')

            return h_deconv2

    def conv_decode2_layer(self, h_deconv1):
        with tf.variable_scope('conv_decode2') as scope_conv:
            W_conv = weight_variable([7, 7, 64, 64])
            variable_summaries(W_conv, "W_conv_decode2")
            b_conv = bias_variable([64])
            variable_summaries(b_conv, "b_conv_decode2")

            h_conv = tf.nn.conv2d(h_deconv1, W_conv, [1, 1, 1, 1], padding="SAME") + b_conv
            h_batch_norm = batch_norm(h_conv)
            h_relu = tf.nn.relu(h_batch_norm)
            h_dropout = tf.nn.dropout(h_relu, self.keep_prob)
            self.image_summary(h_relu, 'conv_decode2/filters')

            return h_dropout


    def deconv1_layer(self, h_pool2):
        with tf.variable_scope('deconv1') as scope_conv:
            W_deconv1 = weight_variable([2, 2, 64, 64])
            variable_summaries(W_deconv1, "W_deconv1")

            h_deconv1 = tf.nn.conv2d_transpose(h_pool2, W_deconv1, [self.batch_size, self.image_size, self.image_size, 64], [1, 2, 2, 1])
            self.image_summary(h_deconv1, 'deconv1/filters')

            return h_deconv1


    def conv_decode1_layer(self, h_deconv2):
        with tf.variable_scope('conv_decode1') as scope_conv:
            W_conv = weight_variable([5, 5, 64, 64])
            variable_summaries(W_conv, "W_conv_decode1")
            b_conv = bias_variable([64])
            variable_summaries(b_conv, "b_conv_decode1")

            h_conv = tf.nn.conv2d(h_deconv2, W_conv, [1, 1, 1, 1], padding="SAME") + b_conv
            h_batch_norm = batch_norm(h_conv)
            h_relu = tf.nn.relu(h_batch_norm)
            self.image_summary(h_relu, 'conv_decode1/filters')

            return h_conv

    def conv_class_layer(self, h_conv_decode1):
        with tf.variable_scope('conv_classification') as scope_conv:

            W_conv_class = weight_variavle_with_weight_decay(
                "W_conv_class",
                [1, 1, 64, self.num_classes],
                msra_initializer(1, 64),
                0.0005)
            variable_summaries(W_conv_class, "W_conv_classification")
            b_conv_class = bias_variable([self.num_classes])
            variable_summaries(b_conv_class, "b_conv_classification")
            h_conv_class = tf.nn.conv2d(h_conv_decode1, W_conv_class, [1, 1, 1, 1], padding="SAME") + b_conv_class
            self.image_summary(h_conv_class, 'conv_class_layer/filters')

            # combine conv_class filters into single classification image
            class_image = tf.argmax(tf.reshape(tf.round(h_conv_class), [-1, self.num_classes, self.image_size, self.image_size]), 1)
            #class_image = tf.reduce_max(max_indices, reduction_indices=[2], keep_dims=True)
            self.class_image = class_image

            class_image = tf.reshape(class_image, [-1, self.image_size, self.image_size, 1])
            class_image = tf.cast(class_image, tf.float32)

            self.image_summary(class_image, "class_image")

        return h_conv_class


    def calculate_loss(self, logits, labels):
        with tf.variable_scope('Loss') as scope_conv:

            logits = tf.reshape(logits, [-1, self.num_classes])
            labels = tf.reshape(labels, [-1])

            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits, labels)
            cross_entropy_mean = tf.reduce_mean(cross_entropy)
            tf.scalar_summary('loss', cross_entropy_mean)
            self.calculated_loss = cross_entropy_mean

            #logits = tf.reshape(logits, [-1, 255])
            #epsilon = tf.constant(value=1e-10)
            #logits  = logits + epsilon

            #labels_flat = tf.reshape(labels, (-1, 1))

            #labels = tf.reshape(tf.one_hot(labels_flat, depth=255), [-1, 255])

            #softmax = tf.nn.softmax(logits)

            #cross_entropy = - tf.reduce_sum((labels * tf.log(softmax + epsilon)), reduction_indices=[1])

            #cross_entropy_mean = tf.reduce_mean(cross_entropy)
            #self.calculated_loss = cross_entropy_mean

            #tf.add_to_collection('losses', cross_entropy_mean)

            #loss = tf.add_n(tf.get_collection('losses'))
            #tf.scalar_summary('loss', loss)
        return logits, labels, cross_entropy

        #    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(y_conv, y)
        #    loss = tf.reduce_mean(cross_entropy)
        #    tf.scalar_summary('loss', loss)
        #return cross_entropy

    def optimize_loss(self, cross_entropy):
        with tf.variable_scope('Optimization') as scope_conv:
            self.train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    def predict(self, y_conv, y):
        with tf.variable_scope('Prediction') as scope_conv:
            y = tf.reshape(y, [-1])
            correct_prediction = tf.equal(tf.argmax(y_conv, 1), y)
            tf.scalar_summary("correct_predictions", tf.reduce_sum(tf.cast(correct_prediction, tf.float32)))

            prediction_image = tf.reshape(tf.cast(correct_prediction, tf.float32), [-1, self.image_size, self.image_size, 1])
            self.image_summary(prediction_image, "prediction_image")

        return correct_prediction

    def calculate_accuracy(self, correct_prediction):
        # TODO: http://stackoverflow.com/questions/37746670/tensorflow-multi-label-accuracy-calculation
        with tf.variable_scope('Accuracy') as scope_conv:
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            tf.scalar_summary('accuracy', self.accuracy)

    def image_summary(self, h_conv, tag_name):

        h_conv_features = tf.unpack(h_conv, axis=3)
        h_conv_max = tf.reduce_max(h_conv)
        h_conv_features_padded = map(lambda t: tf.pad(t-h_conv_max, [[0,0], [0,1], [0,0]]) + h_conv_max, h_conv_features)
        h_conv_imgs = tf.expand_dims(tf.concat(1, h_conv_features_padded), -1)

        tf.image_summary(tag_name, h_conv_imgs, max_images=5)



    def train(self, sess, eval=False):
        if not eval:
            summary, accuracy, loss, _ = sess.run([self.merged, self.accuracy, self.calculated_loss,  self.train_step],
                    feed_dict={
                    self.keep_prob: self.dropout_rate
                },
                options=self.run_options,
                run_metadata=self.run_metadata)
            return accuracy, loss, summary, self.run_metadata
        if eval:
            summary, accuracy, loss,  _ = sess.run([self.merged, self.accuracy, self.calculated_loss,  self.train_step],
                    feed_dict={
                    self.keep_prob: 0
                },
                options=self.run_options,
                run_metadata=self.run_metadata)
            return accuracy, loss, summary, self.run_metadata

    def evaluate_once(self, sess):
        predictions, summary, image, label, class_img = sess.run([self.top_k_op, self.merged, self.image_image, self.label_image, self.class_image],
            feed_dict={
                self.keep_prob: 1
        })

        return predictions, summary, image, label, class_img

    def save(self, sess, global_step=None):
        return self.saver.save(sess, self.checkpoint_file, global_step=global_step)

    def load(self, sess):
        self.saver.restore(sess, self.checkpoint_file)




