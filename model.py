import tensorflow as tf

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

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

    def __init__(self, images, labels, batch_size=50, num_epochs=500, dropout_rate=0.5, eval=False, checkpoint_file="output/model.ckpt"):
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.dropout_rate = dropout_rate
        self.checkpoint_file = checkpoint_file
        self.eval = eval

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
        tf.image_summary('input', x, max_images=50)
        conv1 = self.conv1_layer(x)
        conv2 = self.conv2_layer(conv1)

        deconv2 = self.deconv2_layer(conv2)
        conv2_decode = self.conv_decode2_layer(deconv2)

        deconv1 = self.deconv1_layer(conv2_decode)
        conv1_decode = self.conv_decode1_layer(deconv1)

        conv_class = self.conv_class_layer(conv1_decode)
        return conv_class

    def loss(self, logits, labels):
        labels = tf.reshape(labels, [self.batch_size, 128*128])

        logits, labels, loss = self.calculate_loss(logits, labels)
        self.optimize_loss(loss)
        prediction = self.predict(logits, labels)
        self.calculate_accuracy(prediction)

    def evaluate(self, images, labels):

        # build a graph that computes predictions from the inference model
        logits = self.inference(images)

        labels = tf.cast(labels, tf.int32)
        labels = tf.argmax(labels, 1)


        # calculate predictions
        self.top_k_op = tf.nn.in_top_k(logits, labels, 1)



    def conv1_layer(self, x):

        with tf.variable_scope('conv1') as scope_conv:
            W_conv1 = weight_variable([5, 5, 4, 64])
            variable_summaries(W_conv1, "W_conv1")
            b_conv1 = bias_variable([64])
            variable_summaries(b_conv1, "b_conv1")

            x_image = tf.reshape(x, [self.batch_size, 128, 128, 4])

            h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
            h_pool1 = max_pool_2x2(h_conv1)

            self.image_summary(h_conv1, 'conv1/filters')

            return h_pool1


    def conv2_layer(self, h_pool1):
        with tf.variable_scope('conv2') as scope_conv:
            W_conv2 = weight_variable([5, 5, 64, 64])
            variable_summaries(W_conv2, "W_conv2")
            b_conv2 = bias_variable([64])
            variable_summaries(b_conv2, "b_conv2")


            h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
            print(h_conv2)
            h_pool2 = max_pool_2x2(h_conv2)
            print(h_pool2)

            self.image_summary(h_conv2, 'conv2/filters')

            return h_pool2

    def deconv2_layer(self, h_pool2):
        with tf.variable_scope('deconv2') as scope_conv:
            W_deconv2 = weight_variable([2, 2, 64, 64])
            variable_summaries(W_deconv2, "W_deconv2")

            h_deconv2 = tf.nn.conv2d_transpose(h_pool2, W_deconv2, [self.batch_size, 64, 64, 64], [1, 2, 2, 1])
            self.image_summary(h_deconv2, 'deconv2/filters')

            return h_deconv2

    def conv_decode2_layer(self, h_deconv1):
        with tf.variable_scope('conv_decode2') as scope_conv:
            W_conv_decode2 = weight_variable([5, 5, 64, 64])
            variable_summaries(W_conv_decode2, "W_conv_decode2")
            b_conv_decode2 = bias_variable([64])
            variable_summaries(b_conv_decode2, "b_conv_decode2")

            h_conv_decode2 = tf.nn.conv2d(h_deconv1, W_conv_decode2, [1, 1, 1, 1], padding="SAME") + b_conv_decode2
            self.image_summary(h_decode2, 'conv_decode2/filters')

            return h_conv_decode2


    def deconv1_layer(self, h_pool2):
        with tf.variable_scope('deconv1') as scope_conv:
            W_deconv1 = weight_variable([2, 2, 64, 64])
            variable_summaries(W_deconv1, "W_deconv1")

            h_deconv1 = tf.nn.conv2d_transpose(h_pool2, W_deconv1, [self.batch_size, 128, 128, 64], [1, 2, 2, 1])
            self.image_summary(h_deconv1, 'deconv1/filters')

            return h_deconv1


    def conv_decode1_layer(self, h_deconv2):
        with tf.variable_scope('conv_decode1') as scope_conv:
            W_conv_decode1 = weight_variable([5, 5, 64, 64])
            variable_summaries(W_conv_decode1, "W_conv_decode1")
            b_conv_decode1 = bias_variable([64])
            variable_summaries(b_conv_decode1, "b_conv_decode1")

            h_conv_decode1 = tf.nn.conv2d(h_deconv2, W_conv_decode1, [1, 1, 1, 1], padding="SAME") + b_conv_decode1
            self.image_summary(h_decode1, 'conv_decode1/filters')

            return h_conv_decode1

    def conv_class_layer(self, h_conv_decode1):
        with tf.variable_scope('conv_classification') as scope_conv:
            W_conv_class = weight_variable([1, 1, 64, 255])
            variable_summaries(W_conv_class, "W_conv_classification")
            b_conv_class = bias_variable([255])
            variable_summaries(b_conv_class, "b_conv_classification")

            h_conv_class = tf.nn.conv2d(h_conv_decode1, W_conv_class, [1, 1, 1, 1], padding="SAME") + b_conv_class
            self.image_summary(h_conv_class, 'conv_class_layer/filters')
            #h_pool_class = max_pool_2x2(h_conv_class)
            #h_pool_class = tf.reshape(h_pool_class, [self.batch_size, 64* 64, 255])

        return h_conv_class


    def calculate_loss(self, logits, labels):
        with tf.variable_scope('Loss') as scope_conv:

            logits = tf.reshape(logits, [-1, 255])
            epsilon = tf.constant(value=1e-10)
            logits  = logits + epsilon

            labels_flat = tf.reshape(labels, (-1, 1))

            labels = tf.reshape(tf.one_hot(labels_flat, depth=255), [-1, 255])

            softmax = tf.nn.softmax(logits)

            cross_entropy = - tf.reduce_sum((labels * tf.log(softmax + epsilon)), reduction_indices=[1])

            cross_entropy_mean = tf.reduce_mean(cross_entropy)
            self.calculated_loss = cross_entropy_mean

            tf.add_to_collection('losses', cross_entropy_mean)

            loss = tf.add_n(tf.get_collection('losses'))
            tf.scalar_summary('loss', loss)
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
            correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y, 1))
        return correct_prediction

    def calculate_accuracy(self, correct_prediction):
        with tf.variable_scope('Accuracy') as scope_conv:
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            tf.scalar_summary('accuracy', self.accuracy)

    def image_summary(self, h_conv, tag_name):

        h_conv_features = tf.unpack(h_conv, axis=3)
        h_conv_max = tf.reduce_max(h_conv)
        h_conv_features_padded = map(lambda t: tf.pad(t-h_conv_max, [[0,0], [0,1], [0,0]]) + h_conv_max, h_conv_features)
        h_conv_imgs = tf.expand_dims(tf.concat(1, h_conv_features_padded), -1)

        tf.image_summary(tag_name, h_conv_imgs, max_images=5)



    def train(self, sess):
        summary, accuracy, loss,  _ = sess.run([self.merged, self.accuracy, self.calculated_loss,  self.train_step],
                feed_dict={
                #self.keep_prob: self.dropout_rate
            },
            options=self.run_options,
            run_metadata=self.run_metadata)
        return accuracy, loss, summary, self.run_metadata

    def evaluate_once(self, sess):
        predictions = sess.run([self.top_k_op],
            feed_dict={
                #self.keep_prob: 1
        })

        return predictions

    def save(self, sess, global_step=None):
        return self.saver.save(sess, self.checkpoint_file, global_step=global_step)

    def load(self, sess):
        self.saver.restore(sess, self.checkpoint_file)




