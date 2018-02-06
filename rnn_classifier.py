
class ImageClassifier:

    def __init__(self, num_classes, time_steps, image_size, pixel_vector_size, batch_size=50, num_epochs=500, dropout_rate=0.5, eval=False, checkpoint_file="output/model.ckpt-1000-5000-2500"):
        self.image_size = image_size
        self.time_steps = time_steps
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.pixel_vector_size = pixel_vector_size
        self.num_epochs = num_epochs
        self.dropout_rate = dropout_rate
        self.checkpoint_file = checkpoint_file
        self.eval = eval

        self._build_model()


    def _build_model(self):

        with tf.device("/cpu:0"):
            with tf.variable_scope("conv_inputs"):
                self.conv_x = tf.placeholder(tf.float32, [self.time_steps, self.batch_size, image_size, image_size, 4], name="images")
                self.keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

            with tf.variable_scope("rnn_inputs"):
                self.x = tf.placeholder(tf.float32, [self.time_steps, self.batch_size, self.pixel_vector_size], name="inputs")
                self.y = tf.placeholder(tf.int64, [self.time_steps, self.batch_size, self.pixel_vector_size], name="labels")

        with tf.device("/gpu:0"):

            # initial conv layer to produce vector
            self.conv_out = self._conv_layer(0, 64, self.conv_x)


            # define rnn
            self.cell = tf.contrib.cudnn_rnn.CudnnLSTM(
                    num_layers=self.time_steps,
                    num_units=self.pixel_vector_size,
                    input_size=self.pixel_vector_size,
                    dropout=1 - self.keep_prob if not self.eval else 0)

            params_size_tensor = self.cell.params_size()
            self.rnn_params = self.get_variable("rnn_params",
                    initializer = tf.random_uniform([params_size_tensor], -1, 1),
                    validate_shape = False)

            c = tf.zeros([self.time_steps, self.batch_size, self.pixel_vector_size], tf.float32)
            h = tf.zeros([self.time_steps, self.batch_size, self.pixel_vector_size], tf.float32)

            self.initial_lstm_state = (tf.contrib.rnn.LSTMStateTuple(h=h, c=c), not self.eval)

            outputs, h, c = self.cell(self.x, h, c, self.rnn_params, not self.eval)

            outputs = tf.transpose(outputs, [1, 0, 2])
            outputs = tf.reshape(outputs, [-1, self.pixel_vector_size])

            state = (tf.contrib.rnn.LSTMStateTuple(h=h, c=c),)


            # define readout layer
            softmax_w = tf.get_variable("softmax_w", [self.pixel_vector_size, self.num_labels], dtype=tf.float32)
            softmax_b = tf.get_variable("softmax_b", [self.num_labels], dtype=tf.float32)
            logits = tf.nn.wx_plus_b(outputs, softmax_w, softmax_b)
            #logits = tf.reshape(logits, [self.batch_size, self.time_steps, self.num_labels])

            # define loss function
            cross_entropy = tf.nn.sparse_softmax_crossentropy_with_logits(logits=logits, labels=self.labels)
            self.loss = tf.reduce_mean(cross_entropy)


            # training operation
            self.update = tf.train.AdamOptimizer(1e-4).minimize(self.loss)


    def _conv_layer(self, layer_no, input_channels, x):

        with tf.variable_scope('conv%s' % layer_no) as scope_conv:
            W_conv = weight_variable([3, 3, input_channels, 64], "W_conv%s" % layer_no)
            variable_summaries("W-conv%s" % layer_no, W_conv)
            b_conv = bias_variable([64], "b_conv%s" % layer_no)
            variable_summaries("b-conv%s" % layer_no, b_conv)

            tf.add_to_collection("loss_vars", W_conv)
            tf.add_to_collection("loss_vars", b_conv)

            h_conv = conv2d(x, W_conv) + b_conv
            h_batch_norm = batch_norm(h_conv)
            h_relu = tf.nn.relu(h_batch_norm)

            self.image_summary('conv{}/filters'.format(layer_no), h_relu)
            self.image_summary('conv{}/weights'.format(layer_no), W_conv)

            return h_pool
