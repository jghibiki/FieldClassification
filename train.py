from __future__ import print_function
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
import sys
from model import ImageClassifier
from datetime import datetime
import inputs
import numpy as np

tf.flags.DEFINE_integer("batch_size", 1, "The batch size to use while training (default: 1).")
tf.flags.DEFINE_integer("num_epochs", 5, "The number of epochs to train for (default:5).")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (defualt: 0.5).")

tf.flags.DEFINE_integer("checkpoint_every", 1000, "Checkpoint model after this many steps.")
tf.flags.DEFINE_integer("summary_every", 50, "Save training summary after this many steps.")
tf.flags.DEFINE_integer("report_every", 1, "Output the current steps training stats after this many steps.")

tf.flags.DEFINE_string("output_dir", "output/", "The name of the directory to save checkpoints and summaries to.")
tf.flags.DEFINE_string("summary_train_dir", "summaries/train/", "The name of the directory to save training summaries to")

FLAGS = tf.app.flags.FLAGS

NUM_CLASSES = 15
IMAGE_SIZE = 256

def main(argv=None):

    print("Parameters:")
    for k,v in FLAGS.__flags.items():
        print(k, "=", v)
    print()

    #sess = tf.InteractiveSession()
    input_generator = inputs.train_pipeline(batch_size=FLAGS.batch_size, num_epochs=FLAGS.num_epochs)

    classifier_model = ImageClassifier(NUM_CLASSES, IMAGE_SIZE, batch_size=FLAGS.batch_size)

    sess = tf.Session()

    summary_dir = FLAGS.output_dir + FLAGS.summary_train_dir
    train_writer = tf.train.SummaryWriter(summary_dir, sess.graph)

    coord = tf.train.Coordinator()


    with sess.as_default():
	sess.run(tf.global_variables_initializer())
	sess.run(tf.local_variables_initializer())

        step = 0
        start = datetime.now()
        for batch in input_generator:

            accuracy, loss, summary, run_metadata, embeddings = classifier_model.train(sess, batch)

            if step % FLAGS.report_every is 0:
                now = datetime.now()
                elapsed = now - start
                average = elapsed / step if not step is 0 else 0
                print("Step %08d, Accuracy %.6f, Loss %.6f, Average Time %s/step, Elapsed Time %s%s" % (step, accuracy*100, loss, average, elapsed, ", Created Summary" if step %FLAGS.summary_every is 0 else ""))
                sys.stdout.flush()

            if step % FLAGS.summary_every is 0:
                train_writer.add_run_metadata(run_metadata, 'step%d' % step)
                train_writer.add_summary(summary, step)

            if step % FLAGS.checkpoint_every is 0:
                classifier_model.save(sess, global_step=step)

            step += 1



if __name__ == "__main__":
    tf.app.run()
