from __future__ import print_function
import tensorflow as tf
import numpy as np
import sys
from model import ImageClassifier
from datetime import datetime
import sat4

x, y = sat4.input_pipeline("data/test.tfrecord", 50, num_epochs=1)
classifier_model = ImageClassifier(x, y, eval=True, checkpoint_file="output/model.ckpt-7000" )

sess = tf.Session()

train_writer = tf.train.SummaryWriter('output/eval', sess.graph)

coord = tf.train.Coordinator()

with sess.as_default():
    sess.run(tf.initialize_all_variables())
    sess.run(tf.initialize_local_variables())

    classifier_model.load(sess)

    threads = tf.train.start_queue_runners(coord=coord)


    try:
        step = 0
        correct = 0 # counts correct predictions
        total = 0 # counts total evaluated
        start = datetime.now()
        while not coord.should_stop():

            predictions = classifier_model.evaluate_once(sess)
            correct += np.sum(predictions)
            total += len(predictions)

            if step % 50 is 0:

                now = datetime.now()
                elapsed = now - start
                average = elapsed / step if not step is 0 else 0
                print("Step %d, Evaluation Accuracy %g, Average Time %s/step, Elapsed Time %s" % (step, (correct/float(total)), average, elapsed))
                sys.stdout.flush()

            step += 1
    except tf.errors.OutOfRangeError:
        print()
        print("Done evaluating, completed in %d steps" % step)
    finally:
        coord.request_stop()

    coord.join(threads)
