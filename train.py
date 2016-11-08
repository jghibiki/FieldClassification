from __future__ import print_function
import tensorflow as tf
import sys
from model import ImageClassifier
from datetime import datetime
import inputs

sess = tf.InteractiveSession()
batch_size = 5
x, y = inputs.input_pipeline("data/train.tfrecord", batch_size, num_epochs=5)
classifier_model = ImageClassifier(x, y, batch_size=batch_size)

#sess = tf.Session()

train_writer = tf.train.SummaryWriter('output/train', sess.graph)

coord = tf.train.Coordinator()

with sess.as_default():
    sess.run(tf.initialize_all_variables())
    sess.run(tf.initialize_local_variables())

    threads = tf.train.start_queue_runners(coord=coord)

    try:
        step = 0
        start = datetime.now()
        while not coord.should_stop():

            accuracy, loss, summary, run_metadata = classifier_model.train(sess)

            if step % 1 is 0:
                train_writer.add_run_metadata(run_metadata, 'step%d' % step)
                train_writer.add_summary(summary, step)

                now = datetime.now()
                elapsed = now - start
                average = elapsed / step if not step is 0 else 0
                print("Step %d, Accuracy %.6f, Loss %.6f, Average Time %s/step, Elapsed Time %s" % (step, accuracy*100, loss, average, elapsed))
                sys.stdout.flush()

            if step % 1 is 0:
                classifier_model.save(sess, global_step=step)

            step += 1

    except tf.errors.OutOfRangeError:
        print()
        print("Done training for 1 epochs, %d steps" % step)
    except:
        raise
    finally:
        coord.request_stop()

    coord.join(threads)



