from __future__ import print_function
import tensorflow as tf
import numpy as np
import sys
from model import ImageClassifier
from datetime import datetime
import inputs
from PIL import Image


# generate lookup tables
np.random.seed(1)
#n = 256
#max_value = 16581375 #255**3
#interval = int(max_value / n)
#colors = [hex(I)[2:].zfill(6) for I in range(0, max_value, interval)]
#
#color_lut = [(int(i[:2], 16), int(i[2:4], 16), int(i[4:], 16)) for i in colors]
color_lut =range(256, 0, -1) * 3 # [ x for x in range(256) ]
np.random.shuffle(color_lut)


NUM_CLASSES = 255
IMAGE_SIZE = 128

x, y = inputs.train_pipeline("data/test.tfrecord", IMAGE_SIZE, batch_size=1, num_epochs=1)
classifier_model = ImageClassifier(
        x, y, NUM_CLASSES, IMAGE_SIZE,
        batch_size=1, eval=True)
#sess = tf.Session()
sess = tf.InteractiveSession()

train_writer = tf.train.SummaryWriter('output/eval', sess.graph)

coord = tf.train.Coordinator()

with sess.as_default():
    sess.run(tf.initialize_all_variables())
    sess.run(tf.initialize_local_variables())

    classifier_model.load(sess)

    threads = tf.train.start_queue_runners(coord=coord)

    test_writer = tf.train.SummaryWriter("output/eval/", sess.graph)

    try:
        step = 0
        correct = 0 # counts correct predictions
        total = 0 # counts total evaluated
        start = datetime.now()
        while not coord.should_stop():

            predictions, summary, image, label, class_img = classifier_model.evaluate_once(sess)
            correct += np.sum(predictions)
            total += len(predictions)


            image = Image.fromarray(np.uint8(np.asarray(image[0])))
            label = Image.fromarray(np.uint8(np.asarray(label[0])))
            class_img = np.squeeze(np.asarray(class_img[0], np.uint8), 2)
            class_img = Image.fromarray(class_img, "L")

            class_img = class_img.convert(mode="RGB")
            #class_img = class_img.point(lambda i: i * 1.2 + 10)
            class_img = class_img.point(color_lut)

            images = [image, label, class_img]
            widths, heights = zip(*(i.size for i in images))

            total_width = sum(widths)
            max_height = max(heights)

            new_im = Image.new('RGB', (total_width, max_height))

            x_offset = 0
            for im in images:
              new_im.paste(im, (x_offset,0))
              x_offset += im.size[0]

            new_im.save('classifications/%s.png' % step)

            if step % 10 is 0:

                now = datetime.now()
                elapsed = now - start
                average = elapsed / step if not step is 0 else 0
                print("Step %d, Evaluation Accuracy %g, Average Time %s/step, Elapsed Time %s" % (step, (correct/float(total)), average, elapsed))
                sys.stdout.flush()

            if step % 10 is 0:
                test_writer.add_summary(summary, step)

            step += 1
    except tf.errors.OutOfRangeError:
        print()
        print("Done evaluating, completed in %d steps" % step)
    finally:
        coord.request_stop()

    coord.join(threads)
