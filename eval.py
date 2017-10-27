from __future__ import print_function

import sys, os
from datetime import datetime
import inputs
import csv
import itertools

import tensorflow as tf
import numpy as np
from PIL import Image, ImageFont, ImageDraw

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

from calculate_labels import lookup, unique
import config


tf.flags.DEFINE_boolean("confusion_matrix", False, "Toggles building a confusion matrix")

tf.flags.DEFINE_string("output_dir", "output/", "The name of the directory to save checkpoints and summaries to.")
tf.flags.DEFINE_string("model_name", "model/", "The name of the directory to save training summaries to")

tf.flags.DEFINE_string("model_type", None, "The name of the model to use. Availiable models: cnn, adv")

FLAGS = tf.app.flags.FLAGS

names = [
    "Background",
    "Open Water",
    "Developed",
    "Barren",
    "Forest",
    "Shrubland",
    "Herbaceous",
    "Planted/Cultivated",
    "Wetlands"
]

# generate lookup tables
np.random.seed(1)
color_lut =range(256, 0, -1) * 3

np.random.shuffle(color_lut)

banded_lut = []

for x in range(256):
 banded_lut.append(( color_lut[x], color_lut[256+x], color_lut[(256*2)+x] ))


font = ImageFont.load_default()
txt = Image.new("RGB", (100,600), (255, 255, 255))
draw = ImageDraw.Draw(txt)
y = 0
for color, label in zip(banded_lut, names):
    print(color)
    draw.text((0,y), label, font=font, fill=color)
    y+= 10

txt.save("classifications/legend.png")



def main(argv=None):

    print("Parameters:")
    for k,v in FLAGS.__flags.items():
        print(k, "=", v)
    print()

    input_generator = inputs.test_pipeline()

    if FLAGS.model_type == "cnn":
        from cnn_classifier import ImageClassifier
        classifier_model = ImageClassifier(config.NUM_CLASSES, config.IMAGE_SIZE, batch_size=1, checkpoint_file=FLAGS.output_dir + FLAGS.model_name + "/", eval=True )
    elif FLAGS.model_type == "adv":
        from adversarial_classifier import ImageClassifier
        classifier_model = ImageClassifier(config.NUM_CLASSES, config.IMAGE_SIZE, batch_size=1, checkpoint_file=FLAGS.output_dir + FLAGS.model_name + "/", eval=True )
    else:
        raise Exception("--model_type parameter required.")

    sess = tf.Session()
    #sess = tf.InteractiveSession()

    conf_actual = []
    conf_predicted = []


    with sess.as_default():
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        classifier_model.load(sess)


        test_writer = tf.summary.FileWriter(FLAGS.output_dir + FLAGS.model_name + "/eval/", sess.graph)

        emb = []
        imgs = []

        conf_matrix = [ [ 0 for x in range(config.NUM_CLASSES) ] for y in range(config.NUM_CLASSES) ]

        total_pixels = 0
        per_class_accuracy = [ 0 for i in range(config.NUM_CLASSES) ]
        per_class_count = [ 0 for i in range(config.NUM_CLASSES) ]

        try:
            step = 0
            correct = 0 # counts correct predictions
            total = 0 # counts total evaluated
            start = datetime.now()
            for batch in input_generator:

                predictions, summary, image, label, class_img, image_tensor = classifier_model.evaluate_once(sess, batch)
                correct += np.sum(predictions)
                total += len(predictions)

                class_img.shape = [config.IMAGE_SIZE, config.IMAGE_SIZE]
                label.shape = [config.IMAGE_SIZE, config.IMAGE_SIZE]

                # calculate class-wise accuracies
                for actual, predicted in zip(label.flatten(), class_img.flatten()):
                    per_class_count[lookup[unique[int(actual)]]] += 1
                    if lookup[unique[int(actual)]] == lookup[unique[int(predicted)]]:
                        per_class_accuracy[
                            lookup[
                                unique[
                                    int(actual)
                                ]
                            ]] += 1

                print(" ".join([ str(float(a)/float(b)*100)+"%" if b else "100%" for a, b in zip(per_class_accuracy, per_class_count) ]))


                # generate confusion matrix
                if step % 100 == 0:
                    conf_predicted = conf_predicted + list(label.flatten())
                    conf_actual = conf_actual + list(class_img.flatten())


                image = Image.fromarray(np.uint8(np.asarray(image[0])))

                label = np.vectorize(lambda x: unique[int(x)])(label)
                label.shape = [1, config.IMAGE_SIZE, config.IMAGE_SIZE]
                label = Image.fromarray(np.uint8(np.asarray(label[0])))

                class_img = np.vectorize(lambda x: unique[int(x)])(class_img.flatten())
                class_img.shape = [config.IMAGE_SIZE, config.IMAGE_SIZE]
                class_img = np.asarray(class_img, np.uint8)
                class_img = Image.fromarray(class_img, "L")

                predictions.shape = (config.IMAGE_SIZE, config.IMAGE_SIZE)
                error_img = Image.fromarray(np.asarray(predictions, dtype=np.uint8) * 255)

                label = label.convert(mode="RGB")
                label = label.point(color_lut)

                class_img = class_img.convert(mode="RGB")
                #class_img = class_img.point(lambda i: i * 1.2 + 10)
                class_img = class_img.point(color_lut)


                overlay_img = Image.blend(image, class_img, 0.4)

                images = [image, label, class_img, overlay_img, error_img ]
                widths, heights = zip(*(i.size for i in images))

                total_width = sum(widths)
                max_height = max(heights)

                new_im = Image.new('RGB', (total_width, max_height))

                x_offset = 0
                for im in images:
                  new_im.paste(im, (x_offset,0))
                  x_offset += im.size[0]

                # make dir if it doesn't exist
                if not os.path.exists(FLAGS.output_dir + FLAGS.model_name + '/classifications/'):
                    os.makedirs(FLAGS.output_dir + FLAGS.model_name + '/classifications/')

                new_im.save(FLAGS.output_dir + FLAGS.model_name + '/classifications/%s.png' % step)

                if step % 10 is 0:

                    now = datetime.now()
                    elapsed = now - start
                    average = elapsed / step if not step is 0 else 0
                    print("Step %d, Evaluation Accuracy %g, Average Time %s/step, Elapsed Time %s" %
                            (step, (correct/float(total)), average, elapsed))
                    sys.stdout.flush()

                if step % 10 is 0:
                    test_writer.add_summary(summary, step)

                step += 1
        except tf.errors.OutOfRangeError:
            pass #TODO: determine if this is still needed

        print()
        print("Done evaluating, completed in %d steps" % step)
        print("Per class accuracies:")
        print(" ".join([ str(float(a)/float(b)*100)+"%" if b else "100%" for a, b in zip(per_class_accuracy, per_class_count) ]))
        print("Per class pixel counts:")
        print(" ".join([ str(b) for b in per_class_count ]))

        cnf_matrix = confusion_matrix(conf_actual, conf_predicted)
        np.set_printoptions(precision=2)

        def plot_confusion_matrix(cm, classes,
                  normalize=False,
                  title='Confusion matrix',
                  cmap=plt.cm.Blues):
            """
            This function prints and plots the confusion matrix.
            Normalization can be applied by setting `normalize=True`.
            """
            if normalize:
                cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
                print("Normalized confusion matrix")
            else:
                print('Confusion matrix, without normalization')

            print(cm)

            plt.imshow(cm, interpolation='nearest', cmap=cmap)
            plt.title(title)
            plt.colorbar()
            tick_marks = np.arange(len(classes))
            plt.xticks(tick_marks, classes, rotation=45)
            plt.yticks(tick_marks, classes)

            fmt = '.2f' if normalize else 'd'
            thresh = cm.max() / 2.
            for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
                plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

            plt.tight_layout()
            plt.ylabel('True label')
            plt.xlabel('Predicted label')

        plt.figure()
        plot_confusion_matrix(cnf_matrix, classes=
            ["unlabeled",
             "developed",
             "barren",
             "forest",
             "shrubland",
             "herbacous",
             "pasture/hay",
             "wetlands"],
              normalize=True,
              title='Normalized confusion matrix')


        plt.savefig(FLAGS.output_dir + FLAGS.model_name + "/classifications/conf_matrix.png")



if __name__ == "__main__":
    tf.app.run()
