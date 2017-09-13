from __future__ import print_function
import tensorflow as tf
import numpy as np
import sys
from model import ImageClassifier
from datetime import datetime
import inputs
import csv
from PIL import Image, ImageFont, ImageDraw
from calculate_labels import lookup, unique


tf.flags.DEFINE_boolean("confusion_matrix", False, "Toggles building a confusion matrix")

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



IMAGE_SIZE = 512
NUM_CLASSES = 9

def main(argv=None):

    print("Parameters:")
    for k,v in FLAGS.__flags.items():
        print(k, "=", v)
    print()

    input_generator = inputs.test_pipeline()
    classifier_model = ImageClassifier(NUM_CLASSES, IMAGE_SIZE, batch_size=1, eval=True, checkpoint_file="output/model.ckpt-1000-5000-2500-1000")

    #sess = tf.Session()
    sess = tf.InteractiveSession()


    with sess.as_default():
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        classifier_model.load(sess)


        test_writer = tf.summary.FileWriter("output/eval/", sess.graph)

        emb = []
        imgs = []

        conf_matrix = [ [ 0 for x in range(NUM_CLASSES) ] for y in range(NUM_CLASSES) ]

        total_pixels = 0
        per_class_accuracy = [ 0 for i in range(NUM_CLASSES) ]
        per_class_count = [ 0 for i in range(NUM_CLASSES) ]

        try:
            step = 0
            correct = 0 # counts correct predictions
            total = 0 # counts total evaluated
            start = datetime.now()
            for batch in input_generator:

                predictions, summary, image, label, class_img, image_tensor = classifier_model.evaluate_once(sess, batch)
                correct += np.sum(predictions)
                total += len(predictions)

                class_img.shape = [IMAGE_SIZE, IMAGE_SIZE]
                label.shape = [IMAGE_SIZE, IMAGE_SIZE]

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
                for actual, predicted in zip(label.flatten(), class_img.flatten()):
                    index_of_actual = lookup[ unique[ int(actual) ] ]
                    index_of_predicted = lookup[ unique[ int(predicted) ] ]
                    #print("Actual", names[index_of_actual], "Predicted", names[index_of_predicted])
                    conf_matrix[
                        lookup[
                            unique[
                                int(actual)
                            ]
                        ]][ lookup[
                            unique[
                                int(predicted)
                            ]
                        ]] += 1

                image = Image.fromarray(np.uint8(np.asarray(image[0])))

                label = np.vectorize(lambda x: unique[int(x)])(label)
                label.shape = [1, IMAGE_SIZE, IMAGE_SIZE]
                label = Image.fromarray(np.uint8(np.asarray(label[0])))

                class_img = np.vectorize(lambda x: unique[int(x)])(class_img.flatten())
                class_img.shape = [IMAGE_SIZE, IMAGE_SIZE]
                class_img = np.asarray(class_img, np.uint8)
                class_img = Image.fromarray(class_img, "L")

                predictions.shape = (IMAGE_SIZE, IMAGE_SIZE)
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

                new_im.save('classifications/%s.png' % step)

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


        if FLAGS.confusion_matrix:
            with open("confusion_matrix.csv", "wb") as f:
                writer = csv.writer(f)
                for row in conf_matrix:
                    writer.writerow(row)


if __name__ == "__main__":
    tf.app.run()
