from __future__ import print_function
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
import numpy as np
import sys
from model import ImageClassifier
from datetime import datetime
import inputs
import csv
from PIL import Image, ImageFont, ImageDraw
from calculate_labels import lookup, unique


tf.flags.DEFINE_boolean("save_projections", False, "Toggles saving projections")
tf.flags.DEFINE_boolean("confusion_matrix", False, "Toggles building a confusion matrix")

FLAGS = tf.app.flags.FLAGS

names = [
        "background",
        "corn",
        "Sorghum",
        "Soybeans",
        "Sunflower",
        "Barley",
        "Durum Wheat",
        "Spring Wheat",
        "Winter Wheat",
        "Rye",
        "Oats",
        "Millet",
        "Canola",
        "Flaxseed",
        "Alfalfa",
        "Other Hay/Non Alfalfa",
        "Buckwheat",
        "Sugarbeets",
        "Dry Beans",
        "Potatoes",
        "Other Crops",
        "Peas",
        "Fallow/Idle Cropland",
        "Open Water",
        "Developed/Open Space",
        "Developed/Low Intensity",
        "Developed/Med Intensity",
        "Developed/High Intensity",
        "Barren",
        "Deciduous Forest",
        "Evergreen Forest",
        "Mixed Forest",
        "Shrubland",
        "Grass/Pasture",
        "Woody Wetlands",
        "Herbaceous Wetlands",
        "Radishes"
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



IMAGE_SIZE = 256
NUM_CLASSES = 15

def main(argv=None):

    print("Parameters:")
    for k,v in FLAGS.__flags.items():
        print(k, "=", v)
    print()

    input_generator = inputs.test_pipeline()
    classifier_model = ImageClassifier(NUM_CLASSES, IMAGE_SIZE, batch_size=1, eval=True, checkpoint_file="output/model.ckpt-1000-5000-2500-3000")

    #sess = tf.Session()
    sess = tf.InteractiveSession()


    with sess.as_default():
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        classifier_model.load(sess)


        test_writer = tf.train.SummaryWriter("output/eval/", sess.graph)

        emb = []
        imgs = []

        conf_matrix = [ [ 0 for x in range(NUM_CLASSES) ] for y in range(NUM_CLASSES) ]

        try:
            step = 0
            correct = 0 # counts correct predictions
            total = 0 # counts total evaluated
            forgive = 0
            start = datetime.now()
            for batch in input_generator:

                predictions, summary, image, label, class_img, image_tensor = classifier_model.evaluate_once(sess, batch)
                correct += np.sum(predictions)
                total += len(predictions)

                class_img.shape = [IMAGE_SIZE, IMAGE_SIZE]
                label.shape = [IMAGE_SIZE, IMAGE_SIZE]

                forgive_a = []

                for y in range(0, IMAGE_SIZE):
                    for x in range(0, IMAGE_SIZE):

                        above = False
                        if( y > 5 ):
                            above = class_img[y][x] == label[y-1][x]
                            above = above or class_img[y][x] == label[y-2][x]
                            above = above or class_img[y][x] == label[y-3][x]
                            above = above or class_img[y][x] == label[y-4][x]
                            above = above or class_img[y][x] == label[y-5][x]

                        below = False
                        if (y < IMAGE_SIZE-6):
                            below = class_img[y][x] == label[y+1][x]
                            below = below or class_img[y][x] == label[y+2][x]
                            below = below or class_img[y][x] == label[y+3][x]
                            below = below or class_img[y][x] == label[y+4][x]
                            below = below or class_img[y][x] == label[y+5][x]

                        left = False
                        if (x > 5):
                            left = class_img[y][x] == label[y][x-1]
                            left = left or class_img[y][x] == label[y][x-2]
                            left = left or class_img[y][x] == label[y][x-3]
                            left = left or class_img[y][x] == label[y][x-4]
                            left = left or class_img[y][x] == label[y][x-5]

                        right = False
                        if( x < IMAGE_SIZE-6):
                            right = class_img[y][x] == label[y][x+1]
                            right = right or class_img[y][x] == label[y][x+2]
                            right = right or class_img[y][x] == label[y][x+3]
                            right = right or class_img[y][x] == label[y][x+4]
                            right = right or class_img[y][x] == label[y][x+5]

                        at = class_img[y][x] == label[y][x]

                        forgive_a.append((at or above or below or left or right ))


                a = np.sum(np.asarray(forgive_a, dtype=bool))
                forgive += a


                if FLAGS.confusion_matrix:
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

                if FLAGS.save_projections:
                    emb.append(image_tensor)
                    imgs.append(class_img)

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

                forgive_a = np.asarray(forgive_a)
                forgive_a.shape = [IMAGE_SIZE, IMAGE_SIZE]
                forgive_a = Image.fromarray(np.asarray(forgive_a, dtype=np.uint8) * 255)

                label = label.convert(mode="RGB")
                label = label.point(color_lut)

                class_img = class_img.convert(mode="RGB")
                #class_img = class_img.point(lambda i: i * 1.2 + 10)
                class_img = class_img.point(color_lut)


                overlay_img = Image.blend(image, class_img, 0.4)

                images = [image, label, class_img, overlay_img, error_img, forgive_a]
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
                    print("Step %d, Evaluation Accuracy %g, Acc #2 %g, Average Time %s/step, Elapsed Time %s" %
                            (step, (correct/float(total)), (forgive/float(total)), average, elapsed))
                    sys.stdout.flush()

                if step % 10 is 0:
                    test_writer.add_summary(summary, step)

                step += 1
        except tf.errors.OutOfRangeError:
            print()
            print("Done evaluating, completed in %d steps" % step)

        if FLAGS.save_projections:

            emb = np.array(image_tensor)
            print(emb.shape)
            emb.shape = ( emb.shape[0] *  IMAGE_SIZE * IMAGE_SIZE, NUM_CLASSES)

            imgs = np.array(imgs, np.int64)
            imgs.shape = (imgs.shape[0] * IMAGE_SIZE * IMAGE_SIZE, 1)
            imgs = np.squeeze(imgs)

            metadata_file = open('output/eval/metadata.tsv', 'w')
            for i in range(imgs.shape[0]):
                metadata_file.write('%s\n' % (names[imgs[i]]))
            metadata_file.close()

            emb_var = tf.Variable(emb, name='embedding_of_images')
            sess.run(emb_var.initializer)


            summary_writer = tf.summary.FileWriter("output/eval")
            config = projector.ProjectorConfig()
            embedding = config.embeddings.add()
            embedding.tensor_name = emb_var.name
            embedding.metadata_path = 'output/eval/metadata.tsv'

            projector.visualize_embeddings(summary_writer, config)

            saver = tf.train.Saver([emb_var])
            saver.save(sess, 'output/eval/model.ckpt', 1)

        if FLAGS.confusion_matrix:
            with open("confusion_matrix.csv", "wb") as f:
                writer = csv.writer(f)
                for row in conf_matrix:
                    writer.writerow(row)


if __name__ == "__main__":
    tf.app.run()
