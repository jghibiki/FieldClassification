import tensorflow as tf
from PIL import Image
import numpy as np
import calculate_labels

NUM_IMAGES = 6291
IMAGE_SIZE = 256


def calculate_splits():

    image_list = np.arange(NUM_IMAGES)
    np.random.shuffle(image_list)
    test_size = int(image_list.shape[0]*0.1)
    test = image_list[:test_size]
    train = image_list[test_size:]

    return (train, test)


def getImage(base, i):
    image_r = Image.open("%s/IMG-R-%08d.png" % (base, i))
    image_g = Image.open("%s/IMG-G-%08d.png" % (base, i))
    image_b = Image.open("%s/IMG-B-%08d.png" % (base, i))
    image_a = Image.open("%s/IMG-A-%08d.png" % (base, i))


    #r = np.array(image_r, dtype=np.float32)
    #a = np.array(image_a, dtype=np.float32)
    #np.seterr(invalid='ignore')
    #ndvi = (a-r)/(a+r)

    ##ndvi = (ndvi + 1) * (2**15 - 1)
    ##ndvi = ndvi.astype(np.uint8)


    #print(r)
    #print(a)
    #print(ndvi)
    #print(ndvi.shape)

    image = np.array([
        np.array(image_r)[..., np.newaxis],
        np.array(image_g)[..., np.newaxis],
        np.array(image_b)[..., np.newaxis],
        np.array(image_a)[..., np.newaxis]
    ])
    image = np.concatenate(image, axis=-1)
    return image

def getLabel(base, i):
    labels = Image.open("%s/LBL-%08d.png" % (base, i))

    labels = np.asarray(labels)

    simplified_labels = [ [ calculate_labels.lookup[pixel] if pixel != 0 else 1 for pixel in y ] for y in labels ]
    simplified_labels = np.asarray(simplified_labels, np.uint8)

    return simplified_labels


def train_pipeline(batch_size, num_epochs=1):

    train, _ = calculate_splits()

    for j in range(num_epochs):
        for i in range(0, len(train)-batch_size, batch_size):
            img_batch = []
            lbl_batch = []
            for num in train[i:i+batch_size]:

                img = getImage("raw_images", num)
                lbl = getLabel("raw_images", num)

                img_batch.append(img)
                lbl_batch.append(lbl)

            yield (img_batch, lbl_batch)


def test_pipeline():

    _, test  = calculate_splits()

    for num in range(0, len(test)):

        img = getImage("raw_images", num)
        lbl = getLabel("raw_images", num)

        yield ([img], [lbl])

