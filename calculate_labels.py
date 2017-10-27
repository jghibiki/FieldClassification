from __future__ import print_function
import numpy as np
from PIL import Image

import config

labels = []

FORCE_NUM_LABELS = 10

for image_no in range(1, config.NUM_IMAGES):
    im2 = Image.open("raw_images/LBL-%08d.png" % image_no)
    im2 = np.array(im2).flatten()
    #im2 = np.array(im2.GetRasterBand(1).ReadAsArray()).flatten()
    labels += np.unique(im2).tolist()

#unique = [0, 1]

unique = sorted(set(labels))

counter = -1
while len(unique) < config.NUM_CLASSES:
    unique.append(counter)
    counter -= 1


lookup = { value: key for key, value in enumerate(unique) }
print("Labels Present: \n%s\nTotal: %s" % (unique, len(unique)))
print(lookup)

