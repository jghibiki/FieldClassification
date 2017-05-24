
import numpy as np
from PIL import Image

NUM_IMAGES = 9201

bins = np.zeros(256)
for image_no in range(1,NUM_IMAGES):
    im2 = Image.open("raw_images/LBL-%08d.png" % image_no)
    im2 = np.array(im2).flatten()
    #im2 = np.array(im2.GetRasterBand(1).ReadAsArray()).flatten()
    bins += np.bincount(im2, minlength=256)


it = np.nditer(bins, flags=['f_index'])
bsum = np.sum(bins)
while not it.finished:
    print("{0}: {1}%".format(it.index, (it[0]/bsum)*100))
    it.iternext()

