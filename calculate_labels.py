from __future__ import print_function
import gdal
import numpy as np
from PIL import Image

labels = []

NUM_IMAGES = 1
for image_no in range(1,NUM_IMAGES+1):
    im2 = gdal.Open("images/%s_label.tif" % image_no)
    im2 = np.array(im2.GetRasterBand(1).ReadAsArray()).flatten()
    labels += np.unique(im2).tolist()


unique = sorted(set(labels))

print("Labels Present: \n%s\nTotal: %s" % (unique, len(unique)))

lookup = { value: key for key, value in enumerate(unique) }

