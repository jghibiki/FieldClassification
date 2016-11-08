from __future__ import print_function
import gdal
from PIL import Image
import numpy as np
import sys
import csv

im = gdal.Open("input.tif")
im2 = gdal.Open("labels.tif")

im = np.array([
    np.array(im.GetRasterBand(1).ReadAsArray()),
    np.array(im.GetRasterBand(2).ReadAsArray()),
    np.array(im.GetRasterBand(3).ReadAsArray()),
    np.array(im.GetRasterBand(4).ReadAsArray())
])

im2 = np.array(im2.GetRasterBand(1).ReadAsArray())

source_w, source_h = len(im[0]), len(im[0][0])

im.shape = (source_w, source_h, 4)

out_w, out_h = 128, 128
print("Output image size: %sx%s" % (out_h, out_w))
print("Input image size: %sx%s" % (source_h, source_w))

k = 0
for i in xrange(0, source_h, out_h):
    for j in xrange(0, source_w, out_w):
        if k % 10 is 0:
            print("Exporting %008d x:%s y:%s\r" % (k, i, j), end="")
            sys.stdout.flush()


        a = im[j:j+out_w, i:i+out_h]
        label = im2[j:j+out_w, i:i+out_h]

        # deals with cut off images
        if(len(a) is 128 and len(a[0]) is 128 and len(a[0][0]) is 4):
            if(len(label) is 128 and len(label[0]) is 128):

                # modify to send data to tfrecords file
                a = Image.fromarray(a)
                a.save("raw_images/IMG-%008d.png" % k)

                label = Image.fromarray(label)
                label.save("raw_images/LBL-%008d.png" % k)

                b = a.convert("RGB")
                b.save("raw_images/IMG-%008d-visible.png" % k)
                k += 1

print()
