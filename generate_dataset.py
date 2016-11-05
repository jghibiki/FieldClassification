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
l_w, l_h = out_w/4, out_h/4
print("Output image size: %sx%s" % (out_h, out_w))
print("Input image size: %sx%s" % (source_h, source_w))
print("Label image size: %sx%s" % (out_h-(l_h*2), out_w-(l_w*2)))

k = 0
for i in xrange(0, source_h, out_h):
    for j in xrange(0, source_w, out_w):
        if k % 100 is 0:
            print("Exporting %006d x:%s y:%s\r" % (k, i, j), end="")
            sys.stdout.flush()
        label_x1 = j + l_w
        label_x2 = j + out_w - l_w
        label_y1 = i + l_w
        label_y2 = i + out_h - l_h

        if label_x1 > source_w or label_x2 > source_w or label_y1 > source_h or label_y2 > source_h:
            continue

        a = im[j:j+out_w, i:i+out_h]
        label = im2[label_x1:label_x2, label_y1:label_y2]

        # modify to send data to tfrecords file
        a = Image.fromarray(a)
        a.save("raw_images/IMG-%s.png" % k)

        label = Image.fromarray(label)
        label.save("raw_images/LBL-%s.png" % k)

        b = a.convert("RGB")
        b.save("raw_images/IMG-%s-visible.png" % k)
        k += 1

print()
