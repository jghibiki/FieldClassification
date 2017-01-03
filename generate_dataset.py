from __future__ import print_function
import gdal
from PIL import Image
import numpy as np
import sys
import csv

#lut = []
#with open("label_ledgend.csv") as f:
#    reader = csv.reader(f)
#    next(reader)
#    for line in reader:
#        if line[2] == "" or line[3] == "" or line[4] =="":
#            lut.append([0, 0, 0])
#        else:
#            lut.append([int(float(line[2])*255), int(float(line[3])*255), int(float(line[4])*255)])
#    lut.append([0,0,0])

print("Exporting Image Data")
print()
k = 0

NUM_IMAGES = 4
for image_no in xrange(1,NUM_IMAGES+1):
    im = gdal.Open("input_%s.tif" % image_no)
    im2 = gdal.Open("labels_%s.tif" % image_no)

    im_r = np.array(im.GetRasterBand(1).ReadAsArray())
    im_g = np.array(im.GetRasterBand(2).ReadAsArray())
    im_b = np.array(im.GetRasterBand(3).ReadAsArray())
    im_a = np.array(im.GetRasterBand(4).ReadAsArray())

    im2 = np.array(im2.GetRasterBand(1).ReadAsArray())

    image_r = Image.fromarray(im_r)
    image_g = Image.fromarray(im_g)
    image_b = Image.fromarray(im_b)
    image_a = Image.fromarray(im_a)
    label = Image.fromarray(im2)

    source_w, source_h = len(im_r), len(im_r[0])
    label_w, label_h = len(im2), len(im2[0])


    out_w, out_h = 128, 128
    print("Image #%s" % image_no)
    print("Output image size: %sx%s" % (out_h, out_w))
    print("Input image size: %sx%s" % (source_h, source_w))
    print("Label image size: %sx%s" % (label_h, label_w))

    for i in xrange(0, source_w, out_w):
        for j in xrange(0, source_h, out_h):

            #if k % 0 is 0:
            print("Exported %008d x1:%s y1:%s x2:%s, y2:%s\r" % (k, i, j, i+out_w, j+out_h), end="")
            sys.stdout.flush()


            # modify to send data to tfrecords file
            r = image_r.crop((i, j, i+out_h, j+out_w))
            count = np.bincount(np.reshape(np.array(r), (-1)))

            if not np.argmax(count) == 0 :
                r.load()
                r.save("raw_images/IMG-R-%008d.png" % k)

                g = image_g.crop((i, j, i+out_h, j+out_w))
                g.load()
                g.save("raw_images/IMG-G-%008d.png" % k)

                b = image_b.crop((i, j, i+out_h, j+out_w))
                b.load()
                b.save("raw_images/IMG-B-%008d.png" % k)

                a = image_a.crop((i, j, i+out_h, j+out_w))
                a.load()
                a.save("raw_images/IMG-A-%008d.png" % k)

                label_img = label.crop((i, j, i+(out_h), j+(out_w)))
                label_img.load()
                label_img.save("raw_images/LBL-%008d.png" % k)

                b = Image.merge("RGB", [r, g, b])
                b.save("raw_images/IMG-%008d-visible.png" % k)


                #c = label_img.point(lut)
                #c.save("raw_images/LBL-%008d-visible.png" % k)

                k += 1
    print()
print()

