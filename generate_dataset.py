from __future__ import print_function
import gdal
from PIL import Image
import math
import numpy as np
import sys
import csv
from os import listdir
from os.path import isfile, join


sys.setrecursionlimit(10**6)

print("Exporting Image Data")
print()
k = 0


file_path = "images/images"
files = [f for f in listdir(file_path) if isfile(join(file_path, f))]

for image_no, file_name in enumerate(files):
    im = np.array(Image.open("images/images/%s" % file_name))
    label = np.array(Image.open("images/gt/%s" % file_name))
    im2 = np.array(label)

    im_r = im[:,:,0]
    im_g = im[:,:,1]
    im_b = im[:,:,2]
    #im_a = im[:,:,3]

    image_r = Image.fromarray(im_r).convert("L")
    image_g = Image.fromarray(im_g).convert("L")
    image_b = Image.fromarray(im_b).convert("L")
    #image_a = Image.fromarray(im_a).convert("L")
    label = Image.fromarray(im2)

    source_w, source_h = len(im_r), len(im_r[0])
    label_w, label_h = len(im2), len(im2[0])


    out_w, out_h = 1000, 1000
    print("Image #%s" % image_no)
    print("Output image size: %sx%s" % (out_h, out_w))
    print("Input image size: %sx%s" % (source_h, source_w))
    print("Label image size: %sx%s" % (label_h, label_w))

    for i in range(0, source_w, int(out_w/2)):
        for j in range(0, source_h, int(out_h/2)):

            #if k % 0 is 0:
            print("Exported %008d x1:%s y1:%s x2:%s, y2:%s\r" % (k, i, j, i+out_w, j+out_h), end="")
            sys.stdout.flush()


            # modify to send data to tfrecords file
            r = image_r.crop((i, j, i+out_h, j+out_w))
            img_count = np.bincount(np.reshape(np.array(r), (-1)))

            label_img = label.crop((i, j, i+(out_h), j+(out_w)))
            label_count = np.bincount(np.reshape(np.array(label_img), (-1)))


            if not np.argmax(img_count) == 0 and not np.argmax(label_count) == 0:
                r.load()
                r.save("raw_images/IMG-R-%008d.png" % k)

                g = image_g.crop((i, j, i+out_h, j+out_w))
                g.load()
                g.save("raw_images/IMG-G-%008d.png" % k)

                b = image_b.crop((i, j, i+out_h, j+out_w))
                b.load()
                b.save("raw_images/IMG-B-%008d.png" % k)

                #a = image_a.crop((i, j, i+out_h, j+out_w))
                #a.load()
                #a.save("raw_images/IMG-A-%008d.png" % k)

                label_img = np.array(label_img)

                def convert(x):
                    if x < 0:
                        return 1
                    if x == 0:
                        return 150
                    if x > 1:
                        return 255
                    return x

                vec_convert = np.vectorize(convert)
                label_img = vec_convert(label_img)

                #label_img = [ [ x if x in LABELS else 0 for x in y ] for y in label_img ]

                label_img = Image.fromarray(np.array(label_img, np.uint8))

                label_img.save("raw_images/LBL-%008d.png" % k)

                b = Image.merge("RGB", [r, g, b])
                b.save("raw_images/IMG-%008d-visible.png" % k)


                #c = label_img.point(lut)
                #c.save("raw_images/LBL-%008d-visible.png" % k)

                k += 1
    print()
print()

