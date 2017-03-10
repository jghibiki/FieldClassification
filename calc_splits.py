import numpy as np

NUM_IMAGES = 18406
IMAGE_SIZE = 128

np.random.seed(161) # just a randomly chosen number


image_list = np.arange(NUM_IMAGES)
np.random.shuffle(image_list)
test_size = int(image_list.shape[0]*0.1)
test = image_list[:test_size]
train = image_list[test_size:]

np.save("train", train)
np.save("test", test)
