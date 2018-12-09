import os
import numpy as np
from random import shuffle
import cv2 as cv

aug_path = "aug_data"
train_xs = np.empty(dtype="float32", shape=(int(len(os.listdir(aug_path))*0.8)+1, 224, 224, 3))
train_ys = np.zeros(dtype="float32", shape=(int(len(os.listdir(aug_path))*0.8)+1, 2))

test_xs = np.empty(dtype="float32", shape=(int(len(os.listdir(aug_path))*0.2), 224, 224, 3))
test_ys = np.zeros(dtype="float32", shape=(int(len(os.listdir(aug_path))*0.2), 2))
print(train_xs.shape, train_ys.shape)
print(test_xs.shape, test_ys.shape)

train_index = 0
test_index = int(len(os.listdir(aug_path))*0.8)+1

for i in range(len(os.listdir(aug_path))):
	print(i)
	if i < test_index:
		im = cv.imread(os.path.join(aug_path, os.listdir(aug_path)[i]))
		train_xs[i] = im/255
		cat = os.listdir(aug_path)[i].split("_")[0]
		if cat == "not":
			train_ys[i][1] = 1
		else:
			train_ys[i][0] = 1
	else:
		im = cv.imread(os.path.join(aug_path, os.listdir(aug_path)[i]))
		test_xs[i-test_index] = im/255
		cat = os.listdir(aug_path)[i].split("_")[0]
		if cat == "not":
			test_ys[i-test_index][1] = 1
		else:
			test_ys[i-test_index][0] = 1

np.save("train/xs.npy", train_xs)
np.save("train/ys.npy", train_ys)
np.save("test/xs.npy", test_xs)
np.save("test/ys.npy", test_ys)
