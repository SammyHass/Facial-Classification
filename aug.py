import os
import numpy as np
import cv2 as cv

data_path = os.path.join("data")
aug_path = os.path.join("aug_data")

cats = []
for i in range(len(os.listdir(data_path))):
	cats.append(os.listdir(data_path)[i])

for i in range(len(cats)):
	for j in range(len(os.listdir(os.path.join(data_path, cats[i])))):
		print(os.path.join(data_path, cats[i], os.listdir(os.path.join(data_path, cats[i]))[j]))
		frame = cv.imread(os.path.join(data_path, cats[i], os.listdir(os.path.join(data_path, cats[i]))[j]))
		if frame.shape[0] * frame.shape[1] > 500:
			im = cv.resize(frame, (224, 224))
			cv.imwrite(os.path.join(aug_path, cats[i]+"_"+str(j)+".png"), im)

print("Found {} categories, {} and resized to 224x244 and put into {}".format(len(cats), cats, aug_path))
