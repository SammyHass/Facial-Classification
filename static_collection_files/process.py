import numpy as np
import cv2 as cv
import os

files = os.listdir("not_sammy")
face_cascade = cv.CascadeClassifier("../haarcascade_frontalface_default.xml")

for i in range(len(files) - 1):
	frame = cv.imread(os.path.join("not_sammy", files[i]))
	gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
	faces = face_cascade.detectMultiScale(gray, 1.3, 5)
	print(files[i])
	if type(faces) == np.ndarray:
		i+=1
		print(i)
		cv.imwrite(os.path.join("..", "data", "not_sammy", "frame_{}.png".format(i)), frame[faces[0][1]: faces[0][1] + faces[0][3], faces[0][0]: faces[0][0] + faces[0][2]])