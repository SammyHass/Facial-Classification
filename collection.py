import cv2 as cv
import os
import numpy as np
face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')

cap = cv.VideoCapture(0)
inp_title = input("Name of input: ")
if not os.path.exists(os.path.join("data", inp_title)):
	os.mkdir("data/" +  inp_title)
i = len(os.listdir("data/" + inp_title))-1
while (True):
	ret, frame = cap.read()
	if ret == True:
		gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
		faces = face_cascade.detectMultiScale(gray, 1.3, 5)
		if (type(faces) == np.ndarray):
			i+=1
			print(faces[0][0])
			cv.imwrite(os.path.join("data", inp_title, "frame_{}.png".format(i)), frame[faces[0][1]: faces[0][1] + faces[0][3], faces[0][0]: faces[0][0] + faces[0][2]])
			cv.rectangle(frame,(faces[0][0], faces[0][1]),(faces[0][0]+faces[0][2],faces[0][1]+faces[0][3]),(255,0,0),2)
		cv.imshow("Frame", frame)
		if cv.waitKey(25) & 0xFF == ord("q"):
			break

	else:
		break
cap.release()

cv.destroyAllWindows()
