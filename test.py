from keras.models import load_model
import os
import cv2 as cv
import numpy as np



people = ["Sammy", "Not Sammy"]
face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')


model_path = "models"
models = ["classifier.h5"]

model = load_model("models/" + models[0])

cap = cv.VideoCapture(0)
while (True):
	ret, frame = cap.read()
	gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
	faces = face_cascade.detectMultiScale(gray, 1.3, 5)
	if (type(faces) == np.ndarray):
		im = cv.resize(frame[faces[0][1]: faces[0][1] + faces[0][3], faces[0][0]: faces[0][0] + faces[0][2]], (224, 224)) / 255
		cv.putText(frame, people[np.argmax(model.predict(np.reshape(im, (1, 224, 224, 3))))], (0,50), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
		cv.rectangle(frame,(faces[0][0], faces[0][1]),(faces[0][0]+faces[0][2],faces[0][1]+faces[0][3]),(255,0,0),2)
		print(model.predict(np.reshape(im, (1, 224, 224, 3))))
		
	cv.imshow("Prediction", frame)

	if ret == True:
		if cv.waitKey(25) & 0xFF == ord("q"):
			break
		
		
	else:
		break
cap.release()

cv.destroyAllWindows()