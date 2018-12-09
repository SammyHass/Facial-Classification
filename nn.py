import numpy as np
import cv2 as cv
import os
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD

train_path = "train"
test_path = "test"
x_train = np.load(os.path.join(train_path, "xs.npy"))
y_train = np.load(os.path.join(train_path, "ys.npy"))
x_test = np.load(os.path.join(test_path, "xs.npy"))
y_test = np.load(os.path.join(test_path, "ys.npy"))


model = Sequential()

model.add(Conv2D(32, (3, 3), activation="relu", input_shape=(224, 224, 3)))
model.add(Conv2D(32, (3, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, (3, 3), activation="relu"))
model.add(Conv2D(128, (3, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(2, activation="softmax"))

sgd = SGD(lr=0.01)
model.compile(loss="mean_squared_error", optimizer=sgd, metrics=["accuracy"])

model.fit(x_train, y_train, batch_size=1, epochs=2)
score = model.evaluate(x_test, y_test, batch_size=1)
print(score)
model.save("models/classifier.h5")