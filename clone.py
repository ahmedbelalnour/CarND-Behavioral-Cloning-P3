import csv
import cv2
import numpy as np

def process_image(image):
	return image

lines = []
with open('./simulation_training_data/driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	for row in reader:
		lines.append(row)

images = []
measurments = []

correction_factor = [0.0, +0.2, -0.2]
for line in lines:
	for i in range(3):
		source_path = line[i]
		filename = source_path.split('/')[-1]
		current_path = './simulation_training_data/IMG/' + filename
		image = cv2.imread(current_path)
		image_flipped = np.fliplr(image)
		images.append(image)
		images.append(image_flipped)
		steering = float(line[3]) + correction_factor[i]
		measurments.append(steering)
		measurment_flipped = -steering
		measurments.append(measurment_flipped)

x_train = process_image(np.array(images))
y_train = process_image(np.array(measurments))

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout, Lambda
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))

#first convolution layer
model.add(Convolution2D(6, 5, 5, border_mode='valid'))
model.add(Activation('relu'))
model.add(MaxPooling2D((2, 2)))

#second convolution layer
model.add(Convolution2D(16, 5, 5, border_mode='valid'))
model.add(Activation('relu'))
model.add(MaxPooling2D((2, 2)))

model.add(Flatten())

model.add(Dense(84))
model.add(Activation('relu'))

model.add(Dense(1))

model.compile(loss = 'mse', optimizer = 'adam')
model.fit(x_train, y_train, validation_split = 0.2, shuffle = True, nb_epoch = 5)

model.save('model.h5')

