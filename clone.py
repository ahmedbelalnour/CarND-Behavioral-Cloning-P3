import csv
import cv2
import numpy as np
from numpy import newaxis

def process_image(image):
	return image

lines = []
with open('./simulation_training_data/driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	for row in reader:
		lines.append(row)

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(lines, test_size=0.2)
import sklearn
from sklearn.utils import shuffle

def generator(samples, batch_size = 32):
	num_samples = len(samples)
	correction_factor = [0.0, +0.2, -0.2]

	while 1: # Loop forever so the generator never terminates
		shuffle(samples)
		for offset in range(0, num_samples, batch_size):
			batch_samples = samples[offset:offset+batch_size]
			
			images = []
			measurments = []
			
			for batch_sample in batch_samples:
				for i in range(3):
					#print("batch_sample: ", batch_sample)
					source_path = batch_sample[i]
					filename = source_path.split('/')[-1]
					current_path = './simulation_training_data/IMG/' + filename
					image = cv2.imread(current_path)
					image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
					image = image[..., newaxis]
					image_flipped = np.fliplr(image)
					images.append(image)
					images.append(image_flipped)
					steering = float(batch_sample[3]) + correction_factor[i]
					measurments.append(steering)
					measurment_flipped = -steering
					measurments.append(measurment_flipped)

			x_data = process_image(np.array(images))
			y_data = process_image(np.array(measurments))
			yield sklearn.utils.shuffle(x_data, y_data)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout, Lambda
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import Cropping2D

model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,1)))
model.add(Cropping2D(cropping=((70,25), (0,0))))

#first convolution layer
model.add(Convolution2D(32, 5, 5, subsample = (2,2), activation = 'relu'))

#second convolution layer
model.add(Convolution2D(64, 5, 5, subsample = (2,2), activation = 'relu'))

#third convolution layer
model.add(Convolution2D(128, 5, 5, subsample = (2,2), activation = 'relu'))

#forth convolution layer
model.add(Convolution2D(256, 3, 3, activation = 'relu'))

#second dropout layer
model.add(Dropout(0.4))

#fifth convolution layer
model.add(Convolution2D(256, 3, 3, activation = 'relu'))

model.add(Flatten())

#first fully connected
#model.add(Dense(1164))
#model.add(Activation('relu'))

#second fully connected
model.add(Dense(100))
model.add(Activation('relu'))

#third fully connected
model.add(Dense(50))
model.add(Activation('relu'))

#forth fully connected
model.add(Dense(10))
model.add(Activation('relu'))

#forth/last fully connected
model.add(Dense(1))

model.compile(loss = 'mse', optimizer = 'adam')
#history_object = model.fit(x_train, y_train, validation_split = 0.2, shuffle = True, nb_epoch = 5)
model.fit_generator(train_generator, samples_per_epoch = len(train_samples), 
	validation_data=validation_generator,
	nb_val_samples=len(validation_samples), nb_epoch=3)

model.save('model.h5')

import matplotlib.pyplot as plt


### print the keys contained in the history object
# print(history_object.history.keys())

### plot the training and validation loss for each epoch
# plt.plot(history_object.history['loss'])
# plt.plot(history_object.history['val_loss'])
#plt.title('model mean squared error loss')
#plt.ylabel('mean squared error loss')
#plt.xlabel('epoch')
#plt.legend(['training set', 'validation set'], loc='upper right')
#plt.show()
