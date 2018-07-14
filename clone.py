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
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25), (0,0))))
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
#history_object = model.fit(x_train, y_train, validation_split = 0.2, shuffle = True, nb_epoch = 5)
model.fit_generator(train_generator, samples_per_epoch = len(train_samples), 
	validation_data=validation_generator,
	nb_val_samples=len(validation_samples), nb_epoch=3)

model.save('model.h5')

import matplotlib.pyplot as plt

### print the keys contained in the history object
print(history_object.history.keys())

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()