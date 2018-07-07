import csv
import cv2
import numpy as np

lines = []
with open('./simulation_training_data/driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	for row in reader:
		lines.append(row)

images = []
measurments = []
for line in lines:
	source_path = line[0]
	filename = source_path.split('\\')[-1]
	current_path = './simulation_training_data/IMG/' + filename
	image = cv2.imread(current_path)
	print(current_path)
	images.append(image)
	measurment = float(line[3])
	measurments.append(measurment)
	
x_train = np.array(images)
y_train = np.array(measurments)	


from keras.models import Sequential
from keras.layers import Flatten, Dense


model = Sequential()
model.add(Flatten(input_shape = (160,320,3))) 
model.add(Dense(1))

model.compile(loss = 'mse', optimizer = 'adam')
model.fit(x_train, y_train, validation_split = 0.2, shuffle = True, nb_epoch = 7)

model.save('model.h5')

