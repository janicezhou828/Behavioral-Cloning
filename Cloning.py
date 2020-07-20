import csv
import cv2
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
from keras.layers.convolutional import Conv2D, Convolution2D, Cropping2D
from keras.layers.pooling import MaxPooling2D

lines = []
with open ('./Data/Round1/driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		lines.append (line)
images = []
measurements = []

for line in lines:
	for i in range (3):
		source_path = line [i]
		tokens = source_path.split('\\')
		filename = tokens[-1]
		local_path = "./Data/Round1/IMG/" + filename
		image = cv2.imread(local_path)
		images.append(image)
	correction = 0.2
	measurement = line[3]
	measurements.append(float(measurement))
	measurements.append(float(measurement)+correction)
	measurements.append(float(measurement)-correction)

augmented_images = []
augmented_measurements = []

for image, measurement in zip(images, measurements):
	augmented_images.append(image)
	#measurement_fl = float (measurement)
	augmented_measurements.append(measurement)
	
	flipped_image = cv2.flip(image, 1)
	flipped_measurement = measurement* (-1.0)

	augmented_images.append(flipped_image)
	augmented_measurements.append(flipped_measurement)
	#exit ()

X_train = np.array(augmented_images)
y_train = np.array (augmented_measurements)

# To print the whole array for inspection
#import sys
#np.set_printoptions(threshold=sys.maxsize)
#print (X_train.shape)
#print (y_train.shape)
#print (y_train)
#print (type(y_train[2]))
#exit ()

# NVIDIA Model
model = Sequential()
model.add(Lambda(lambda x: x/255.0-0.5, input_shape=(160,320,3)))

model.add(Cropping2D(cropping = ((70,25),(0,0))))
model.add(Conv2D(24, (5, 5), strides = (2,2), activation="relu"))
model.add(Conv2D(36, (5, 5), strides = (2,2), activation="relu"))
model.add(Conv2D(48, (5, 5), strides = (2,2), activation="relu"))
model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(Conv2D(64, (3, 3), activation="relu"))

model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

# LeNet
# model = Sequential()
# model.add(Lambda(lambda x: x/255.0-0.5, input_shape=(160,320,3)))
#model.add(Cropping2D(cropping = ((60,0),(0,0))))
#model.add(Convolution2D(6,5,5,activation = 'relu'))
#model.add(MaxPooling2D())
#model.add(Convolution2D(16,5,5,activation = 'relu'))
#model.add(MaxPooling2D())
#model.add(Flatten())
#model.add(Dense(120))
#model.add(Dense(84))
#model.add(Dense(1))

model.compile(optimizer='adam',loss='mse')
model.fit(X_train,y_train,validation_split = 0.2 , shuffle = True, epochs = 5)


# Model 1: 1 layer NN + Lambda (normalization)
# Model 2: Implement LeNet
# Model 3: Augment data by flipping
# Model 4: Use left and right cameras
# Mdoel 5: Cropping 
# Model 6: Implement Nvidia end to end learning 
model.save('model6.h5')


 