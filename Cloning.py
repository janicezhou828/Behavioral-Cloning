import csv
import cv2
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

lines = []
with open ('./Data/Round1/driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		lines.append (line)
images = []
measurements = []

for line in lines:
	source_path = line[0]
	tokens = source_path.split('\\')
	filename = tokens[-1]
	local_path = "./Data/Round1/IMG/" + filename
	image = cv2.imread(local_path)
	images.append(image)
	measurement = line[3]
	measurements.append(measurement)

#print (len(images))

	
X_train = np.array(images)
y_train_st = np.array (measurements)
y_train = y_train_st.astype(np.float)

#print (X_train.shape)
#print (y_train)
#print (y_train.shape)


model = Sequential()
model.add(Lambda(lambda x: x/255.0-0.5, input_shape=(160,320,3)))


model.add(Convolution2D(6,5,5,activation = 'relu'))
model.add(MaxPooling2D())
model.add(Convolution2D(16,5,5,activation = 'relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(120))
model.add(Dense(84))
model.add(Dense(1))

model.compile(optimizer='adam',loss='mse')
model.fit(X_train,y_train,validation_split = 0.2 , shuffle = True, epochs = 5)

# Test model: 1 layer NN
# Model 1: 1 layer NN + Lambda (normalization)
# Model 2: Implement LeNet
# Model 3: 
model.save('model3.h5')


