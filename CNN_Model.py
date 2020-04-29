#Importing necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import keras
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.datasets import mnist
from tensorflow.python.keras.layers import Flatten
from tensorflow.python.keras.layers.convolutional import Conv2D
from tensorflow.python.keras.layers.convolutional import MaxPooling2D
from tensorflow.python.keras.utils.np_utils import to_categorical
import random
from tensorflow.python.keras.layers import Dropout
from tensorflow.python.keras.models import Model

np.random.seed(0)

## the data, split between train and test sets
(X_train,y_train),(X_test,y_test)=mnist.load_data()
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)

#The dimension of the training data is (60000,28,28). 
#The CNN model will require one more dimension so we reshape the matrix to shape (60000,28,28,1).
X_train = X_train.reshape(60000, 28, 28, 1)
X_test = X_test.reshape(10000, 28, 28, 1)
 
# convert class vectors to binary class matrices #One Hot Encoding 
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

#Normalization to scale down the images between 0 to 1
X_train = X_train/255
X_test = X_test/255

def leNet_model():
  # create model
  model = Sequential()
  model.add(Conv2D(30, (5, 5), input_shape=(28, 28, 1), activation='relu'))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  
  model.add(Conv2D(15, (3, 3), activation='relu'))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  
  model.add(Flatten())
  model.add(Dense(500, activation='relu'))
  model.add(Dropout(0.5))
  model.add(Dense(num_classes, activation='softmax'))
  # Compile model
  model.compile(Adam(lr = 0.01), loss='categorical_crossentropy', metrics=['accuracy'])
  return model

model=leNet_model()
print(model.summary())

history=model.fit(X_train, y_train, epochs=20,  validation_split = 0.1, batch_size = 400, verbose = 1, shuffle = 1)
print("The model has successfully trained")

model.save('model_Mnist.h5')
print("Saving the model as model_Mnist.h5")

