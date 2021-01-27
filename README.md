# Hand written digits classification using Deep Neural Networks
* Built a Python deep learning project on handwritten digit recognition app.
* Built and trained the Convolutional neural network which is very effective(98% accuracy) for image classification purposes.
* Built the GUI where one can draw a digit on the canvas then we classify the digit and show the results.

## What is Handwritten Digit Recognition?
The handwritten digit recognition is the ability of computers to recognize human handwritten digits. It is a hard task for the machine because handwritten digits are not perfect and can be made with many different flavors. The handwritten digit recognition is the solution to this problem which uses the image of a digit and recognizes the digit present in the image.

## Code and Resources Used
**Python Version:** 3.7
**Packages:** pip install numpy, tensorflow, keras, pillow.

## The MNIST dataset
This is probably one of the most popular datasets among machine learning and deep learning enthusiasts. The MNIST dataset contains 60,000 training images of handwritten digits from zero to nine and 10,000 images for testing. So, the MNIST dataset has 10 different classes. The handwritten digits images are represented as a 28×28 matrix where each cell contains grayscale pixel value.

## Methodology
* **Step 1:** Imported all the modules that are needed for training the model. The Keras library already contains some datasets and MNIST is one of them. So one can easily import the dataset and start working with it. The mnist.load_data() method returns us the training data, its labels and also the testing data and its labels.

* **Step 2:** The image data cannot be fed directly into the model so need to perform some operations and process the data to make it ready for our neural network. The dimension of the training data is (60000,28,28). The CNN model will require one more dimension so we reshape the matrix to shape (60000,28,28,1).

* **Step 3:** CreateD CNN model, a CNN model generally consists of convolutional and pooling layers. It works better for data that are represented as grid structures, this is the reason why CNN works well for image classification problems. The dropout layer is used to deactivate some of the neurons and while training, it reduces over fitting of the model. Then compild the model with the Adam optimizer.

* **Step 4:** The model.fit() function of Keras will start the training of the model. It takes the training data, validation data, epochs, and batch size.

It takes some time to train the model. After training, the weights and model definition in the ‘model_Mnist.h5’ file are been saved.

* **Step 5:** Create GUI to predict digits
Now for the GUI, created a new file in which an interactive window to draw digits on canvas and with a button has been build. The Tkinter library comes in the Python standard library. Created a function predict_digit() that takes the image as input and then uses the trained model to predict the digit.

* **Step 6:** Then created the App class which is responsible for building the GUI for our app. Created a canvas where one can draw by capturing the mouse event and with a button, trigger the predict_digit() function and display the results.

## Results

* **Input that can be drawn by hand**

![alt text](https://github.com/vikasbhadoria69/Hand_written_digits_classification_Deep_Neural_Networks/blob/master/Images/Screenshot%202021-01-27%20012436.png)

* **Output that the model has predicted**

![alt text](https://github.com/vikasbhadoria69/Hand_written_digits_classification_Deep_Neural_Networks/blob/master/Images/Screenshot%202021-01-27%20012507.png)

## Codes for the project
***CNN_Model.py*** notebook contains the backend model that is used to predict.
***GUI.py*** notebook contains code for developing the GUI interface. 
