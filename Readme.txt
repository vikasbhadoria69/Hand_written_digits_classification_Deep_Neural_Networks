In this project, I am going to implement a handwritten digit recognition app using the MNIST dataset. We will be using a special type of deep neural network that is Convolutional Neural Networks. In the end, I am going to build a GUI in which you can draw the digit and recognize it straight away.

What is Handwritten Digit Recognition?
The handwritten digit recognition is the ability of computers to recognize human handwritten digits. It is a hard task for the machine because handwritten digits are not perfect and can be made with many different flavors. The handwritten digit recognition is the solution to this problem which uses the image of a digit and recognizes the digit present in the image.

Install the necessary libraries for this project using this command:
pip install numpy, tensorflow, keras, pillow.

The MNIST dataset
This is probably one of the most popular datasets among machine learning and deep learning enthusiasts. The MNIST dataset contains 60,000 training images of handwritten digits from zero to nine and 10,000 images for testing. So, the MNIST dataset has 10 different classes. The handwritten digits images are represented as a 28×28 matrix where each cell contains grayscale pixel value.

Step 1: First, I am going to import all the modules that we are going to need for training our model. The Keras library already contains some datasets and MNIST is one of them. So I can easily import the dataset and start working with it. The mnist.load_data() method returns us the training data, its labels and also the testing data and its labels.

Step 2: The image data cannot be fed directly into the model so we need to perform some operations and process the data to make it ready for our neural network. The dimension of the training data is (60000,28,28). The CNN model will require one more dimension so we reshape the matrix to shape (60000,28,28,1).

Step 3: Now I will create our CNN model in Python data science project. A CNN model generally consists of convolutional and pooling layers. It works better for data that are represented as grid structures, this is the reason why CNN works well for image classification problems. The dropout layer is used to deactivate some of the neurons and while training, it reduces offer fitting of the model. I will then compile the model with the Adam optimizer.

Step 4: The model.fit() function of Keras will start the training of the model. It takes the training data, validation data, epochs, and batch size.

It takes some time to train the model. After training, I save the weights and model definition in the ‘model_Mnist.h5’ file.

Step 5: Create GUI to predict digits
Now for the GUI, I have created a new file in which I build an interactive window to draw digits on canvas and with a button, I can recognize the digit. The Tkinter library comes in the Python standard library. I have created a function predict_digit() that takes the image as input and then uses the trained model to predict the digit.

Then I create the App class which is responsible for building the GUI for our app. I create a canvas where we can draw by capturing the mouse event and with a button, we trigger the predict_digit() function and display the results.