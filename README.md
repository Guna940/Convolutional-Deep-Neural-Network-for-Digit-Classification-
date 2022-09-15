# Ex-03-Convolutional Deep Neural Network for Digit Classification

## AIM

To Develop a convolutional deep neural network for digit classification and to verify the response for scanned handwritten images.

## Problem Statement and Dataset
Problem Statement:-

The handwritten digit recognition is the capability of computer applications to recognize the human handwritten digits. It is a hard task for the machine because handwritten digits are not perfect and can be made with many different shapes and sizes. The handwritten digit recognition system is a way to tackle this
problem which uses the image of a digit and recognizes the digit present in the image. Convolutional Neural Network model created using PyTorch library over the MNIST dataset to recognize handwritten digits .Handwritten Digit Recognition is the capability of a computer to fete the mortal handwritten integers from different sources like images, papers, touch defenses, etc, and classify.  them into 10 predefined classes (0-9). This has been a  Content of bottomless- exploration in the field of deep literacy. 

Dataset:-

The dataset that is being used here is the MNIST digits classification dataset. Keras is a deep learning API written in Python and MNIST is a dataset provided by this API. This dataset consists of 60,000 training images and 10,000 testing images. It is a decent dataset for individuals who need to have a go at pattern recognition as we will perform in just a minute.When the Keras API is called, there are four values returned namely- x_train, y_train, x_test, and y_test. Do not worry, I will walk you through this.

## Neural Network Model

Include the neural network model diagram.

## DESIGN STEPS

### STEP 1:
  Import the libraries and load the dataset.
### STEP 2:
  Preprocess the data.
### STEP 3:
  Create the model and train the model with the dataset and seprate the data into training and testing.
### STEP 4:
  Evaluate the model by using and create CNN with the help of libraries and load the model with appropriate data.
### STEP 5:
  Create GUI to predict digits using hand written images with help of before created model.
## PROGRAM
```python3
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import utils
import pandas as pd
from sklearn.metrics import classification_report,confusion_matrix
from tensorflow.keras.preprocessing import image
```
```python3
(X_train, y_train), (X_test, y_test) = mnist.load_data()
single_image= X_train[2]
plt.imshow(single_image,cmap='gray')
X_train.min()
X_train.max()
X_train_scaled = X_train/255.0
X_test_scaled = X_test/255.0
y_train[2]
y_train_onehot = utils.to_categorical(y_train,10)
y_test_onehot = utils.to_categorical(y_test,10)
type(y_train_onehot)
y_train_onehot.shape
single_image = X_train[600]
plt.imshow(single_image,cmap='gray')
y_train_onehot[600]
X_train_scaled = X_train_scaled.reshape(-1,28,28,1)
X_test_scaled = X_test_scaled.reshape(-1,28,28,1)
```
```python3
model=keras.Sequential()
layer1=layers.Conv2D(filters=25, kernel_size=(3, 3), activation='relu', input_shape=(28,28,1))
model.add(layer1)
layer2=layers.MaxPooling2D((2, 2))
model.add(layer2)
layer3=layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu')
model.add(layer3)
layer4=layers.MaxPooling2D((2, 2))
model.add(layer4)
layer5=layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu')
model.add(layer5)
layer6=layers.MaxPooling2D((2, 2))
model.add(layer6)
layer7=layers.Flatten()
model.add(layer7)
layer8=layers.Dense(64, activation='relu')
model.add(layer8)
layer9=layers.Dense(10, activation='softmax')
model.add(layer9)
model.summary()
```
```python3
model.compile(optimizer='adam',
                 loss='categorical_crossentropy',
                 metrics=['accuracy'])
model.fit(X_train_scaled ,y_train_onehot, epochs=5,
          batch_size=64, 
          validation_data=(X_test_scaled,y_test_onehot))
metrics = pd.DataFrame(model.history.history)
metrics.head()
metrics[['accuracy','val_accuracy']].plot()
metrics[['loss','val_loss']].plot()
x_test_predictions = np.argmax(model.predict(X_test_scaled), axis=1)
print(confusion_matrix(y_test,x_test_predictions))
print(classification_report(y_test,x_test_predictions))
```
```python3
img = image.load_img('image1.jpeg')
img_tensor = tf.convert_to_tensor(np.asarray(img))
img_28 = tf.image.resize(img_tensor,(28,28))
img_28_gray = tf.image.rgb_to_grayscale(img_28)
img_28_gray_scaled = img_28_gray.numpy()/255.0
x_single_prediction = np.argmax(
    model.predict(img_28_gray_scaled.reshape(1,28,28,1)),
     axis=1)
plt.imshow(img_28_gray_scaled.reshape(28,28),cmap='gray')
img_28_gray_inverted = 255.0-img_28_gray
img_28_gray_inverted_scaled = img_28_gray_inverted.numpy()/255.0
x_single_prediction = np.argmax(
    model.predict(img_28_gray_inverted_scaled.reshape(1,28,28,1)),
     axis=1)
print(x_single_prediction)
```
## OUTPUT

### Training Loss, Validation Loss Vs Iteration Plot

![Screenshot (30)](https://user-images.githubusercontent.com/89703145/190477865-478b66bc-7dab-400b-989b-521118c41a8b.png)         ![Screenshot (31)](https://user-images.githubusercontent.com/89703145/190477940-54f30b9e-0a47-4a91-8504-6750687ddc7e.png)


### Classification Report

![Screenshot (29)](https://user-images.githubusercontent.com/89703145/190478449-47ee0a10-c13e-43da-a5bd-310012b3e1de.png)

### Confusion Matrix

![Screenshot (28)](https://user-images.githubusercontent.com/89703145/190478493-b352c458-79d5-45ae-b04f-c9682c6e6a8c.png)

### New Sample Data Prediction
![Screenshot (33)](https://user-images.githubusercontent.com/89703145/190479534-0f6c49ff-57b1-49b3-9f71-727adeedae64.png)   ![Screenshot (32)](https://user-images.githubusercontent.com/89703145/190478638-765900a3-e203-42ec-9640-4cbc8c2994cc.png)

## RESULT
Therefore,hence constructed a Convolutional Neural Network model for Digit Classification.
