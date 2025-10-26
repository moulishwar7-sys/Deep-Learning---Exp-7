# Deep-Learning---Exp-7

**Implement an Autoencoder in TensorFlow/Keras**

# AIM

To develop a convolutional autoencoder for image denoising application.

# THEORY

An Autoencoder is an unsupervised neural network that learns to compress input data into a lower-dimensional representation and then reconstruct it back to its original form. It consists of an encoder that reduces the input dimensions and a decoder that rebuilds the input from this compressed data. The model is trained to minimize the reconstruction error between the original and reconstructed data. Autoencoders are widely used for dimensionality reduction, denoising, and anomaly detection tasks.

**Neural Network Model**

<img width="1035" height="205" alt="Screenshot 2025-10-26 212708" src="https://github.com/user-attachments/assets/8b55b67f-8f19-4134-80ce-1529883a764a" />


# DESIGN STEPS

STEP 1: Import the necessary libraries and dataset.

STEP 2: Load the dataset and scale the values for easier computation.

STEP 3:** Add noise to the images randomly for both the train and test sets.

STEP 4: Build the Neural Model using Convolutional Layer Pooling Layer Up Sampling Layer. Make sure the input shape and output shape of the model are identical.

STEP 5: Pass test data for validating manually.

STEP 6: Pass test data for validating manually.


# Name: Moulishwar G

# Register Number: 2305001020

# PROGRAM


```
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import utils
from tensorflow.keras import models
from tensorflow.keras.datasets import mnist
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

(x_train, _), (x_test, _) = mnist.load_data()
x_train.shape
x_train_scaled = x_train.astype('float32') / 255.
x_test_scaled = x_test.astype('float32') / 255.
x_train_scaled = np.reshape(x_train_scaled, (len(x_train_scaled), 28, 28, 1))
x_test_scaled = np.reshape(x_test_scaled, (len(x_test_scaled), 28, 28, 1))

noise_factor = 0.5
x_train_noisy = x_train_scaled + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train_scaled.shape)
x_test_noisy = x_test_scaled + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test_scaled.shape)

x_train_noisy = np.clip(x_train_noisy, 0., 1.)
x_test_noisy = np.clip(x_test_noisy, 0., 1.)

n = 10
plt.figure(figsize=(20, 2))
for i in range(1, n + 1):
    ax = plt.subplot(1, n, i)
    plt.imshow(x_test_noisy[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()

input_img = keras.Input(shape=(28, 28, 1))

x=layers.Conv2D(16,(5,5),activation='relu',padding='same')(input_img)
x=layers.MaxPooling2D((2,2),padding='same')(x)
x=layers.Conv2D(4,(3,3),activation='relu',padding='same')(x)
x=layers.MaxPooling2D((2,2),padding='same')(x)
x=layers.Conv2D(4,(3,3),activation='relu',padding='same')(x)
x=layers.MaxPooling2D((2,2),padding='same')(x)
x=layers.Conv2D(8,(7,7),activation='relu',padding='same')(x)
encoded = layers.MaxPooling2D((2, 2), padding='same')(x)

x=layers.Conv2D(4,(3,3),activation='relu',padding='same')(encoded)
x=layers.UpSampling2D((2,2))(x)
x=layers.Conv2D(4,(3,3),activation='relu',padding='same')(x)
x=layers.UpSampling2D((2,2))(x)
x=layers.Conv2D(8,(5,5),activation='relu',padding='same')(x)
x=layers.UpSampling2D((2,2))(x)
x=layers.Conv2D(16,(5,5),activation='relu',padding='same')(x)
x=layers.UpSampling2D((2,2))(x)
x=layers.Conv2D(16,(5,5),activation='relu')(x)
x=layers.UpSampling2D((1,1))(x)
decoded = layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

autoencoder = keras.Model(input_img, decoded)

autoencoder.summary()

autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

autoencoder.fit(x_train_noisy, x_train_scaled,
                epochs=3,
                batch_size=256,
                shuffle=True,
                validation_data=(x_test_noisy, x_test_scaled))

print("Rithiga Sri.B 212221230083")
metrics = pd.DataFrame(autoencoder.history.history)
metrics[['loss','val_loss']].plot()

decoded_imgs = autoencoder.predict(x_test_noisy)
print("Rithiga Sri.B 212221230083")
n = 10
plt.figure(figsize=(20, 4))
for i in range(1, n + 1):
    # Display original
    ax = plt.subplot(3, n, i)
    plt.imshow(x_test_scaled[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # Display noisy
    ax = plt.subplot(3, n, i+n)
    plt.imshow(x_test_noisy[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # Display reconstruction
    ax = plt.subplot(3, n, i + 2*n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
```

# OUTPUT

**Model Summary**

<img width="578" height="651" alt="Screenshot 2025-10-26 213314" src="https://github.com/user-attachments/assets/e21015a3-7cc4-432c-af99-b48f9cfc33ca" />


**Training loss**


<img width="968" height="716" alt="image" src="https://github.com/user-attachments/assets/1a4454a3-e5c0-44a3-84f6-13d832cac4d8" />


**Original vs Noisy Vs Reconstructed Image**


<img width="1033" height="253" alt="Screenshot 2025-10-26 213051" src="https://github.com/user-attachments/assets/5b937256-fae2-4521-88b0-c7fc1e5f5253" />


**RESULT**

Thus we have successfully developed a convolutional autoencoder for image denoising application.
