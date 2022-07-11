import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D, Input, ReLU, BatchNormalization, Flatten, MaxPool2D
from dc_ldpc import DropConnectDense

# Load MNIST dataset as NumPy arrays
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Preprocess the data
x_train = x_train.reshape(-1, 784).astype('float32') / 255
x_test = x_test.reshape(-1, 784).astype('float32') / 255

X = tf.keras.layers.Input(shape=(784,))
x = DropConnectDense(units=128, prob=0.2, activation="relu", use_bias=True)(X)
x = DropConnectDense(units=64, prob=0.5, activation="relu", use_bias=True)(x)
y = Dense(10, activation="softmax")(x)

model = tf.keras.models.Model(X, y)

# Hyperparameters
batch_size=64
epochs=20

# Compile the model
model.compile(
    optimizer=tf.keras.optimizers.Adam(0.0001),  # Utilize optimizer
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy'])

# Train the network
history = model.fit(
    x_train,
    y_train,
    batch_size=batch_size,
    validation_split=0.1,
    epochs=epochs)

# Evaluate the network
print('Evaluate on test data:')
results = model.evaluate(x_test, y_test, batch_size=128, verbose = 2)
print('Test loss = {0}, Test acc: {1}'.format(results[0], results[1]))