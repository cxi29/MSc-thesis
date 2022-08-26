import numpy as np
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
from tensorflow.keras.layers import Dense, Dropout, Conv2D, Input, ReLU, BatchNormalization, Flatten, MaxPool2D
import tensorflow.keras.backend as K

# import sys, os.path
# abs_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# sys.path.append(abs_path)
# module_path1 = os.path.join(abs_path, 'dc_ldpc')
# module_path2 = os.path.join(abs_path, 'dropconnect')
# sys.path.append(module_path1)
# sys.path.append(module_path2)

from dc_ldpc.dropconnect_ldpc import *
from dropconnect.DClayers_tf import *

from matplotlib import pyplot as plt

# Load MNIST dataset as NumPy arrays
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Preprocess the data
x_train = x_train.reshape(-1, 784).astype('float32') / 255
x_test = x_test.reshape(-1, 784).astype('float32') / 255

X = tf.keras.layers.Input(shape=(784,))
# # Network 1
# x = LDPC_DropConnectDense(units=128, prob=0.5, activation="relu", use_bias=True)(X)
# x = LDPC_DropConnectDense(units=64, prob=0.5, activation="relu", use_bias=True)(x)
# # Network 2
# x = LDPC_DropConnect(Dense(128, activation='relu'), prob=0.5)(X)
# x = LDPC_DropConnect(Dense(64, activation='relu'), prob=0.5)(x)
# # Network 3 - fully connected  
# x = Dense(128, activation='relu')(X)
# x = Dense(64, activation='relu')(x)
# Network 4 - DropConnect
x = DropConnectDense(units=128, prob=0.5, activation="relu", use_bias=True)(X)
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

# Plot
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss - MNIST with DropConnect')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()