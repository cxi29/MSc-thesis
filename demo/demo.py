import numpy as np
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
from tensorflow.keras.layers import Dense, Dropout, Conv2D, Input, ReLU, BatchNormalization, Flatten, MaxPool2D
import tensorflow.keras.backend as K

from dc_ldpc.dropconnect_ldpc import *
from dropconnect.DClayers_tf import *

import os
abs_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from matplotlib import pyplot as plt

# Load MNIST dataset as NumPy arrays
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Preprocess the data
x_train = x_train.reshape(-1, 784).astype('float32') / 255
x_test = x_test.reshape(-1, 784).astype('float32') / 255
X = tf.keras.layers.Input(shape=(784,))


""" Build models """
# Network #1: Test LDPC_DC_Dense
#     Use LDPC matrix as DropConnect mask, flatten 2D input directly, no convolutional layers used.
x1 = LDPC_DropConnectDense(units=128, prob=0.5, activation="relu", use_bias=True)(X)
x1 = LDPC_DropConnectDense(units=64, prob=0.5, activation="relu", use_bias=True)(x1)
y1 = Dense(10, activation="softmax")(x1)
md1 = tf.keras.models.Model(X, y1)
md1.summary()


# Network #2: Test LDPC_DC_Wrapper
#     Use LDPC matrix as DropConnect mask, flatten 2D input directly, no convolutional layers used.

x2 = LDPC_DropConnect(Dense(128, activation='relu'), prob=0.5)(X)
x2 = LDPC_DropConnect(Dense(64, activation='relu'), prob=0.5)(x2)
y2 = Dense(10, activation="softmax")(x2)
md2= tf.keras.models.Model(X, y2)
md2.summary()


#Network 3:Fully connected network with two similiar hidden dense layers.
x3 = Dense(128, activation='relu')(X)
x3 = Dense(64, activation='relu')(x3)
y3 = Dense(10, activation="softmax")(x3)
md3= tf.keras.models.Model(X, y3)
md3.summary()

# Network 4 - Original DropConnect
x4 = DropConnectDense(units=128, prob=0.4, activation="relu", use_bias=True)(X)
x4 = DropConnectDense(units=64, prob=0.5, activation="relu", use_bias=True)(x4)
y4 = Dense(10, activation="softmax")(x4)
md4= tf.keras.models.Model(X, y4)
md4.summary()

models = [md1, md2, md3, md4]

""" Compile and evaluate models """
histories = []
test_results = []

for md in models:
    md.compile(
        optimizer=tf.keras.optimizers.Adam(0.0001),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy']        
        )
    history = md.fit(
        x_train,
        y_train,
        batch_size=64,
        validation_split=0.1,
        epochs=100
        )
    histories.append(history)

    # Save the models
    mdsave_path = os.path.join(abs_path, 'saved_models')
    print('Training completed, save the model.')
    md.save(mdsave_path + '\\model_%d' %(models.index(md) + 1))
    print('Model is saved in folder: %s ...' %(mdsave_path))

    # Evaluation
    results = md.evaluate(x_test, y_test, batch_size=128, verbose=2)
    print('Test loss = {0}, Test acc: {1}'.format(results[0], results[1]))
    test_results.append(results)


""" Make some plots """
figsave_path = os.path.join(abs_path, 'plottings')

def results_plot(histories):
    """ 
    :param history: history attribute after model training
    """
    for i in range(len(histories)):
		# plot loss
        plt.subplot(2, 1, 1)
        plt.title('Cross Entropy Loss')
        epochs:int = range(1, len(histories[i].history['loss']) + 1)
        plt.xlabel('Epochs')
        plt.plot(histories[i].history['loss'], color='blue', label='train')
        plt.plot(histories[i].history['val_loss'], color='orange', label='test')
        # plot accuracy
        plt.subplot(2, 1, 2)
        plt.title('Classification Accuracy')
        plt.xlabel('Epochs')
        plt.plot(histories[i].history['accuracy'], color='blue', label='train')
        plt.plot(histories[i].history['val_accuracy'], color='orange', label='test')
        
        plt.suptitle('Test results of model %d' %(i))
        plt.savefig(figsave_path + '\\diagnostics_model_%d.png' %(i))

print('Plot the fitting results...')
results_plot(histories)
