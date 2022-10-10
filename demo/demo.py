import numpy as np
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
from tensorflow.keras.layers import Dense, Dropout, Conv2D, Input, ReLU, BatchNormalization, Flatten, MaxPool2D
import tensorflow.keras.backend as K

from dc_ldpc.dropconnect_ldpc import *
from dropconnect.DClayers_tf import *

import os
abs_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import time

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
x0 = LDPC_DropConnectDense(units=128, prob=0.5, activation="relu", use_bias=True)(X)
x0 = LDPC_DropConnectDense(units=64, prob=0.5, activation="relu", use_bias=True)(x0)
y0 = Dense(10, activation="softmax")(x0)
md0 = tf.keras.models.Model(X, y0)
md0.summary() 

# x1 = LDPC_DropConnectDense(units=128, prob=0.5, activation="relu", use_bias=True)(X)
# x1 = LDPC_DropConnectDense(units=64, prob=1/4, activation="relu", use_bias=True)(x1)
# y1 = Dense(10, activation="softmax")(x1)
# md1 = tf.keras.models.Model(X, y1)
# md1.summary() 

# x2 = LDPC_DropConnectDense(units=128, prob=0.5, activation="relu", use_bias=True)(X)
# x2 = LDPC_DropConnectDense(units=64, prob=1/8, activation="relu", use_bias=True)(x2)
# y2 = Dense(10, activation="softmax")(x2)
# md2 = tf.keras.models.Model(X, y2)
# md2.summary()

# x3 = LDPC_DropConnectDense(units=128, prob=0.5, activation="relu", use_bias=True)(X)
# x3 = LDPC_DropConnectDense(units=64, prob=1/16, activation="relu", use_bias=True)(x3)
# y3 = Dense(10, activation="softmax")(x3)
# md3 = tf.keras.models.Model(X, y3)
# md3.summary()

# x4 = LDPC_DropConnectDense(units=128, prob=1/4, activation="relu", use_bias=True)(X)
# x4 = LDPC_DropConnectDense(units=64, prob=1/4, activation="relu", use_bias=True)(x4)
# y4 = Dense(10, activation="softmax")(x4)
# md4 = tf.keras.models.Model(X, y4)
# md4.summary()

# x5 = LDPC_DropConnectDense(units=128, prob=1/8, activation="relu", use_bias=True)(X)
# x5 = LDPC_DropConnectDense(units=64, prob=1/8, activation="relu", use_bias=True)(x5)
# y5 = Dense(10, activation="softmax")(x5)
# md5 = tf.keras.models.Model(X, y5)
# md5.summary()

# # # Network #2: Test LDPC_DC_Wrapper
# # #     Use LDPC matrix as DropConnect mask, flatten 2D input directly, no convolutional layers used.

# # x2 = LDPC_DropConnect(Dense(128, activation='relu'), prob=0.5)(X)
# # x2 = LDPC_DropConnect(Dense(64, activation='relu'), prob=0.5)(x2)
# # y2 = Dense(10, activation="softmax")(x2)
# # md2= tf.keras.models.Model(X, y2)
# # md2.summary()


# #Network 3:Fully connected network with two similiar hidden dense layers.
# x6 = Dense(128, activation='relu')(X)
# x6 = Dense(64, activation='relu')(x6)
# y6 = Dense(10, activation="softmax")(x6)
# md6= tf.keras.models.Model(X, y6)
# md6.summary()

# # Network 4 - Original DropConnect
# x7 = DropConnectDense(units=128, prob=0.5, activation="relu", use_bias=True)(X)
# x7 = DropConnectDense(units=64, prob=0.5, activation="relu", use_bias=True)(x7)
# y7 = Dense(10, activation="softmax")(x7)
# md7= tf.keras.models.Model(X, y7)
# md7.summary()

# models = [md0, md1, md2, md3, md4, md5, md6, md7]
models = [md0]

""" Compile and evaluate models """
histories = []
test_results = []
# Apply Early Stopping
callback = tf.keras.callbacks.EarlyStopping(min_delta = 0.0001, patience = 10)

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
        epochs=50,
        callbacks=[callback]
        )
    histories.append(history)

    # # Save the models
    # mdsave_path = os.path.join(abs_path, 'saved_models')
    # print('Training completed, save the model.')
    # md.save(mdsave_path + '\\model_%d_%d' %(models.index(md), int(time.time())))
    # print('Model is saved in folder: %s ...' %(mdsave_path))

    # Evaluation
    results = md.evaluate(x_test, y_test, batch_size=64, verbose=1)
    print('Test loss = {0}, Test acc: {1}'.format(results[0], results[1]))
    test_results.append(results)

with open('results.txt', 'w') as f:
    for line in test_results:
        f.write(f"{line}\n")


""" Make some plots """
figsave_path = os.path.join(abs_path, 'plottings')

def results_plot(histories):
    """ 
    :param history: history attribute after model training
    """
    for i in range(len(histories)):
        fig = plt.figure()
		# plot loss
        # plt.subplot(2, 1, 1)
        # plt.title('Cross Entropy Loss')
        epochs:int = range(1, len(histories[i].history['loss']) + 1)
        plt.xlabel('Epochs')
        plt.ylabel('Cross Entropy Loss')
        plt.yticks(np.arange(0.0, 1.0, 0.05))
        plt.ylim(0.0, 1.0)
        plt.plot(histories[i].history['loss'], color='blue')
        plt.plot(histories[i].history['val_loss'], color='orange')
        plt.grid(True)
        plt.legend(['training', 'testing'])
        # # plot accuracy
        # plt.subplot(2, 1, 2)
        # plt.title('Classification Accuracy')
        # plt.xlabel('Epochs')
        # plt.plot(histories[i].history['accuracy'], color='blue', label='train')
        # plt.plot(histories[i].history['val_accuracy'], color='orange', label='test')
        
        # plt.suptitle('Test results of model %d' %(i))
        # plt.suptitle('Test results of model 0')
        # plt.tight_layout()
        # plt.savefig(figsave_path + '\\diagnostics_model_%d_%d.png' %(i, int(time.time())))
        # plt.savefig(figsave_path + '\\diagnostics_model_0_%d_%d.png' %(i, int(time.time())))
        if (i < 6):
            plt.suptitle('Test results of model #1 with config. %d' %(i+1))
            plt.savefig(figsave_path + '\\diagnostics_model_0_%d_%d.png' %(i, int(time.time())))
        else:
            plt.suptitle('Test results of model #%d' %(i-3))
            plt.savefig(figsave_path + '\\diagnostics_model_%d_%d.png' %(i-3, int(time.time())))            
        plt.close(fig)

print('Plot the fitting results...')
results_plot(histories)

