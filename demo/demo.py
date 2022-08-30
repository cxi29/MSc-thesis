import numpy as np
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
from tensorflow.keras.layers import Dense, Dropout, Conv2D, Input, ReLU, BatchNormalization, Flatten, MaxPool2D
import tensorflow.keras.backend as K

from dc_ldpc.dropconnect_ldpc import *
from dropconnect.DClayers_tf import *
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
    results = md.evaluate(x_test, y_test, batch_size=128, verbose=2)
    print('Test loss = {0}, Test acc: {1}'.format(results[0], results[1]))
    test_results.append(results)

# """ Compile the models """
# md1.compile(
#     optimizer=tf.keras.optimizers.Adam(0.0001),
#     loss=tf.keras.losses.SparseCategoricalCrossentropy(),
#     metrics=['accuracy'])

# md2.compile(
#     optimizer=tf.keras.optimizers.Adam(0.0001),
#     loss=tf.keras.losses.SparseCategoricalCrossentropy(),
#     metrics=['accuracy'])

# md3.compile(
#     optimizer=tf.keras.optimizers.Adam(0.0001),
#     loss=tf.keras.losses.SparseCategoricalCrossentropy(),
#     metrics=['accuracy'])

# md4.compile(
#     optimizer=tf.keras.optimizers.Adam(0.0001),
#     loss=tf.keras.losses.SparseCategoricalCrossentropy(),
#     metrics=['accuracy'])



# """ Network training and evaluation"""
# hist1 = md1.fit(
#     x_train,
#     y_train,
#     batch_size=64,
#     validation_split=0.1,
#     epochs=100)
# print('Evaluate model 1 on test data:')
# results_1 = md1.evaluate(x_test, y_test, batch_size=128, verbose=2)
# print('Test loss = {0}, Test acc: {1}'.format(results_1[0], results_1[1]))

# hist2 = md2.fit(
#     x_train,
#     y_train,
#     batch_size=64,
#     validation_split=0.1,
#     epochs=100)
# print('Evaluate model 2 on test data:')
# results_2 = md2.evaluate(x_test, y_test, batch_size=128, verbose=2)
# print('Test loss = {0}, Test acc: {1}'.format(results_2[0], results_2[1]))

# hist3 = md3.fit(
#     x_train,
#     y_train,
#     batch_size=64,
#     validation_split=0.1,
#     epochs=100)
# print('Evaluate model 3 on test data:')
# results_3 = md3.evaluate(x_test, y_test, batch_size=128, verbose=2)
# print('Test loss = {0}, Test acc: {1}'.format(results_3[0], results_3[1]))

# hist4 = md4.fit(
#     x_train,
#     y_train,
#     batch_size=64,
#     validation_split=0.1,
#     epochs=100)
# print('Evaluate model 4 on test data:')
# results_4 = md4.evaluate(x_test, y_test, batch_size=128, verbose=2)
# print('Test loss = {0}, Test acc: {1}'.format(results_4[0], results_4[1]))

# def evaluate_model(dataX, dataY, n_folds=5):
# 	scores, histories = list(), list()
# 	# prepare cross validation
# 	kfold = KFold(n_folds, shuffle=True, random_state=1)
# 	# enumerate splits
# 	for train_ix, test_ix in kfold.split(dataX):
# 		# define model
# 		model = define_model()
# 		# select rows for train and test
# 		trainX, trainY, testX, testY = dataX[train_ix], dataY[train_ix], dataX[test_ix], dataY[test_ix]
# 		# fit model
# 		history = model.fit(trainX, trainY, epochs=10, batch_size=32, validation_data=(testX, testY), verbose=0)
# 		# evaluate model
# 		_, acc = model.evaluate(testX, testY, verbose=0)
# 		print('> %.3f' % (acc * 100.0))
# 		# stores scores
# 		scores.append(acc)
# 		histories.append(history)
# 	return scores, histories

""" Make some plots """
import os
abs_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
save_path = os.path.join(abs_path, 'plottings')

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
        plt.savefig(save_path + '\\diagnostics_model_%d.png' %(i))

# loss = history.history['loss']
# val_loss = history.history['val_loss']
# epochs:int = range(1, len(loss) + 1)
# plt.plot(epochs, loss, 'y', label='Training loss')
# plt.plot(epochs, val_loss, 'r', label='Validation loss')
# plt.title('Training and validation loss - MNIST with DropConnect')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
# plt.show()