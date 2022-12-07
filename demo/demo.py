import numpy as np
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, Input, BatchNormalization, Flatten, MaxPool2D
import tensorflow.keras.backend as K

from dc_ldpc.dropconnect_ldpc import *
# from dropconnect.DClayers_tf import *

import os
abs_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import time

from matplotlib import pyplot as plt

""" Load dataset as NumPy arrays """
# MNIST
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
# CIFAR-10
# (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# Preprocess the data
# x_train = x_train.reshape(-1, 784).astype('float32') / 255
# x_test = x_test.reshape(-1, 784).astype('float32') / 255
# X = tf.keras.layers.Input(shape=(784,))
x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255
x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255


""" Build models """
# # Network #1: Test LDPC_DC_Dense
# #     Use LDPC matrix as DropConnect mask, flatten 2D input directly, no convolutional layers used.
# x0 = LDPC_DropConnectDense(units=128, prob=0.5, activation="relu", use_bias=True)(X)
# x0 = LDPC_DropConnectDense(units=64, prob=0.5, activation="relu", use_bias=True)(x0)
# y0 = Dense(10, activation="softmax")(x0)
# md0 = tf.keras.models.Model(X, y0)
# md0.summary() 

# x1 = LDPC_DropConnectDense(units=128, prob=0.5, activation="relu", use_bias=True)(X)
# x1 = LDPC_DropConnectDense(units=64, prob=3/4, activation="relu", use_bias=True)(x1)
# y1 = Dense(10, activation="softmax")(x1)
# md1 = tf.keras.models.Model(X, y1)
# md1.summary() 

# x2 = LDPC_DropConnectDense(units=128, prob=0.5, activation="relu", use_bias=True)(X)
# x2 = LDPC_DropConnectDense(units=64, prob=7/8, activation="relu", use_bias=True)(x2)
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

# # # # Network #2: Test LDPC_DC_Wrapper
# # # #     Use LDPC matrix as DropConnect mask, flatten 2D input directly, no convolutional layers used.

# # # x2 = LDPC_DropConnect(Dense(128, activation='relu'), prob=0.5)(X)
# # # x2 = LDPC_DropConnect(Dense(64, activation='relu'), prob=0.5)(x2)
# # # y2 = Dense(10, activation="softmax")(x2)
# # # md2= tf.keras.models.Model(X, y2)
# # # md2.summary()


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
# models = [md0, md1, md2,md6]

model = Sequential(name="Model1_Config0")
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPool2D((2, 2), strides=2))
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPool2D((2, 2), strides=2))
model.add(Flatten())
model.add(Dense(units=128, activation="relu"))
model.add(LDPC_DropConnectDense(units=32, prob=0.5, activation="relu"))
model.add(Dense(units=10, activation="softmax"))


""" Compile and evaluate models """
histories = []
test_results = []
# Apply Early Stopping
callback = tf.keras.callbacks.EarlyStopping(min_delta = 0.0001, patience = 10)


# TODO: Lookup norm of gradients and weights

# for md in models:
#     md.compile(
#         optimizer=tf.keras.optimizers.Adam(0.0001),
#         loss=tf.keras.losses.SparseCategoricalCrossentropy(),
#         metrics=['accuracy']        
#         )
    

#     history = md.fit(
#         x_train,
#         y_train,
#         batch_size=64,
#         validation_split=0.1,
#         epochs=100,
#         callbacks=[callback]
#         )
#     histories.append(history)

batch_size = 128
epochs = 100

model.compile(
    optimizer=tf.keras.optimizers.Adam(0.0001),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']        
    )
history = model.fit(
    x_train, y_train, 
    batch_size=batch_size, 
    epochs=1, 
    validation_split=0.1,
    callbacks=[callback]
    )
# histories.append(history)
model.summary() 

    # # Save the models
    # mdsave_path = os.path.join(abs_path, 'saved_models')
    # print('Training completed, save the model.')
    # md.save(mdsave_path + '\\model_%d_%d' %(models.index(md), int(time.time())))
    # print('Model is saved in folder: %s ...' %(mdsave_path))

    # Evaluation
#     results = md.evaluate(x_test, y_test, batch_size=64, verbose=1)
#     print('Test loss = {0}, Test acc: {1}'.format(results[0], results[1]))
#     test_results.append(results)

# result_path = 'result_%d.txt' %(int(time.time()))
# with open(result_path, 'w') as f:
#     for line in test_results:
#         f.write(f"{line}\n")

# results = model.evaluate(x_test, y_test, batch_size=64, verbose=1)
# print('Test loss = {0}, Test acc: {1}'.format(results[0], results[1]))
# test_results.append(results)

result_path = 'result_{0}{1}.txt'.format(time.localtime().tm_mon, time.localtime().tm_mday)
# with open(result_path, 'a+') as f:
    # f.write("Evaluation results of %s: \n" %(model.name))
    # for line in test_results:
    #     f.write(f"{line}\n\n")


""" Make some plots """
figsave_path = os.path.join(abs_path, 'plottings')

# def results_plot(histories):
    # """ 
    # :param history: history attribute after model training
    # """
    # for i in range(len(histories)):
    #     fig = plt.figure()
	# 	# plot loss
    #     # plt.subplot(2, 1, 1)
    #     # plt.title('Cross Entropy Loss')
    #     epochs:int = range(1, len(histories[i].history['loss']) + 1)
    #     plt.xlabel('Epochs')
    #     # plt.ylabel('Cross Entropy Loss')
    #     plt.yticks(np.arange(0.0, 1.0, 0.05))
    #     plt.ylim(0.0, 1.0)
    #     plt.plot(histories[i].history['loss'], color='magenta')
    #     plt.plot(histories[i].history['val_loss'], color='blue')
    #     plt.plot(histories[i].history['accuracy'], color='orange', label='train')
    #     plt.plot(histories[i].history['val_accuracy'], color='green', label='test')
    #     plt.grid(True)
    #     plt.legend(['loss', 'val_loss', 'accuracy', 'val_accuracy'])
    #     plt.suptitle('Training results of model #1 with config. #1')
    #     plt.savefig(figsave_path + '\\diagnostics_{0}{1}_{2}.png'.format(time.localtime().tm_mon, time.localtime().tm_mday, int(time.time())))            
        # # plot accuracy
        # plt.cla()
        # plt.suptitle('Training results of model #1 with config. #1')
        # plt.xlabel('Epochs')
        # plt.ylabel('Accuracy')
        # plt.ylabel('Cross Entropy Loss')
        # plt.yticks(np.arange(0.7, 1.0, 0.025))
        # plt.ylim(0.7, 1.0)        
        # plt.plot(histories[i].history['accuracy'], color='blue', label='train')
        # plt.plot(histories[i].history['val_accuracy'], color='orange', label='test')
        # plt.grid(True)
        # plt.savefig(figsave_path + '\\accuracy_{0}{1}_{2}.png'.format(time.localtime().tm_mon, time.localtime().tm_mday, int(time.time())))            
        # plt.close(fig)

        # plt.suptitle('Test results of model %d' %(i))
        # plt.suptitle('Test results of model 0')
        # plt.tight_layout()
        # plt.savefig(figsave_path + '\\diagnostics_model_%d_%d.png' %(i, int(time.time())))
        # plt.savefig(figsave_path + '\\diagnostics_model_0_%d_%d.png' %(i, int(time.time())))
        # if (i < 3):
        #     plt.suptitle('Test results of model #1 with config. %d' %(i+1))
        #     plt.savefig(figsave_path + '\\diagnostics_model_0_%d_%d.png' %(i, int(time.time())))
        # else:
        #     plt.suptitle('Test results of model #%d' %i)
        #     plt.savefig(figsave_path + '\\diagnostics_model_%d_%d.png' %(i-3, int(time.time())))            
        # plt.close(fig)

# print('Plot the fitting results...')
# results_plot(histories)

""" Monitor norm of weights in each layers during training """
from tensorflow.keras.callbacks import Callback
from matplotlib import pyplot as plt


class WeightCapture(Callback):
    def __init__(self):
        super().__init__()
        self.weights = []
        self.epochs = []
    def on_epoch_end(self, epoch, logs=None):
        self.epochs.append(epoch) # remember the epoch axis
        weight = {}
        for layer in model.layers:
            if not layer.weights:
                continue
            name = layer.weights[0].name.split("/")[1]
            # weight[name] = layer.weights[0].numpy()
            weight[name] = layer.get_weights()[0]   # layer.get_weights()[0] returns the kernel, layer.get_weights()[1] returns the bias, dtype=np.ndarray
        self.weights.append(weight)

def plotweight(capture_cb):
    "Plot the weights' mean and s.d. across epochs"
    _, ax = plt.subplots(2, 1, sharex=True, constrained_layout=True)
    ax[0].set_title("Mean of weights")
    ax[1].set_title("Standard Deviation")
    plt.xlabel('Epochs')
    # weight = []
    # for epoch in capture_cb.epochs:
    for key in capture_cb.weights[0]:   # 'key' is the name of layer
        # weight[key].append(w[epoch][key].mean() for w in capture_cb.weights)
        ax[0].plot(capture_cb.epochs, [w[key].mean() for w in capture_cb.weights], label=key)
        ax[1].plot(capture_cb.epochs, [w[key].std() for w in capture_cb.weights], label=key)
    ax[0].legend()
    ax[1].legend()
    plt.xlabel('Epochs')
    plt.show()


# capture_cb = WeightCapture()

model.compile(
    optimizer=tf.keras.optimizers.Adam(0.0001),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']        
    )


model.fit(x_train, y_train, batch_size=batch_size, epochs=1, validation_split=0.1, callbacks=[capture_cb])

model.evaluate(x_test, y_test, batch_size=32)

# plotweight(capture_cb)


""" Monitor norm of gradients in each layers during training """

optimizer = tf.keras.optimizers.Adam(0.0001)
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()

def train_model(X, y, model, n_epochs=epochs, batch_size=batch_size):
    "Run training loop manually"
    train_dataset = tf.data.Dataset.from_tensor_slices((X, y))
    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)

    gradhistory = []
    losshistory = []
    def recordweight():
        data = {}
        for g,w in zip(grads, model.trainable_weights):
            if '/kernel:' not in w.name:
                continue # skip bias
            name = w.name.split("/")[0]
            data[name] = g.numpy()
        gradhistory.append(data)
        losshistory.append(loss_value.numpy())
    for epoch in range(n_epochs):
        for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
            with tf.GradientTape() as tape:
                y_pred = model(x_batch_train, training=True)
                loss_value = loss_fn(y_batch_train, y_pred)

            grads = tape.gradient(loss_value, model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))

            if step == 0:
                recordweight()
    # After all epochs, record again
    recordweight()
    return gradhistory, losshistory

from matplotlib import pyplot as plt 

def plot_gradient(gradhistory, losshistory):
    "Plot gradient mean and sd across epochs"
    _, ax = plt.subplots(3, 1, sharex=False, constrained_layout=True, figsize=(8, 12))
    ax[0].set_title("Gradient mean of weights")
    for key in gradhistory[0]:
        ax[0].plot(range(len(gradhistory)), [w[key].mean() for w in gradhistory], label=key)
    plt.xlabel('Epochs')
    ax[0].legend()
    ax[1].set_title("Standard Deviation of weight gradients")
    for key in gradhistory[0]:
        ax[1].semilogy(range(len(gradhistory)), [w[key].std() for w in gradhistory], label=key)
    plt.xlabel('Epochs')
    ax[1].legend()
    ax[2].set_title("Loss")
    ax[2].plot(range(len(losshistory)), losshistory)
    plt.xlabel('Epochs')
    plt.show()

gradhistory, losshistory = train_model(x_train, y_train, model)
plot_gradient(gradhistory, losshistory)