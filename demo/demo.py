import numpy as np
import tensorflow as tf
# tf.compat.v1.disable_eager_execution()
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, Input, BatchNormalization, Flatten, MaxPool2D
import tensorflow.keras.backend as K

from dc_ldpc.dropconnect_ldpc import *
from dropconnect.dropconnect_tensorflow import *
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

model = Sequential(name="Model0")
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPool2D((2, 2), strides=2))
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPool2D((2, 2), strides=2))
model.add(Flatten())
model.add(Dense(units=128, activation="relu"))
model.add(Dense(units=64, activation="relu"))
model.add(Dense(units=10, activation="softmax"))

model10 = Sequential(name="Model1_Config0")
model10.add(Conv2D(32, (3, 3), activation='relu'))
model10.add(MaxPool2D((2, 2), strides=2))
model10.add(Conv2D(64, (3,3), activation='relu'))
model10.add(MaxPool2D((2, 2), strides=2))
model10.add(Flatten())
model10.add(Dense(units=128, activation="relu"))
model10.add(LDPC_DropConnectDense(units=64, prob=0.5, activation="relu"))
model10.add(Dense(units=10, activation="softmax"))

model11 = Sequential(name="Model1_Config1")
model11.add(Conv2D(32, (3, 3), activation='relu'))
model11.add(MaxPool2D((2, 2), strides=2))
model11.add(Conv2D(64, (3,3), activation='relu'))
model11.add(MaxPool2D((2, 2), strides=2))
model11.add(Flatten())
model11.add(Dense(units=128, activation="relu"))
model11.add(LDPC_DropConnectDense(units=64, prob=0.125, activation="relu"))
model11.add(Dense(units=10, activation="softmax"))

model12 = Sequential(name="Model1_Config2")
model12.add(Conv2D(32, (3, 3), activation='relu'))
model12.add(MaxPool2D((2, 2), strides=2))
model12.add(Conv2D(64, (3,3), activation='relu'))
model12.add(MaxPool2D((2, 2), strides=2))
model12.add(Flatten())
model12.add(Dense(units=128, activation="relu"))
model12.add(LDPC_DropConnectDense(units=64, prob=0.25, activation="relu"))
model12.add(Dense(units=10, activation="softmax"))

model13 = Sequential(name="Model1_Config3")
model13.add(Conv2D(32, (3, 3), activation='relu'))
model13.add(MaxPool2D((2, 2), strides=2))
model13.add(Conv2D(64, (3,3), activation='relu'))
model13.add(MaxPool2D((2, 2), strides=2))
model13.add(Flatten())
model13.add(Dense(units=128, activation="relu"))
model13.add(LDPC_DropConnectDense(units=64, prob=0.75, activation="relu"))
model13.add(Dense(units=10, activation="softmax"))

model20 = Sequential(name="Model2_Config0")
model20.add(Conv2D(32, (3, 3), activation='relu'))
model20.add(MaxPool2D((2, 2), strides=2))
model20.add(Conv2D(64, (3,3), activation='relu'))
model20.add(MaxPool2D((2, 2), strides=2))
model20.add(Flatten())
model20.add(Dense(units=128, activation="relu"))
model20.add(DropConnectDense(units=32, prob=0.5, activation="relu"))
model20.add(Dense(units=10, activation="softmax"))

model21 = Sequential(name="Model2_Config1")
model21.add(Conv2D(32, (3, 3), activation='relu'))
model21.add(MaxPool2D((2, 2), strides=2))
model21.add(Conv2D(64, (3,3), activation='relu'))
model21.add(MaxPool2D((2, 2), strides=2))
model21.add(Flatten())
model21.add(Dense(units=128, activation="relu"))
model21.add(DropConnectDense(units=64, prob=0.125, activation="relu"))
model21.add(Dense(units=10, activation="softmax"))

model22 = Sequential(name="Model2_Config2")
model22.add(Conv2D(32, (3, 3), activation='relu'))
model22.add(MaxPool2D((2, 2), strides=2))
model22.add(Conv2D(64, (3,3), activation='relu'))
model22.add(MaxPool2D((2, 2), strides=2))
model22.add(Flatten())
model22.add(Dense(units=128, activation="relu"))
model22.add(DropConnectDense(units=64, prob=0.25, activation="relu"))
model22.add(Dense(units=10, activation="softmax"))

model23 = Sequential(name="Model2_Config3")
model23.add(Conv2D(32, (3, 3), activation='relu'))
model23.add(MaxPool2D((2, 2), strides=2))
model23.add(Conv2D(64, (3,3), activation='relu'))
model23.add(MaxPool2D((2, 2), strides=2))
model23.add(Flatten())
model23.add(Dense(units=128, activation="relu"))
model23.add(DropConnectDense(units=64, prob=0.75, activation="relu"))
model23.add(Dense(units=10, activation="softmax"))

# Add the models you want to train here
models = [model, model10, model11, model12, model13, model20, model21, model22, model23]


""" Compile and evaluate models """
histories = []
test_results = []
# # Apply Early Stopping
# callback = tf.keras.callbacks.EarlyStopping(min_delta = 0.001, patience = 10)

batch_size = 128
epochs = 30

# for md in models:
#     md.compile(
#         optimizer=tf.keras.optimizers.Adam(0.0001),
#         loss=tf.keras.losses.SparseCategoricalCrossentropy(),
#         metrics=['accuracy']        
#         )
    

#     history = md.fit(
#         x_train,
#         y_train,
#         batch_size=batch_size,
#         validation_split=0.1,
#         epochs=epochs,
#         callbacks=[callback]
#         )
#     histories.append(history)

#     # Evaluation
#     results = md.evaluate(x_test, y_test, batch_size=64, verbose=1)
#     print('Test loss = {0}, Test acc: {1}'.format(results[0], results[1]))
#     test_results.append(results)

# # compile() and fit() for single model
# model.compile(
#     optimizer=tf.keras.optimizers.Adam(0.0001),
#     loss=tf.keras.losses.SparseCategoricalCrossentropy(),
#     metrics=['accuracy']        
#     )
# history = model.fit(
#     x_train, y_train, 
#     batch_size=batch_size, 
#     epochs=1, 
#     validation_split=0.1,
#     callbacks=[callback]
#     )

# model.summary() 

""" Save the models """
# mdsave_path = os.path.join(abs_path, 'saved_models')
# print('Training completed, save the model.')
# md.save(mdsave_path + '\\model_%d_%d' %(models.index(md), int(time.time())))
# print('Model is saved in folder: %s ...' %(mdsave_path))



# result_path = 'result_%d.txt' %(int(time.time()))
# with open(result_path, 'w') as f:
#     for line in test_results:
#         f.write(f"{line}\n")

# results = model.evaluate(x_test, y_test, batch_size=64, verbose=1)
# print('Test loss = {0}, Test acc: {1}'.format(results[0], results[1]))
# test_results.append(results)

# result_path = 'result_{0}{1}.txt'.format(time.localtime().tm_mon, time.localtime().tm_mday)
# with open(result_path, 'a+') as f:
#     f.write("Evaluation results of %s: \n" %(model.name))
#     for line in test_results:
#         f.write(f"{line}\n\n")


""" Make some plots """
figsave_path = os.path.join(abs_path, 'plottings')

# def results_plot(histories):
#     """ 
#     :param history: history attribute after model training
#     """
#     fig = plt.figure()
#     # plot loss
#     # plt.subplot(2, 1, 1)
#     # plt.title('Cross Entropy Loss')
#     plt.xlabel('epoch')
#     plt.ylabel('val_loss')
#     # plt.yticks(np.arange(0.0, 1.0, 0.05))
#     # plt.ylim(0.0, 1.0)
#     plt.yscale("log")
#     for i in range(len(histories)):
#     # plt.plot(histories[i].history['loss'], color='magenta')
#     # plt.plot(histories[i].history['val_loss'], color='blue')
#     # plt.plot(histories[i].history['accuracy'], color='orange', label='train')
#     # plt.plot(histories[i].history['val_accuracy'], color='green', label='test')
#         epochs:int = range(1, len(histories[i].history['loss']) + 1)
#         plt.plot(epochs, histories[i].history['val_loss'])
#     plt.grid(True)
#     # plt.legend(['loss', 'val_loss', 'accuracy', 'val_accuracy'])
#     plt.legend(['Model0', 'Model1_config0', 'Model1_config1', 'Model1_config2', 'Model1_config3', 'Model2_config0', 'Model2_config1', 'Model2_config2', 'Model2_config3'])
#     plt.title('Empirical loss of SNNs with early stopping')
#     plt.tight_layout()
#     plt.savefig(figsave_path + '\\diagnostics_{0}{1}_{2}.png'.format(time.localtime().tm_mon, time.localtime().tm_mday, int(time.time())))            
#     # # plot accuracy
#     # plt.cla()
#     # plt.suptitle('Training results of model #1 with config. #1')
#     # plt.xlabel('Epochs')
#     # plt.ylabel('Accuracy')
#     # plt.ylabel('Cross Entropy Loss')
#     # plt.yticks(np.arange(0.7, 1.0, 0.025))
#     # plt.ylim(0.7, 1.0)        
#     # plt.plot(histories[i].history['accuracy'], color='blue', label='train')
#     # plt.plot(histories[i].history['val_accuracy'], color='orange', label='test')
#     # plt.grid(True)
#     # plt.savefig(figsave_path + '\\accuracy_{0}{1}_{2}.png'.format(time.localtime().tm_mon, time.localtime().tm_mday, int(time.time())))            
#     # plt.close(fig)

#     # plt.suptitle('Test results of model %d' %(i))
#     # plt.suptitle('Test results of model 0')
#     # plt.savefig(figsave_path + '\\diagnostics_model_%d_%d.png' %(i, int(time.time())))
#     # if (i < 3):
#     #     plt.suptitle('Test results of model #1 with config. %d' %(i+1))
#     #     plt.savefig(figsave_path + '\\diagnostics_model_0_%d_%d.png' %(i, int(time.time())))
#     # else:
#     #     plt.suptitle('Test results of model #%d' %i)
#     #     plt.savefig(figsave_path + '\\diagnostics_model_%d_%d.png' %(i-3, int(time.time())))            
#     plt.close(fig)

# print('Plot the fitting results...')
# results_plot(histories)

""" Monitor norm of weights in each layers during training """
# from tensorflow.keras.callbacks import Callback
# from matplotlib import pyplot as plt


# class WeightCapture(Callback):
#     def __init__(self):
#         super().__init__()
#         self.weights = []
#         self.epochs = []
#     def on_epoch_end(self, epoch, logs=None):
#         self.epochs.append(epoch) # remember the epoch axis
#         weight = {}
#         for layer in md.layers:
#             if not layer.weights:
#                 continue
#             name = layer.weights[0].name.split("/")[1]
#             # weight[name] = layer.weights[0].numpy()
#             weight[name] = layer.get_weights()[0]   # layer.get_weights()[0] returns the kernel, layer.get_weights()[1] returns the bias, dtype=np.ndarray
#         self.weights.append(weight)

# def plotweight(capture_cb):
#     "Plot the weights' mean and s.d. across epochs"
#     _, ax = plt.subplots(2, 1, sharex=True, constrained_layout=True)
#     ax[0].set_title("Mean of weights -- %s" %(capture_cb.model.name))
#     ax[1].set_title("Standard deviation of weights -- %s" %(capture_cb.model.name))
#     plt.xlabel('Epochs')
#     # weight = []
#     # for epoch in capture_cb.epochs:
#     for key in capture_cb.weights[0]:   # 'key' is the name of layer
#         # weight[key].append(w[epoch][key].mean() for w in capture_cb.weights)
#         ax[0].plot(capture_cb.epochs, [w[key].mean() for w in capture_cb.weights], label=key)
#         ax[1].plot(capture_cb.epochs, [w[key].std() for w in capture_cb.weights], label=key)
#     ax[0].legend()
#     ax[1].legend()
#     plt.xlabel('Epochs')
#     # plt.show()
#     plt.savefig(figsave_path + '\\weights_%s_%d.png' %(capture_cb.model.name, int(time.time())))


for md in models:
    # capture_cb = WeightCapture()

    md.compile(
        optimizer=tf.keras.optimizers.Adam(0.0001),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy']        
        )


    # md.fit(x_train, y_train, batch_size=batch_size, epochs=50, validation_split=0.1, callbacks=[capture_cb])
    md.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)

    results = md.evaluate(x_test, y_test, batch_size=32)

    test_results.append(results)

result_path = 'result_{0}{1}.txt'.format(time.localtime().tm_mon, time.localtime().tm_mday)
with open(result_path, 'a+') as f:
    f.write("Evaluation results of %s: \n" %(md.name))
    for line in test_results:
        f.write(f"{line}\n\n")

#     plotweight(capture_cb)



""" Monitor norm of gradients in each layers during training """
# optimizer = tf.keras.optimizers.Adam(0.0001)
# loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()

# def train_model(X, y, model, n_epochs=epochs, batch_size=batch_size):
#     "Run training loop manually"
#     train_dataset = tf.data.Dataset.from_tensor_slices((X, y))
#     train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)
 
#     gradhistory = []
#     losshistory = []
#     def recordweight():
#         data = {}
#         for g,w in zip(grads, model.trainable_weights):
#             if '/kernel:' not in w.name:
#                 continue # skip bias
#             name = w.name.split("/")[0]
#             data[name] = g.numpy()
#         gradhistory.append(data)
#         losshistory.append(loss_value.numpy())
#     for _ in range(n_epochs):
#         for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
#             with tf.GradientTape() as tape:
#                 y_pred = model(x_batch_train, training=True)
#                 loss_value = loss_fn(y_batch_train, y_pred)
 
#             grads = tape.gradient(loss_value, model.trainable_weights)
#             optimizer.apply_gradients(zip(grads, model.trainable_weights))
 
#             if step == 0:
#                 recordweight()
#     # After all epochs, record again
#     recordweight()
#     return gradhistory, losshistory

# from matplotlib import pyplot as plt 

# def plot_gradient(gradhistory, losshistory):
#     "Plot gradient mean and sd across epochs"
#     _, ax = plt.subplots(2, 1, sharex=False, constrained_layout=True, figsize=(8, 12))
#     ax[0].set_title("Mean")
#     for key in gradhistory[0]:
#         ax[0].plot(range(len(gradhistory)), [w[key].mean() for w in gradhistory], label=key)
#     ax[0].legend()
#     ax[1].set_title("Standard Deviation")
#     for key in gradhistory[0]:
#         ax[1].semilogy(range(len(gradhistory)), [w[key].std() for w in gradhistory], label=key)
#     ax[1].legend()
#     plt.suptitle("Statistics of gradients per layer -- Model20")
#     plt.savefig(figsave_path + '\\gradients_%s_%d.png' %('Model20', int(time.time())))

# gradhistory, losshistory = train_model(x_train, y_train, model20)
# plot_gradient(gradhistory, losshistory)

""" Plot elbos """
# num_conf = 4
# elbos = []
# ep = np.arange(0,50)
# for i in range(num_conf):
#     with open ('elbo_Model3_Config{}.txt'.format(i)) as f:
#         lines = f.readlines()
#         elb = [float(s) for s in lines[:50]]
#         elbos.append(elb)
# plt.figure()
# for elbo in elbos:
#     plt.plot(ep, elbo)
# plt.legend(["m3c0", "m3c1", "m3c2", "m3c3"])
# plt.ylim(-13.5, -2, 0.5)
# plt.grid()
# plt.xlabel('Epochs')
# plt.title('Negative ELBO')
# plt.show()
# # plt.savefig(figsave_path + '\\elbo{0}{1}_{2}.png'.format(time.localtime().tm_mon, time.localtime().tm_mday, int(time.time())))
