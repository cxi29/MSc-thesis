# """ CNN for MNIST classification """
# example of loading the mnist dataset
import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, BatchNormalization, ReLU
import os
from matplotlib import pyplot as plt

# load dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# summarize loaded dataset
print('Train: X=%s, y=%s' % (x_train.shape, y_train.shape))
print('Test: X=%s, y=%s' % (x_test.shape, y_test.shape))
# # plot first few images
# for i in range(9):
# 	# define subplot
# 	plt.subplot(330 + 1 + i)
# 	# plot raw pixel data
# 	plt.imshow(x_train[i], cmap=plt.get_cmap('gray'))
# # show the figure
# plt.show()
# reshape dataset to have a single channel
x_train = np.expand_dims(x_train, axis=-1).astype('float32') / 255
x_test = np.expand_dims(x_test, axis=-1).astype('float32') / 255

# define cnn model
X = tf.keras.layers.Input(shape=(28, 28, 1))
x = Conv2D(filters=64, kernel_size=3, strides=(1, 1), padding='valid')(X)
x = BatchNormalization()(x)
x = ReLU()(x)
x = MaxPool2D((2,2))(x)
x = Conv2D(filters=128, kernel_size=3, strides=(1, 1), padding='valid')(x)
x = BatchNormalization()(x)
x = ReLU()(x)
x = MaxPool2D((2,2))(x)
x = Flatten()(x)
x = Dense(units=64, activation="relu", use_bias=True)(x)
y = Dense(10, activation="softmax")(x)

model = tf.keras.models.Model(X, y)


model.compile(
    optimizer=tf.keras.optimizers.Adam(0.0001),  # Utilize optimizer
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy'])

history = model.fit(
	x_train,
	y_train,
	batch_size=64,
	validation_split=0.1,
	epochs=10)

# Save the model
abs_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
save_path = os.path.join(abs_path, 'DropConnect_LDPC\\saved_models')
model.save('save_path' + '\\savingtest')

# Reload the model
from tensorflow.keras.models import load_model
md_reload = load_model('save_path' + '\\savingtest')
md_reload.summary()

history = md_reload.evaluate(x_test, y_test, batch_size=128, verbose=2)

# """ Plotting """
# import matplotlib.pyplot as plt
# import numpy as np
# import os

# fig=plt.figure()
# data=np.arange(900).reshape((30,30))
# for i in range(1,5):
#     ax=fig.add_subplot(2,2,i)        
#     ax.imshow(data)

# fig.suptitle('Main title') # or plt.suptitle('Main title')
# # plt.show()

# abs_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# save_path = os.path.join(abs_path, 'DropConnect_LDPC\\plottings')
# plt.savefig(save_path + '\\test.png')

# # emmm... another (more secure) way?
# # import matplotlib.pyplot as plt
# # fig, ax = plt.subplots( nrows=1, ncols=1 )  # create figure & 1 axis
# # ax.plot([0,1,2], [10,20,3])
# # fig.savefig('path/to/save/image/to.png')   # save the figure to file
# # plt.close(fig)    # close the figure window