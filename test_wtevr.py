# """ CNN for MNIST classification """
# example of loading the mnist dataset
# import tensorflow as tf
# import numpy as np
# from tensorflow.keras.datasets import mnist
# from tensorflow.keras import Sequential
# from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, BatchNormalization, ReLU
# import os
# from matplotlib import pyplot as plt

# # load dataset
# (x_train, y_train), (x_test, y_test) = mnist.load_data()
# # summarize loaded dataset
# print('Train: X=%s, y=%s' % (x_train.shape, y_train.shape))
# print('Test: X=%s, y=%s' % (x_test.shape, y_test.shape))
# # # plot first few images
# # for i in range(9):
# # 	# define subplot
# # 	plt.subplot(330 + 1 + i)
# # 	# plot raw pixel data
# # 	plt.imshow(x_train[i], cmap=plt.get_cmap('gray'))
# # # show the figure
# # plt.show()
# # reshape dataset to have a single channel
# x_train = np.expand_dims(x_train, axis=-1).astype('float32') / 255
# x_test = np.expand_dims(x_test, axis=-1).astype('float32') / 255

# # define cnn model
# X = tf.keras.layers.Input(shape=(28, 28, 1))
# x = Conv2D(filters=64, kernel_size=3, strides=(1, 1), padding='valid')(X)
# x = BatchNormalization()(x)
# x = ReLU()(x)
# x = MaxPool2D((2,2))(x)
# x = Conv2D(filters=128, kernel_size=3, strides=(1, 1), padding='valid')(x)
# x = BatchNormalization()(x)
# x = ReLU()(x)
# x = MaxPool2D((2,2))(x)
# x = Flatten()(x)
# x = Dense(units=64, activation="relu", use_bias=True)(x)
# y = Dense(10, activation="softmax")(x)

# model = tf.keras.models.Model(X, y)


# model.compile(
#     optimizer=tf.keras.optimizers.Adam(0.0001),  # Utilize optimizer
#     loss=tf.keras.losses.SparseCategoricalCrossentropy(),
#     metrics=['accuracy'])

# history = model.fit(
# 	x_train,
# 	y_train,
# 	batch_size=64,
# 	validation_split=0.1,
# 	epochs=10)

# # Save the model
# abs_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# save_path = os.path.join(abs_path, 'DropConnect_LDPC\\saved_models')
# model.save(save_path + '\\savingtest')

# # Reload the model
# from tensorflow.keras.models import load_model
# abs_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# save_path = os.path.join(abs_path, 'DropConnect_LDPC\\saved_models')
# # md_reload = load_model(save_path + '\\savingtest')
# md_reload = load_model(save_path + '\\model_1')
# md_reload.summary()

# x_test = x_test.reshape(-1, 784).astype('float32') / 255

# history = md_reload.evaluate(x_test, y_test, batch_size=128, verbose=1)

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


# """ Have a look at the LDPC matrix generation """
# # from pyldpc import parity_check_matrix
# import numpy as np
# from pyldpc import utils


# def parity_check_matrix(n_code, d_v, d_c, seed=None):
#     """
#     Inherited from pyldpc/code.py

#     """
#     rng = utils.check_random_state(seed)

#     if d_v <= 1:
#         raise ValueError("""d_v must be at least 2.""")

#     if d_c <= d_v:
#         raise ValueError("""d_c must be greater than d_v.""")

#     if n_code % d_c:
#         raise ValueError("""d_c must divide n for a regular LDPC matrix H.""")

#     n_equations = (n_code * d_v) // d_c

#     block = np.zeros((n_equations // d_v, n_code), dtype=int)
#     H = np.empty((n_equations, n_code))
#     H_check = np.empty((1, n_code)) # Check if  there is circle in H
#     block_size = n_equations // d_v

#     # Filling the first block with consecutive ones in each row of the block

#     for i in range(block_size):
#         for j in range(i * d_c, (i+1) * d_c):
#             block[i, j] = 1
#     H[:block_size] = block

#     # reate remaining blocks by permutations of the first block's columns:
#     for i in range(1, d_v):
#         H_t = rng.permutation(block.T).T
#         c = circle_check(H_t, block_size, d_c)
#         while (c == 1):
#             H_t = rng.permutation(block.T).T
#             c = circle_check(H_t, block_size, d_c)
#         H[i * block_size: (i + 1) * block_size] = H_t         
#     H = H.astype(int)
#     return H

# def circle_check(H_t, block_size, d_c):
#     for j in range(block_size):
#         # c_sum = 0
#         H_check = np.sum(H_t[:, j*d_c:(j+1)*d_c], axis=1)
#         for k in range(block_size):
#             # if (H_check[k] > 2):
#             #     return 1
#             # if (H_check[k] == 2):
#             #     c_sum = c_sum +1
#             #     if (c_sum > 1):
#             #         return 1
#             if (H_check[k] > 1):
#                 return 1
#     return 0

# """ Configuration marked: n=128, m=64, dc=4, coding rate=0.5 """
# n = 128
# m = 64
# dc = 4
# dv = m*dc//n
# # dv = 3
# # H = gen_ldpc(n,m,0.95)
# # H, _ = gen_ldpc(64, 50, 5)
# H = parity_check_matrix(n,dv,dc)

# print(H)


# # def decimal_range(start, stop, increment):
# #     while start < stop: # and not math.isclose(start, stop): Py>3.5
# #         yield start
# #         start += increment

# # for p in decimal_range(0, 0.9, 0.1):
# # 	print("The dropping probability is: %f" %(p))
# # 	print(gen_ldpc(n, m, p))

# """ Try to use matlab script to generate... """
# import matlab.engine
# n = 784
# m = 128
# k = n - m
# p = 1/8
# dv = m*p
# dc = n*p
# eng = matlab.engine.start_matlab()
# eng.cd(r'dc_ldpc', nargout = 0)
# print(eng.generate_regular_H(n,k,dv,dc))
# # TODO: store the output matrix and link with the model.

# import numpy as np
# from matplotlib import pyplot as plt
# x = np.arange(1,12)
# acc = [0.07999963016416878, 0.08147271554172039, 0.08131724742986261, 0.08268744261618703, 0.09019401749782265, 0.0824192551568504, 0.08465942001957447, 0.08883231503516435, 0.09520135155059398, 0.08906544685773551, 0.09224988257139921]
# plt.plot(x,acc)
# plt.show()

""" Calculate the rank of matrices """
from dc_ldpc import genldpc
n = 128
m = 64
p = 0.75
dv = int((1-p) * m)
dc = int((1-p) * n)
H = genldpc.parity_check_matrix(n, m, dv, dc)
rank = genldpc.rank_cal(H)
print("The rank of matrix is %d." %(rank))