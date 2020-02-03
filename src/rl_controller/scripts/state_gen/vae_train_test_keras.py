import tensorflow as tf
import numpy as np
import os
import time
import state_gen_util as state_gen_util
from matplotlib import pyplot as plt
from IPython import display

plt.figure()

# train
n_epochs = 1000
batch_size = 16
learn_rate = 5e-6

""" Image data """
data = np.load("/Users/jeonghoon/tmp/dummy_data.npy",allow_pickle=True)
train_dataset_list = []
for j in range(1):
    train_dataset = []
    for i in range(20):
        img = data[i][0][0]/5000
        img = np.reshape(img,[img.shape[0],img.shape[1],1])
        train_dataset.append(img)
    train_dataset = np.array(train_dataset,dtype=np.float32)
    train_dataset_list.append(train_dataset)
train_dataset_list = np.array(train_dataset_list)
print(train_dataset_list.shape)

test_x = train_dataset_list[-1]

""" build graph """
autoencoder = state_gen_util.Autoencoder()
#tic = time.time()
#x_hat = autoencoder(train_dataset[0])
#print(time.time()-tic)

optimizer = tf.keras.optimizers.Adam(1e-4)

""" training """
# train
epochs = 10

with tf.Graph().as_default(), tf.Session() as sess:
    for epoch in range(1, epochs + 1):
        start_time = time.time()
        for train_x in train_dataset_list:
            autoencoder.compute_apply_gradients(train_x, optimizer)
            sess.run((optimizer))
        end_time = time.time()

        if epoch % 1 == 0:
            predictions = autoencoder.sample(1)
            predictions = sess.run(predictions)
            plt.imshow(predictions[0], cmap='gray')
            plt.show()