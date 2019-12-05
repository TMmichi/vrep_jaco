import tensorflow as tf
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import random
from datetime import datetime as dt

def CNN_Encoder(x, z_dim, keep_prob, trainable=True, d_switch=False):
    d_switch = d_switch or trainable
    random.seed(dt.now())
    seed = random.randint(0,12345)
    with tf.compat.v1.variable_scope("CNN_Encoder"):
        print(x.shape)
        batch1 = tf.layers.batch_normalization(
            inputs=x, axis=-1, scale=True, training=trainable, name="BN1")
        layer1 = tf.layers.conv2d(
            inputs=batch1, filters=16, kernel_size=8, strides=(1,1), padding='same',
            activation=tf.nn.relu, trainable=trainable, name="Conv1")
        pool1 = tf.layers.max_pooling2d(
            inputs=layer1,pool_size=[2,2],strides=2)
        dropout1 = tf.layers.dropout(
            inputs=pool1, rate=keep_prob, seed=seed, training=d_switch)
        print(dropout1.shape)

        batch2 = tf.layers.batch_normalization(
            inputs=dropout1, axis=-1, scale=True, training=trainable, name="BN2")
        layer2 = tf.layers.conv2d(
            inputs=batch2, filters=32, kernel_size=4, strides=(1,1), padding='same',
            activation=tf.nn.relu, trainable=trainable, name="Conv2")
        pool2 = tf.layers.max_pooling2d(
            inputs=layer2,pool_size=[2,2],strides=2)
        dropout2 = tf.layers.dropout(
            inputs=pool2, rate=keep_prob, seed=seed, training=d_switch)
        print(dropout2.shape)

        batch3 = tf.layers.batch_normalization(
            inputs=dropout2, axis=-1, scale=True, training=trainable, name="BN3")
        layer3 = tf.layers.conv2d(
            inputs=batch3, filters=64, kernel_size=4, strides=(1,1), padding='same',
            activation=tf.nn.relu, trainable=trainable, name="Conv3")
        pool3 = tf.layers.max_pooling2d(
            inputs=layer3,pool_size=[2,2],strides=2)
        dropout3 = tf.layers.dropout(
            inputs=pool3, rate=keep_prob, seed=seed, training=d_switch)
        print(dropout3.shape)

        batch4 = tf.layers.batch_normalization(
            inputs=dropout3, axis=-1, scale=True, training=trainable, name="BN4")
        layer4 = tf.layers.conv2d(
            inputs=batch4, filters=128, kernel_size=4, strides=(1,1), padding='same',
            activation=tf.nn.relu, trainable=trainable, name="Conv4")
        pool4 = tf.layers.max_pooling2d(
            inputs=layer4,pool_size=[2,2],strides=2)
        dropout4 = tf.layers.dropout(
            inputs=pool4, rate=keep_prob, seed=seed, training=d_switch)
        print(dropout4.shape)

        flat = tf.layers.flatten(dropout4)
        print(flat.shape)
        fc1 = tf.layers.dense(
            inputs=flat, units=1024, name="FC1")
        dropout_fc1 = tf.layers.dropout(
            inputs=fc1, rate=keep_prob, seed=seed, training=d_switch)
        print(dropout_fc1.shape)
        
        fc2 = tf.layers.dense(
            inputs=dropout_fc1, units=2*z_dim, name="FC2")
        print(fc2.shape)
        
        mean = fc2[:, :z_dim]
        stddev = 1e-6 + tf.nn.softplus(fc2[:, z_dim:])

    return mean, stddev

def CNN_Decoder(z, keep_prob, reuse=False):
    with tf.variable_scope("CNN_Decoder", reuse=reuse):
        de_fc1 = tf.layers.dense(
            inputs=z, units=1024, name="de_FC1")
        dropout_de_fc1 = tf.layers.dropout(
            inputs=de_fc1, rate=keep_prob)

        de_fc2 = tf.layers.dense(
            inputs=dropout_de_fc1, units=153600, name="de_FC2")
        dropout_de_fc2 = tf.layers.dropout(
            inputs=de_fc2, rate=keep_prob)

        unflat = tf.reshape(
            tensor=dropout_de_fc2, shape=[-1,30,40,128])

        de_layer1 = tf.layers.conv2d_transpose(
            inputs=unflat, filters=64, kernel_size=5, strides=(2,2), padding='same',
            activation=tf.nn.relu, name="DeConv1")
        de_dropout1 = tf.layers.dropout(
            inputs=de_layer1, rate=keep_prob)

        de_layer2 = tf.layers.conv2d_transpose(
            inputs=de_dropout1, filters=32, kernel_size=5, strides=(2,2), padding='same',
            activation=tf.nn.relu, name="DeConv2")
        de_dropout2 = tf.layers.dropout(
            inputs=de_layer2, rate=keep_prob)

        de_layer3 = tf.layers.conv2d_transpose(
            inputs=de_dropout2, filters=16, kernel_size=6, strides=(2,2), padding='same',
            activation=tf.nn.relu, name="DeConv3")
        de_dropout3 = tf.layers.dropout(
            inputs=de_layer3, rate=keep_prob)
        
        de_layer4 = tf.layers.conv2d_transpose(
            inputs=de_dropout3, filters=1, kernel_size=8, strides=(2,2), padding='same', name="DeConv4")
        
        x_hat = tf.sigmoid(de_layer4)

    return x_hat

data = cv.imread("/home/ljh/Documents/Figure_2.png",0)
data = data/255
plt.imshow(data)
plt.show()
data = np.reshape(data,[1,data.shape[0],data.shape[1],1])


print("shape = ",data.shape)

x = tf.compat.v1.placeholder(tf.float32, shape=[None, 480,640,1], name='input_img')
keep_prob = tf.compat.v1.placeholder(tf.float32, name='keep_prob')

mean, stddev = CNN_Encoder(x,16,keep_prob,trainable=False)
x_hat = CNN_Decoder(mean,keep_prob=0.8)

config = tf.ConfigProto()
config.gpu_options.allow_growth=True

with tf.compat.v1.Session(config=config) as sess:
    counter = 0
    then_init = dt.now()
    sess.run(tf.compat.v1.global_variables_initializer(), feed_dict={keep_prob : 0.8})
    while counter < 100:
        counter += 1
        then = dt.now()
        #output = sess.run((mean,stddev),feed_dict={x:data,keep_prob:0.8})
        output = sess.run((x_hat),feed_dict={x:data,keep_prob:0.8})
        print(output)
        print(output.shape,dt.now()-then_init)
