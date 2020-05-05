#!/usr/bin/env python3
import tensorflow as tf
import numpy as np

#def gaussian_MLP_encoder(x, n_hidden, n_output, keep_prob):
def gaussian_MLP_encoder(x, n_hidden, n_output, keep_prob, d_switch, trainable=True):
    model = tf.keras.Sequential()
    
    batch1 = tf.keras.layers.BatchNormalization(
        axis=-1, scale=True, training=trainable, name="BN1")
    layer1 = tf.keras.layers.Conv2D(
        filters=32, kernel_size=8, strides=(2,2), padding='same', activation=tf.nn.relu, trainable=trainable, name="Conv1")
    pool1 = tf.keras.layers.max_pooling2d(
        pool_size=[2,2],strides=2)
    dropout1 = tf.keras.layers.Dropout(
        rate=keep_prob, seed=seed, training=d_switch)

    batch2 = tf.layers.batch_normalization(
        inputs=dropout1, axis=-1, scale=True, training=trainable, name="BN2")
    layer2 = tf.layers.conv2d(
        inputs=batch2, filters=64, kernel_size=4, strides=(2,2), padding='same', activation=tf.nn.relu, t, name="Conv2")
    pool2 = tf.layers.max_pooling2d(
        layer2,pool_size=[2,2],strides=2)
    dropout2 = tf.layers.dropout(
        inputs=pool2, rate=keep_prob, seed=seed, training=d_switch)

    batch3 = tf.layers.batch_normalization(
        inputs=dropout2, axis=-1, scale=True, training=trainable, name="BN3")
    layer3 = tf.layers.conv2d(
        inputs=batch3, filters=128, kernel_size=4, strides=(2,2), padding='same', activation=tf.nn.relu, name="Conv3")
    pool3 = tf.layers.max_pooling2d(
        layer3,pool_size=[2,2],strides=2)
    dropout3 = tf.layers.dropout(
        inputs=pool3, rate=keep_prob, seed=seed, training=d_switch)

    batch4 = tf.layers.batch_normalization(
        inputs=dropout3, axis=-1, scale=True, training=trainable, name="BN4")
    layer4 = tf.layers.conv2d(
        inputs=batch4, filters=256, kernel_size=4, strides=(2,2), padding='same', activation=tf.nn.relu, name="Conv4")
    pool4 = tf.layers.max_pooling2d(
        layer4,pool_size=[2,2],strides=2)
    dropout4 = tf.layers.dropout(
        inputs=pool4, rate=keep_prob, seed=seed, training=d_switch)

    flat = tf.layers.flatten(dropout4)
    fc1 = tf.layers.dense(
        input=flat, units=1024)
    dropout_fc1 = tf.layers.dropout(input=fc1, rate=keep_prob, seed=seed, training=d_switch)

    conv2d(channel,32,ksize=4,stride=2)
    relu
    conv2d(channel,32,ksize=4,stride=2)
    relu
    conv2d(channel,32,ksize=4,stride=2)
    relu
    conv2d(channel,32,ksize=4,stride=2)
    relu
    flatten
    fc1
    fc2
    fc3 to z



        # initializers
        w_init = tf.contrib.layers.variance_scaling_initializer()
        b_init = tf.constant_initializer(0.)

        # 1st hidden layer
        w0 = tf.get_variable('w0', [x.get_shape()[1], n_hidden], initializer=w_init)
        b0 = tf.get_variable('b0', [n_hidden], initializer=b_init)
        h0 = tf.matmul(x, w0) + b0
        h0 = tf.nn.elu(h0)
        h0 = tf.nn.dropout(h0, keep_prob)

        # 2nd hidden layer
        w1 = tf.get_variable('w1', [h0.get_shape()[1], n_hidden], initializer=w_init)
        b1 = tf.get_variable('b1', [n_hidden], initializer=b_init)
        h1 = tf.matmul(h0, w1) + b1
        h1 = tf.nn.tanh(h1)
        h1 = tf.nn.dropout(h1, keep_prob)

        # output layer
        wo = tf.get_variable('wo', [h1.get_shape()[1], n_output * 2], initializer=w_init)
        bo = tf.get_variable('bo', [n_output * 2], initializer=b_init)
        gaussian_params = tf.matmul(h1, wo) + bo

        # The mean parameter is unconstrained
        mean = gaussian_params[:, :n_output]
        # The standard deviation must be positive. Parametrize with a softplus and
        # add a small epsilon for numerical stability
        stddev = 1e-6 + tf.nn.softplus(gaussian_params[:, n_output:])

    return mean, stddev


# Bernoulli MLP as decoder
def bernoulli_MLP_decoder(z, n_hidden, n_output, keep_prob, reuse=False):

    with tf.variable_scope("bernoulli_MLP_decoder", reuse=reuse):
        unflatten
        convtp
        relu
        convtp
        relu
        convtp
        relu
        convtp
        sigmoid


        # initializers
        w_init = tf.contrib.layers.variance_scaling_initializer()
        b_init = tf.constant_initializer(0.)

        # 1st hidden layer
        w0 = tf.get_variable('w0', [z.get_shape()[1], n_hidden], initializer=w_init)
        b0 = tf.get_variable('b0', [n_hidden], initializer=b_init)
        h0 = tf.matmul(z, w0) + b0
        h0 = tf.nn.tanh(h0)
        h0 = tf.nn.dropout(h0, keep_prob)

        # 2nd hidden layer
        w1 = tf.get_variable('w1', [h0.get_shape()[1], n_hidden], initializer=w_init)
        b1 = tf.get_variable('b1', [n_hidden], initializer=b_init)
        h1 = tf.matmul(h0, w1) + b1
        h1 = tf.nn.elu(h1)
        h1 = tf.nn.dropout(h1, keep_prob)

        # output layer-mean
        wo = tf.get_variable('wo', [h1.get_shape()[1], n_output], initializer=w_init)
        bo = tf.get_variable('bo', [n_output], initializer=b_init)
        x_hat = tf.sigmoid(tf.matmul(h1, wo) + bo)

    return x_hat


# Gateway
def autoencoder(x, dim_img, dim_z, n_hidden, keep_prob):

    # encoding
    mu, sigma = gaussian_MLP_encoder(x, n_hidden, dim_z, keep_prob, d_switch)

    # sampling by re-parameterization technique
    z = mu + sigma * tf.random_normal(tf.shape(mu), 0, 1, dtype=tf.float32)

    # decoding
    x_hat = bernoulli_MLP_decoder(z, n_hidden, dim_img, keep_prob)
    x_hat = tf.clip_by_value(x_hat, 1e-8, 1 - 1e-8)

    # loss
    marginal_likelihood = tf.reduce_sum(x * tf.log(x_hat) + (1 - x) * tf.log(1 - x_hat), 1)
    KL_divergence = 0.5 * tf.reduce_sum(tf.square(mu) + tf.square(sigma) - tf.log(1e-8 + tf.square(sigma)) - 1, 1)

    marginal_likelihood = tf.reduce_mean(marginal_likelihood)
    KL_divergence = tf.reduce_mean(KL_divergence)

    ELBO = marginal_likelihood - KL_divergence

    loss = -ELBO

    return x_hat, z, loss, -marginal_likelihood, KL_divergence


def sensor_fusion(image, camera_position, gripper_pressure, joint_states, attension):
    train_total_data, train_size, _, _, test_data, test_labels = mnist_data.prepare_MNIST_data()
    n_samples = train_size


def graph_init():
    # input placeholders
    x = tf.placeholder(tf.float32, shape=[None, dim_img], name='input_img')
    # dropout
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    # network architecture
    x_hat, z, loss, neg_marginal_likelihood, KL_divergence = autoencoder(x, dim_img, dim_z, n_hidden, keep_prob)

    return x, z, x_hat