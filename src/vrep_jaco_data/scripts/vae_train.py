import tensorflow as tf
import numpy as np
import os
import vae_util

RESULTS_DIR = args.results_path

# network architecture
n_hidden = args.n_hidden
dim_img = IMAGE_SIZE_MNIST**2  # number of pixels for a MNIST image
dim_z = args.dim_z

# train
n_epochs = args.num_epochs
batch_size = args.batch_size
learn_rate = args.learn_rate

""" prepare MNIST data """
train_total_data, train_size, _, _, test_data, test_labels = mnist_data.prepare_MNIST_data()
n_samples = train_size

""" build graph """
# input placeholders
x = tf.placeholder(tf.float32, shape=[None, dim_img], name='input_img')
# dropout
keep_prob = tf.placeholder(tf.float32, name='keep_prob')
# network architecture
y, z, loss, neg_marginal_likelihood, KL_divergence = vae_util.autoencoder(x, dim_img, dim_z, n_hidden, keep_prob, d_switch)
# optimization
train_op = tf.train.AdamOptimizer(learn_rate).minimize(loss)

""" training """
# train
total_batch = int(n_samples / batch_size)
min_tot_loss = 1e99

with tf.Session() as sess:

    sess.run(tf.global_variables_initializer(), feed_dict={keep_prob : 0.8})

    for epoch in range(n_epochs):

        # Random shuffling
        np.random.shuffle(train_total_data)
        train_data_ = train_total_data[:, :-mnist_data.NUM_LABELS]

        # Loop over all batches
        for i in range(total_batch):
            # Compute the offset of the current minibatch in the data.
            offset = (i * batch_size) % (n_samples)
            batch_xs_input = train_data_[offset:(offset + batch_size), :]

            _, tot_loss, loss_likelihood, loss_divergence = sess.run(
                (train_op, loss, neg_marginal_likelihood, KL_divergence),
                feed_dict={x: batch_xs_target, keep_prob : 0.8})

        # print cost every epoch
        print("epoch %d: L_tot %03.2f L_likelihood %03.2f L_divergence %03.2f" % (epoch, tot_loss, loss_likelihood, loss_divergence))

        # if minimum loss is updated or final epoch, plot results
        if min_tot_loss > tot_loss or epoch+1 == n_epochs:
            min_tot_loss = tot_loss