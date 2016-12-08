# CNN Siamese Implementation for face recognition.

import random
import numpy as np
import time
import tensorflow as tf
import math
import pdb
import sys
import h5py
import scipy.io as sio
from sklearn import *
import matplotlib.pyplot as plt
from PlotROC import Plot_ROC_Fn
from PlotHIST import Plot_HIST_Fn
from PlotPR import Plot_PR_Fn
import re
import Siamese_Architecture
import cv2

"""
Parameters and input data
"""

TRAIN_FILE = h5py.File('data/TRAIN.hdf5', 'r')
TEST_FILE = h5py.File('data/TEST.hdf5', 'r')

# Extracting the data and labels from HDF5.
# Pairs and labels have been saved separately in HDF5 files.
# The number of features and samples are extracted from data.
X_train = TRAIN_FILE['pairs']
y_train = TRAIN_FILE['labels']
X_test = TEST_FILE['pairs']
y_test = TEST_FILE['labels']


# Dimensions
num_samples = X_train.shape[0]
height = X_train.shape[1]
width = X_train.shape[2]
num_channels = X_train.shape[3] / 2

# Defining distance initial vectors for evaluation of the final output of the network.
distance_train_output = np.zeros(y_train.shape)
distance_test_output = np.zeros(y_test.shape)


# Defining the graph of network.
graph = tf.Graph()
with graph.as_default():

    # The batch size for gradient update.
    batch_size = 16
    batch = tf.Variable(0, trainable=False)


    # Learning rate policy.
    starter_learning_rate = 0.001
    num_batches_per_epoch = int(num_samples / batch_size)
    NUM_EPOCHS_PER_DECAY = 1
    decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)
    decay_steps = 1
    LEARNING_RATE_DECAY_FACTOR = 0.95
    learning_rate = tf.train.exponential_decay(starter_learning_rate, batch, decay_steps,
                                               LEARNING_RATE_DECAY_FACTOR, staircase=True)

    # Adding the larning rate to summary.
    tf.scalar_summary('learning_rate', learning_rate)

    # Defining the place holders.
    images_L = tf.placeholder(tf.float32, shape=([None, height, width, num_channels]), name='L')
    images_R = tf.placeholder(tf.float32, shape=([None, height, width, num_channels]), name='R')
    labels = tf.placeholder(tf.float32, shape=([None, 1]), name='gt')
    dropout_param = tf.placeholder("float")

    # Sharing the same parameter in order to creating the Siamese Architecture.
    with tf.variable_scope("siamese") as scope:
        model_L = Siamese_Architecture.neural_network(images_L, dropout_param)
        scope.reuse_variables()
        model_R = Siamese_Architecture.neural_network(images_R, dropout_param)


    # Defining the distance metric for the outputs of the network.
    distance = tf.sqrt(tf.reduce_mean(tf.pow(tf.sub(model_L, model_R), 2), 1, keep_dims=True))

    # CALCULATION OF LOSS
    loss = Siamese_Architecture.loss(labels, distance, batch_size)

    #TODO: choosing different options for optimizer
    train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss, global_step=batch)
    # optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)
    # optimizer = tf.train.RMSPropOptimizer(0.0001,momentum=0.9,epsilon=1e-6).minimize(loss)

    # Create the saver.
    saver = tf.train.Saver()
    merged = tf.merge_all_summaries()



# TODO: Launching the graph!
with tf.Session(graph=graph) as sess:

    train_writer = tf.train.SummaryWriter('/home/sina/train',
                                          sess.graph)

    # How many iteration on the whole data is prompted by the user?
    num_epoch = 1
    init = tf.initialize_all_variables()
    sess.run(init)

    """
    *** Uncomment if you want to initialize the weights using numpy.
    This part is for assigning the numpy weights to the network weights.
    This is for initialization. In two cases this part should be commented.
    1- There is no need for initialization with pre-trained weights.
    2- The network is initialized and the weights are loaded from the restored checkpoint.
    """

    # # The initial weights
    # weights = np.load('weights/vgg16_casia.npy')
    # initial_params = weights.item()
    #
    # # Assigning weights
    # vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    # for var in vars:
    #     for key in initial_params:
    #
    #         if str(key) in str(var.name) and 'weights' in str(var.name):
    #             sess.run(tf.assign(var, initial_params[str(key)][0]))
    #             print "{} weights are assigned to {}".format(str(key), str(var.name))
    #         elif str(key) in str(var.name) and 'biases' in str(var.name):
    #             sess.run(tf.assign(var, initial_params[str(key)][1]))
    #             print "{} biases are assigned to {}".format(str(key), str(var.name))
    #         else:
    #             print "processed is passed for {} and {}".format(str(key), str(var))
    #
    # # Uncomment to save the model
    # print("Trying to save the model")
    # save_path = saver.save(sess, "weights/CASIA-model.ckpt")
    # print("Model saved in file: %s" % save_path)
    # sys.exit('No need to continue! The model is saved! Please run again with the saved check point.')

    # """
    # For fine-tuning the model which includes weights must be restored.
    # """
    # # Uncomment if you want to restore the model
    # saver.restore(sess, "weights/CASIA-model.ckpt")
    # print("Model restored.")

    # Training cycle
    for epoch in range(num_epoch):
        avg_loss = 0.
        avg_acc = 0.
        total_batch = int(num_samples / batch_size)
        start_time = time.time()

        # Loop over all batches
        for i in range(total_batch):
            start_idx = i * batch_size
            end_idx = (i + 1) * batch_size

            # Fit training using batch data
            input1, input2, y = Siamese_Architecture.get_batch(start_idx, end_idx, X_train, y_train)
            input1_te, input2_te, y_te = X_test[start_idx:end_idx, :, :, 0:num_channels], X_test[start_idx:end_idx, :, :, num_channels:], y_test

            # # Uncomment if you want to standardalize data with specific mean values for each channel.
            # mean = [148, 110, 105]
            # print mean
            # input1 = Siamese_Architecture.standardalize_Fn(input1, mean)
            # input2 = Siamese_Architecture.standardalize_Fn(input2, mean)
            # input1_te = Siamese_Architecture.standardalize_Fn(input1_te, mean)
            # input2_te = Siamese_Architecture.standardalize_Fn(input2_te, mean)


            # TODO: Running the session and evaluation of three elements.
            _, loss_value, summary = sess.run([train_step, loss, merged],
                                              feed_dict={images_L: input1, images_R: input2, labels: y,
                                                         dropout_param: 0.9})

            train_writer.add_summary(summary, i)

            # This will be repeated for each epoch but for the moment it is the only way
            # because the training data cannot be fed at once.
            distance_train_evaluation = distance.eval(
                feed_dict={images_L: X_train[start_idx:end_idx, :, :, 0:num_channels],
                           images_R: X_train[start_idx:end_idx, :, :, num_channels:],
                           dropout_param: 1.0})
            distance_train_output[start_idx:end_idx,:] = distance_train_evaluation

            # This will be repeated for each epoch but for the moment it is the only way
            # because the test data cannot be fed at once.
            # The upper limit for test data is less that train data.
            if end_idx < X_test.shape[0]:
                distance_test_evaluation = distance.eval(
                    feed_dict={images_L: X_test[start_idx:end_idx, :, :, 0:num_channels],
                               images_R: X_test[start_idx:end_idx, :, :, num_channels:],
                               dropout_param: 1.0})
                distance_test_output[start_idx:end_idx, :] = distance_test_evaluation
            else:
                if start_idx < X_test.shape[0]:
                    distance_test_evaluation = distance.eval(
                        feed_dict={images_L: X_test[start_idx:X_test.shape[0], :, :, 0:num_channels],
                                   images_R: X_test[start_idx:X_test.shape[0], :, :, num_channels:],
                                   dropout_param: 1.0})
                    distance_test_output[start_idx:end_idx, :] = distance_test_evaluation



            # Training output features per batch
            feature1_tr = model_L.eval(feed_dict={images_L: input1, dropout_param: 1.0})
            feature2_tr = model_R.eval(feed_dict={images_R: input2, dropout_param: 1.0})


            # Test output features per whole test set
            feature1_te = model_L.eval(feed_dict={images_L: input1_te, dropout_param: 1.0})
            feature2_te = model_R.eval(feed_dict={images_R: input2_te, dropout_param: 1.0})

            avg_loss += loss_value
            print("batch %d of %d loss= %f" % (i + 1, total_batch, loss_value))
        duration = time.time() - start_time
        print(
            'epoch %d  time: %f average_loss %0.5f' % (
                epoch + 1, duration, avg_loss / (total_batch)))


    # TODO: Test model on test samples
    label_test = np.reshape(y_test, (y_test.shape[0], 1))


"""
TRAIN
"""
# Distance between original features before feeding to the network.
distance_original_train = np.sqrt(
    np.sum(np.power(X_train[:, :, :, 0:num_channels] - X_train[:, :, :, num_channels:], 2), axis=(1, 2, 3)))

# Change the format from (N,1) to (N,).
distance_train_output = distance_train_output[:, 0]

# Plot ROC for training
Plot_ROC_Fn(y_train, distance_original_train, 'Train', 'Input')
Plot_ROC_Fn(y_train, distance_train_output, 'Train', 'Output')

# Plot Precision-Recall for training
Plot_PR_Fn(y_train, distance_original_train , 'Train', 'Input')
Plot_PR_Fn(y_train, distance_train_output, 'Train', 'Output')

# Plot HIST for training
Plot_HIST_Fn(y_train, distance_original_train, 'Train', 'Input')
Plot_HIST_Fn(y_train, distance_train_output, 'Train', 'Output')

"""
TEST
"""
# Distance between original features before feeding to the network.
distance_original_test = np.sqrt(
    np.sum(np.power(X_test[:, :, :, 0:num_channels] - X_test[:, :, :, num_channels:], 2), axis=(1, 2, 3)))

# Change the format from (N,1) to (N,).
distance_test_output = distance_test_output[:, 0]

# Plot ROC for test
Plot_ROC_Fn(label_test, distance_original_test, 'Test', 'Input')
Plot_ROC_Fn(label_test, distance_test_output, 'Test', 'Output')

# Plot Precision-Recall for test
Plot_PR_Fn(label_test, distance_original_test , 'Test', 'Input')
Plot_PR_Fn(label_test, distance_test_output, 'Test', 'Output')

# Plot HIST for test
Plot_HIST_Fn(label_test, distance_original_test, 'Test', 'Input')
Plot_HIST_Fn(label_test, distance_test_output, 'Test', 'Output')
