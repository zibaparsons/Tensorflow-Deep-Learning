# Siamese Architecture for face recognition

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

"""
Parameters and input data
"""
TRAIN_FILE = h5py.File('TRAIN.hdf5', 'r')
TEST_FILE = h5py.File('TEST.hdf5', 'r')

# Extracting the data and labels from HDF5.
# Pairs and labels have been saved separately in HDF5 files.
# The number of features and samples are extracted from data.
X_train = TRAIN_FILE['pairs']
y_train = TRAIN_FILE['labels']
X_test = TEST_FILE['pairs']
y_test = TEST_FILE['labels']
num_features = X_train.shape[1]
num_samples = X_train.shape[0]


"""
Necessary functions.
"""
# The structure of each layer in network
# The defined layer is fully connected.
def MLP_layer(input, num_input_units, num_output_units, name="MLP_layer"):
    with tf.variable_scope(name):
        W = tf.get_variable('W', [num_input_units, num_output_units], tf.float32,
                            tf.random_normal_initializer(mean=0.0001, stddev=0.05))
        return tf.nn.relu(tf.matmul(input, W))

# Multi-layer Perceptron model builder.
# The drop-out parameter must be entered by the user.
def MLP_Model(X, dropout):
    MODEL = MLP_Structure(X, dropout)
    return MODEL

# Siamese structure has two parallel branches. Each of them defined as below.
# Drop-out layer has been used to overcome overfitting.
def MLP_Structure(image, dropout):

    # First layer
    out_1 = MLP_layer(image, num_features, 1000, name='layer_1')
    out_1 = tf.nn.dropout(out_1, dropout)

    # # Second layer
    # out_2 = MLP_layer(out_1, 2000, 1500, name='l2')
    # out_2 = tf.nn.dropout(out_2, dropout)

    # The last output does not have dropout.
    out_2 = MLP_layer(out_1, 1000, 300, name='layer_2')   # 1000,50
    return out_2

# Computation of contrastive loss.
def contrastive_loss(y, distance, margin=1):
    term_1 = y * tf.square(distance)
    # tmp= tf.mul(y,tf.square(d))
    term_2 = (1 - y) * tf.square(tf.maximum((margin - distance), 0))
    return tf.reduce_sum(term_1 + term_2) / batch_size / 2

# Accuracy computation
def compute_accuracy(prediction, labels):
    return labels[prediction.ravel() < 0.5].mean()
    # return tf.reduce_mean(labels[prediction.ravel() < 0.5])

# Extracting the data and label for each batch.
def get_batch(start_idx, end_ind, inputs, labels):
    pair_left_part = inputs[start_idx:end_ind, :, 0]
    pair_right_part = inputs[start_idx:end_ind, :, 1]
    y = np.reshape(labels[start_idx:end_ind], (len(range(start_idx, end_ind)), 1))
    return pair_left_part, pair_right_part, y

# Defining the graph of network.
graph = tf.Graph()
with graph.as_default():
    batch_size = 128
    global_step = tf.Variable(0, trainable=False)
    starter_learning_rate = 0.001
    learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, 10, 0.1, staircase=True)

    images_L = tf.placeholder(tf.float32, shape=([None, num_features]), name='L')
    images_R = tf.placeholder(tf.float32, shape=([None, num_features]), name='R')
    labels = tf.placeholder(tf.float32, shape=([None, 1]), name='gt')
    dropout_param = tf.placeholder("float")

    with tf.variable_scope("siamese") as scope:
        model_L = MLP_Model(images_L, dropout_param)
        scope.reuse_variables()
        model_R = MLP_Model(images_R, dropout_param)

    # Contrastive Loss
    distance = tf.sqrt(tf.reduce_sum(tf.pow(tf.sub(model_L, model_R), 2), 1, keep_dims=True))
    margin = 1
    loss = contrastive_loss(labels, distance, margin)

    t_vars = tf.trainable_variables()
    d_vars = [var for var in t_vars if 'l' in var.name]
    batch = tf.Variable(0)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
    # optimizer = tf.train.RMSPropOptimizer(0.0001,momentum=0.9,epsilon=1e-6).minimize(loss)

# Launch the graph
with tf.Session(graph=graph) as sess:

    num_epoch = 2
    tf.initialize_all_variables().run()

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
            input1, input2, y = get_batch(start_idx, end_idx, X_train, y_train)

            # # Uncomment if you want to extract batches directly from HDF5 file.
            # with h5py.File(Train_File, 'r') as file_open:
            #
            #     # Extracting the labels for the specific batch
            #     dset_labels = file_open['labels']
            #     y = np.empty((batch_size, 1), dtype=np.float64)
            #     dset_labels.read_direct(y, np.s_[start_idx:end_inx, :])  # dest_sel can be omitted
            #     print y
            #
            #     # Extracting the data for the specific batch
            #     dset_data = file_open['pairs']
            #     batch_data = np.empty((batch_size, num_features, 2), dtype=np.float64)
            #     dset_data.read_direct(batch_data, np.s_[start_idx:end_inx, :, :])  # dest_sel can be omitted
            #     input1 = batch_data[:, :, 0]
            #     input2 = batch_data[:, :, 0]

            # input1_te, input2_te, y_te = next_batch(s, e, X_test, y_test)
            _, loss_value, predict = sess.run([optimizer, loss, distance],
                                              feed_dict={images_L: input1, images_R: input2, labels: y, dropout_param: 0.9})
            # Training evaluation
            feature1 = model_L.eval(feed_dict={images_L: input1, dropout_param: 1.0})
            feature2 = model_R.eval(feed_dict={images_R: input2, dropout_param: 1.0})
            tr_acc = compute_accuracy(predict, y)

            # # Test evaluation
            # feature1_te = model1.eval(feed_dict={images_L: input1_te, dropout_f: 1.0})
            # feature2_te = model2.eval(feed_dict={images_R: input2_te, dropout_f: 1.0})
            # te_acc = compute_accuracy(predict, y_te)

            if math.isnan(tr_acc) and epoch != 0:
                print('tr_acc %0.2f' % tr_acc)
                pdb.set_trace()
            avg_loss += loss_value
            avg_acc += tr_acc * 100
        # print('epoch %d loss %0.2f' %(epoch,avg_loss/total_batch))
        duration = time.time() - start_time
        print(
        'epoch %d  time: %f loss %0.5f acc %0.2f' % (epoch, duration, avg_loss / (total_batch), avg_acc / total_batch))

    # Test on Training
    label_train = np.reshape(y_train, (y_train.shape[0], 1))
    distance_train = distance.eval(feed_dict={images_L: X_train[:, :,0], images_R: X_train[:, :, 1], labels: label_train, dropout_param: 1.0})[:,0]
    distance_original_train = np.sqrt(np.sum(np.power(np.subtract(X_train[:, :, 0], X_train[:, :, 1]), 2), 1))
    print distance_original_train .shape

    # Test model on test samples
    label_test = np.reshape(y_test, (y_test.shape[0], 1))
    distance_test = distance.eval(
        feed_dict={images_L: X_test[:, :, 0], images_R: X_test[:, :, 1], labels: label_test, dropout_param: 1.0})[:,0]
    distance_original_test = np.sqrt(np.sum(np.power(np.subtract(X_test[:, :, 0], X_test[:, :, 1]), 2), 1))

# """
# TRAIN
# """
# Plot ROC for training
Plot_ROC_Fn(label_train, distance_original_train, 'Train', 'Input')
Plot_ROC_Fn(label_train, distance_train, 'Train', 'Output')

# # Plot PR for training
# Plot_PR_Fn(label_train, distance_original_train , 'Train', 'Input')
# Plot_PR_Fn(label_train, distance_train, 'Train', 'Output')
# #
# Plot HIST for training
Plot_HIST_Fn(label_train, distance_original_train, 'Train', 'Input')
Plot_HIST_Fn(label_train, distance_train, 'Train', 'Output')

"""
TEST
"""
# Plot ROC for test
Plot_ROC_Fn(label_test, distance_original_test, 'Test', 'Input')
Plot_ROC_Fn(label_test, distance_test, 'Test', 'Output')

# # Plot PR for test
# Plot_PR_Fn(label_test, distance_original_test , 'Test', 'Input')
# Plot_PR_Fn(label_test, distance_test, 'Test', 'Output')
# #
# Plot HIST for test
Plot_HIST_Fn(label_test, distance_original_test, 'Test', 'Input')
Plot_HIST_Fn(label_test, distance_test, 'Test', 'Output')


