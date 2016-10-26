"""
This file contains necessary definitions for Siamese Architecture implementation.
"""

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


def _activation_summary(x):
    """Helper to create summaries for activations.

    Creates a summary that provides a histogram of activations.
    Creates a summary that measure the sparsity of activations.

    Args:
      x: Tensor
    Returns:
      nothing
    """
    # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
    # session. This helps the clarity of presentation on tensorboard.
    # tensor_name = re.sub('%s_[0-9]*/' % 'tower', '', x.op.name)
    tf.histogram_summary(x.op.name + '/activations', x)
    tf.scalar_summary(x.op.name + '/sparsity', tf.nn.zero_fraction(x))


def _variable_on_cpu(name, shape, initializer):
    """Helper to create a Variable stored on CPU memory.

    Args:
      name: name of the variable
      shape: list of ints
      initializer: initializer for Variable

    Returns:
      Variable Tensor
    """
    with tf.device('/cpu:0'):
        var = tf.get_variable(name, shape, initializer=initializer)
    return var


def _variable_with_weight_decay(name, shape, stddev, wd):
    """Helper to create an initialized Variable with weight decay.

    Note that the Variable is initialized with a truncated normal distribution.
    A weight decay is added only if one is specified.

    Args:
      name: name of the variable
      shape: list of ints
      stddev: standard deviation of a truncated Gaussian
      wd: add L2Loss weight decay multiplied by this float. If None, weight
          decay is not added for this Variable.

    Returns:
      Variable Tensor
    """
    var = _variable_on_cpu(name, shape,
                           tf.truncated_normal_initializer(stddev=stddev))
    if wd is not None:
        weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var


def max_pool(x, pool_size, stride, name='max_pooling'):
    """This is the max pooling layer..

    ## Note1: The padding is of type 'VALID'. For more information please
    refer to TensorFlow tutorials.
    ## Note2: Variable scope is useful for sharing the variables.

    Args:
      x: The input of the layer which most probably in the output of previous Convolution layer.
      pool_size: The windows size for max pooling operation.
      stride: stride of the max pooling layer

    Returns:
      The resulting feature cube.
    """
    with tf.variable_scope(name):
        return tf.nn.max_pool(x, ksize=[1, pool_size, pool_size, 1], strides=[1, stride, stride, 1], padding='VALID')


#
def conv_relu(input, kernel_shape, bias_shape, name='conv_relu'):
    """This is the Convolutional layer..

        ## Note 1: The padding is of type 'VALID'. For more information please
        refer to TensorFlow tutorials.
        ## Note 2: Variable scope is useful for sharing the variables.
        ## Note 3: The weight_decay can be used for this layer.
        ## Note 4: Saving the activation summary for this layer, can be useful.
        ## Note 5: ReLU activation can be replaced.

        Args:
          input: The input of the layer.
          kernel_shape: The kernel windows for scanning.
          bias_shape: the bias parameter shape.

        Returns:
          The resulting feature cube.
        """
    with tf.variable_scope(name):
        kernel = _variable_with_weight_decay('weights', kernel_shape,
                                             stddev=0.01, wd=None)
        conv = tf.nn.conv2d(input, kernel, [1, 1, 1, 1], padding='VALID')
        biases = _variable_on_cpu('biases', bias_shape, tf.constant_initializer(0.1))
        bias = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(bias, name=name)
        _activation_summary(conv1)
        return conv1, _activation_summary(conv1)


def convolution_relu(input, kernel_size, num_outputs, name):

    with tf.variable_scope(name):
        conv = tf.contrib.layers.convolution2d(input,
                                                 num_outputs,
                                                 kernel_size=kernel_size,
                                                 stride=[1, 1],
                                                 padding='VALID',
                                                 activation_fn=tf.nn.relu,
                                                 normalizer_fn=tf.contrib.layers.batch_norm,
                                                 normalizer_params=None,
                                                 weights_initializer=tf.contrib.layers.xavier_initializer(
                                                     dtype=tf.float32)
                                                 )
    _activation_summary(conv)
    return conv


#
def FC_layer(input, kernel_shape, bias_shape, name="FC_layer"):
    """This is the Fully-Connected layer.

            ## Note1: ReLU activation can be replaced.
            ## Note2: tf.reshape(input, [-1, kernel_shape[0]]) is for getting the
                       proper dimension to be multiplied by the fully connected weight matrix.

            Args:
              input: The input of the layer which most probably in the output of previous Convolution layer.
              kernel_shape: The kernel windows for scanning.
              bias_shape: the bias parameter shape.

            Returns:
              The feature vector.
            """
    with tf.variable_scope(name):
        weights = tf.get_variable('weights', kernel_shape, tf.float32,
                                  tf.truncated_normal_initializer(mean=0.0, stddev=0.01))
        biases = tf.get_variable('biases', bias_shape, tf.float32,
                                 tf.truncated_normal_initializer(mean=0.01, stddev=0.0))
        input = tf.reshape(input, [-1, kernel_shape[0]])
        return tf.nn.relu(tf.matmul(input, weights) + biases)


def loss(y, distance, batch_size):
    """With this definition the loss will be calculated.
        # loss is the contrastive loss plus the loss caused by
        # weight_dacay parameter(if activated). The kernel of conv_relu is
        # the place for activation of weight decay. The scale of weight_decay
        # loss might not be compatible with the contrastive loss.


        Args:
          y: The labels.
          distance: The distance vector between the output features..
          batch_size: the batch size is necessary because the loss calculation would be over each batch.

        Returns:
          The total loss.
        """

    margin = 1
    term_1 = y * tf.square(distance)
    # tmp= tf.mul(y,tf.square(d))
    term_2 = (1 - y) * tf.square(tf.maximum((margin - distance), 0))
    Contrastive_Loss = tf.reduce_sum(term_1 + term_2) / batch_size / 2
    tf.add_to_collection('losses', Contrastive_Loss)

    return tf.add_n(tf.get_collection('losses'), name='total_loss')


# Accuracy computation
def compute_accuracy(prediction, labels):
    return labels[prediction.ravel() < 0.5].mean()
    # return tf.reduce_mean(labels[prediction.ravel() < 0.5])


# Extracting the data and label for each batch.
def get_batch(start_idx, end_ind, inputs, labels):
    """Get each batch of data which is necessary for the training and testing phase..

        Args:
          start_idx: The start index of the batch in the whole sample size.
          end_idx: The ending index of the batch in the whole sample size.
          inputs: Train/test data.
          labels: Train/test labels.

        Returns:
          pair_left_part: Batch of left images in pairs.
          pair_right_part: Batch of right images in pairs.
          y: The associated labels to the images pairs.

        """
    num_orientation = inputs.shape[3] / 2
    pair_left_part = inputs[start_idx:end_ind, :, :, 0:num_orientation]
    pair_right_part = inputs[start_idx:end_ind, :, :, num_orientation:]
    y = np.reshape(labels[start_idx:end_ind], (len(range(start_idx, end_ind)), 1))
    return pair_left_part, pair_right_part, y


def Siamese_Structure(X, dropout):
    """This function create each branch os Siamese Architecture.
       Basically there are not two branches. They are the same!!

        Args:
          X: The input image(batch).

        Returns:
          The whole NN model.

        """
    MODEL = neural_network(X, dropout)
    return MODEL


def neural_network(x, dropout):
    """This is the whole structure of the CNN.

       Nore: Although the dropout left untouched, it can be define for the FC layers output.

         Args:
           X: The input image(batch).

         Returns:
           The output feature vector.

         """

    ################## SECTION - 1 ##############################

    # Conv_11 layer
    NumFeatureMaps = 64
    kernel_height = 3
    kernel_width = 3
    kernel_size = [kernel_height, kernel_width]
    relu11 = convolution_relu(x, kernel_size, NumFeatureMaps, name='conv11')

    # Conv_12 layer
    # Conv_11 layer
    NumFeatureMaps = 64
    kernel_height = 3
    kernel_width = 3
    kernel_size = [kernel_height, kernel_width]
    relu12 = convolution_relu(relu11, kernel_size, NumFeatureMaps, name='conv12')

    # Pool_1 layer
    pool_1 = max_pool(relu12, 2, 2, name='pool_1')
    pool_1_shape = pool_1.get_shape()

    ###########################################################
    ##################### SECTION - 2 #########################
    ###########################################################

    # Conv_21 layer
    # Number of feature maps
    NumFeatureMaps = 128
    kernel_height = 3
    kernel_width = 3
    kernel_size = [kernel_height, kernel_width]
    relu21 = convolution_relu(pool_1,
                                                         kernel_size, NumFeatureMaps, name='conv21')
    # Conv_22 layer
    NumFeatureMaps = 128
    kernel_height = 3
    kernel_width = 3
    kernel_size = [kernel_height, kernel_width]
    relu22 = convolution_relu(relu21,
                                                         kernel_size, NumFeatureMaps, name='conv22')

    # Pool_2 layer
    pool_2 = max_pool(relu22, 2, 2, name='pool_2')
    pool_2_shape = pool_2.get_shape()

    ###########################################################
    ##################### SECTION - 3 #########################
    ###########################################################

    # Conv_31 layer
    # Number of feature maps
    # Conv_22 layer
    NumFeatureMaps = 256
    kernel_height = 3
    kernel_width = 3
    kernel_size = [kernel_height, kernel_width]
    relu31 = convolution_relu(pool_2,
                                                         kernel_size, NumFeatureMaps, name='conv31')

    # Conv_32 layer
    NumFeatureMaps = 256
    kernel_height = 3
    kernel_width = 3
    kernel_size = [kernel_height, kernel_width]
    relu32 = convolution_relu(relu31,
                                                         kernel_size, NumFeatureMaps, name='conv32')

    # Pool_3 layer
    pool_3 = max_pool(relu32, 2, 2, name='pool_3')
    pool_3_shape = pool_3.get_shape()

    ###########################################################
    ##################### SECTION - 4 #########################
    ################# Fully Connected Layes####################

    # Fully_Connected layer - 1
    shape = pool_3_shape
    num_input_for_fc_layer = int(shape[1] * shape[2] * shape[3])
    NumOutput = 2048
    y_f1 = FC_layer(pool_3, [num_input_for_fc_layer, NumOutput], [NumOutput], name='fc_1')

    # Fully_Connected layer - 2
    shape = y_f1.get_shape()
    num_input_for_fc_layer = int(shape[1])
    NumOutput = 1024
    y_f2 = FC_layer(pool_3, [num_input_for_fc_layer, NumOutput], [NumOutput], name='fc_2')

    return y_f2
