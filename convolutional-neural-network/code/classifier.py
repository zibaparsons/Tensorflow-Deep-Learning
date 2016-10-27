from __future__ import print_function
from Model_Functions import *
import tensorflow as tf
import sys

TRAIN_FILE = h5py.File('data/TRAIN.hdf5', 'r')
TEST_FILE = h5py.File('data/TEST.hdf5', 'r')

# Extracting the data and labels from HDF5.
# Pairs and labels have been saved separately in HDF5 files.
# The number of features and samples are extracted from data.
X_train = TRAIN_FILE['cube']
y_train = TRAIN_FILE['label']
X_test = TEST_FILE['cube']
y_test = TEST_FILE['label']

# Dimensions
num_samples = X_train.shape[0]
height = X_train.shape[1]
width = X_train.shape[2]
num_channels = X_train.shape[3]
N_classes = 259

# Reformat the labels from (N,1) to (N,M) which M is the number of classes
y_test = Reform_Fn(y_test, N_classes)
y_train = Reform_Fn(y_train, N_classes)

# Parameters
learning_rate = 0.01
batch_size = 256


# Defining place holders
image_place = tf.placeholder(tf.float32, shape=([None, height, width, num_channels]), name='image')
label_place = tf.placeholder(tf.float32, shape=([None, N_classes]), name='gt')
dropout_param = tf.placeholder(tf.float32)

# output of the network
pred = neural_network(image_place, dropout_param)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, label_place))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(label_place, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.initialize_all_variables()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    num_epoch = 10

    # iterate on epochs
    for epoch in range(num_epoch):
        total_batch = int(num_samples / batch_size)

        for i in range(total_batch):
            start_idx = i * batch_size
            end_idx = (i + 1) * batch_size

            # Fit training using batch data
            batch_x, batch_y = get_batch(start_idx, end_idx, X_train, y_train)

            # Run optimization op (backprop)
            sess.run(optimizer, feed_dict={image_place: batch_x, label_place: batch_y,
                                           dropout_param: 0.9})

            # Calculate batch loss and accuracy
            loss, acc = sess.run([cost, accuracy], feed_dict={image_place: batch_x,
                                                              label_place: batch_y,
                                                              dropout_param: 1.})
            print("batch " + str(i) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))
    print("Optimization Finished!")

    # Calculate accuracy for 256 mnist test images
    print("Testing Accuracy:", \
        sess.run(accuracy, feed_dict={image_place: X_test,
                                      label_place: y_test,
                                      dropout_param: 1.}))
