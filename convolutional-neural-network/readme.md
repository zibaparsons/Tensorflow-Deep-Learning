# Convolutional Neural Network

## Overview

The aim is to design a simple convolutional Neural Network using `TensorFlow`. The tutorial is aimed to sketch a starup model to the the two follwing:

1. Define an organization for the network architecture, training and evaluation phases.
2. Provides a template framework for constructing larger and more complicated models.

## Model Architecture

Two simple `convolutional layers`(each have max pooling) followed by two `fully-cnnected` layers conisdered. The number of output units for the last fully-connected layer is equal to the number of `classes` becasue a `softmax` has been implemented for the classification task.

## Code Organization

The source code is embeded in `code` folder.

| File                | Explanation   |
| ------------------- |:-------------:|
| Model_Functions.py  | The body of the framework which consists of structure and axillary functions |
| classifier.py       | The main file which has to be run |

## Input

The input format is `HDF5` for this implemetation but it basically can be anything as long as it satisfies the shape properties. For each `TRAIN` and `TEST` data, there are attributes call `cube` and `label`. The `cube` is the data of the shape `(Number_Samples,height,width,Number_Channels)` and the `label` is of the form `(Number_Samples,1)` in which each row has the class label. The label matrix should be transform to the form of `(Number_Samples,Number_Classes)` for which each row is associated with a sample and all of the columns are zero except for the one which belongs to the class number. Method `Reform_Fn` does this in the begining.

## Training

As conventional procedure, updating the gradient is done with batches of the data. Moreover `Batch Normalization` has been implemented for each convolutional layer. No `Batch Normalization` is done for the fully-connected layers. For all the convolutional and fully-connected layers, the `drop-out` has been used with the same parameter however this parameter can be customized for each layer in the code. Traditional `GradientDescentOptimizer` has been used.
