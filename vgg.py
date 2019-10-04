#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 19:23:01 2019

@author: xuwanqian
"""

from __future__ import absolute_import, division, print_function

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, layers
from loadBasicEnv import NAMEDICT, IMGSHAPE, NET_SAVE_NAME


plotx = []
ploty = []

# total classes (how many people)
num_classes = len(NAMEDICT)

# Training parameters.
learning_rate = 0.0001
training_steps = 2000
batch_size = 64
display_step = 100

# Create TF Model.
class ConvNet(Model):
    # Set layers.
    def __init__(self):
        super().__init__()
        # Convolution Layer with 32 filters and a kernel size of 5.
        self.conv11 = layers.Conv2D(64,kernel_size=[3,3],strides=[1,1],padding='same',use_bias=True, activation='relu')
        self.conv12 = layers.Conv2D(64,kernel_size=[3,3],strides=[1,1],padding='same',use_bias=True, activation='relu')
        self.maxpool1 = layers.MaxPool2D(2, strides=2)
        self.conv21 = layers.Conv2D(128,kernel_size=[3,3],strides=[1,1],padding='same',use_bias=True, activation='relu')
        self.conv22 = layers.Conv2D(128,kernel_size=[3,3],strides=[1,1],padding='same',use_bias=True, activation='relu')
        self.maxpool2 = layers.MaxPool2D(2, strides=2)
        self.conv31 = layers.Conv2D(256,kernel_size=[3,3],strides=[1,1],padding='same',use_bias=True, activation='relu')
        self.conv32 = layers.Conv2D(256,kernel_size=[3,3],strides=[1,1],padding='same',use_bias=True, activation='relu')
        self.conv33 = layers.Conv2D(256,kernel_size=[3,3],strides=[1,1],padding='same',use_bias=True, activation='relu')
        self.maxpool3 = layers.MaxPool2D(2, strides=2)
        self.conv41 = layers.Conv2D(512,kernel_size=[3,3],strides=[1,1],padding='same',use_bias=True, activation='relu')
        self.conv42 = layers.Conv2D(512,kernel_size=[3,3],strides=[1,1],padding='same',use_bias=True, activation='relu')
        self.conv43 = layers.Conv2D(512,kernel_size=[3,3],strides=[1,1],padding='same',use_bias=True, activation='relu')
        self.maxpool4 = layers.MaxPool2D(2, strides=2)
        self.conv51 = layers.Conv2D(512,kernel_size=[3,3],strides=[1,1],padding='same',use_bias=True, activation='relu')
        self.conv52 = layers.Conv2D(512,kernel_size=[3,3],strides=[1,1],padding='same',use_bias=True, activation='relu')
        self.conv53 = layers.Conv2D(512,kernel_size=[3,3],strides=[1,1],padding='same',use_bias=True, activation='relu')
        self.maxpool5 = layers.MaxPool2D(2, strides=2)
        # Flatten the data to a 1-D vector for the fully connected layer.
        self.flatten = layers.Flatten()
        # Fully connected layer.
        self.fc6 = layers.Dense(4096,use_bias=True,activation='relu')
        self.fc7 = layers.Dense(4096,use_bias=True,activation='relu')
        self.fc8 = layers.Dense(num_classes,use_bias=True,activation=None)

    # Set forward pass.
    def call(self, x, is_training=False):
        x = tf.reshape(x, [-1, IMGSHAPE, IMGSHAPE, 1])
        x = self.conv11(x)
        x = self.maxpool1(x)
        x = self.conv21(x)
        x = self.maxpool2(x)
        x = self.conv31(x)
        x = self.maxpool3(x)
        x = self.conv41(x)
        x = self.maxpool4(x)
        x = self.conv51(x)
        x = self.maxpool5(x)
        x = self.flatten(x)
        x = self.fc6(x)
        x = self.fc7(x)
        x = self.fc8(x)
        if not is_training:
            # tf cross entropy expect logits without softmax, so only
            # apply softmax when not training.
            x = tf.nn.softmax(x)
        return x


# Cross-Entropy Loss.
# Note that this will apply 'softmax' to the logits.
def cross_entropy_loss(x, y):
    # Convert labels to int 64 for tf cross-entropy function.
    y = tf.cast(y, tf.int64)
    # Apply softmax to logits and compute cross-entropy.
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=x)
    # Average loss across the batch.
    return tf.reduce_mean(loss)

# Accuracy metric.
def accuracy(y_pred, y_true):
    # Predicted class is the index of highest score in prediction vector (i.e. argmax).
    correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.cast(y_true, tf.int64))
    return tf.reduce_mean(tf.cast(correct_prediction, tf.float32), axis=-1)


# Optimization process.
def run_optimization(x, y):
    # Wrap computation inside a GradientTape for automatic differentiation.
    with tf.GradientTape() as g:
        # Forward pass.
        pred = conv_net(x, is_training=True)
        # Compute loss.
        loss = cross_entropy_loss(pred, y)

    # Variables to update, i.e. trainable variables.
    trainable_variables = conv_net.trainable_variables

    # Compute gradients.
    gradients = g.gradient(loss, trainable_variables)

    # Update W and b following gradients.
    optimizer.apply_gradients(zip(gradients, trainable_variables))

if __name__=="__main__":
    from loadData import load_data

    # import Data from loadData.py
    (x_train, y_train), (x_test, y_test) = load_data()
    # Convert to float32.
    x_train, x_test = np.array(x_train, np.float32), np.array(x_test, np.float32)
    # Normalize images value from [0, 255] to [0, 1].
    x_train, x_test = x_train / 255., x_test / 255.


    # Use tf.data API to shuffle and batch data.
    train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    #  train_data = train_data.repeat().shuffle(1000).batch(batch_size).prefetch(1)
    train_data = train_data.repeat().shuffle(5000).batch(batch_size).prefetch(1)
    #  from tensorflow.keras.datasets import mnist
    #  (x_train, y_train), (x_test, y_test) = mnist.load_data()
    #  # Convert to float32.
    #  x_train, x_test = np.array(x_train, np.float32), np.array(x_test, np.float32)
    #  # Normalize images value from [0, 255] to [0, 1].
    #  x_train, x_test = x_train / 255., x_test / 255.
    #
    #
    #  # Use tf.data API to shuffle and batch data.
    #  train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    #  train_data = train_data.repeat().shuffle(10000).batch(batch_size).prefetch(1)
    # Run training for the given number of steps.

    # Build neural network model.
    conv_net = ConvNet()
    # Stochastic gradient descent optimizer.
    optimizer = tf.optimizers.Adam(learning_rate)

    for step, (batch_x, batch_y) in enumerate(train_data.take(training_steps), 1):
        # Run the optimization to update W and b values.
        run_optimization(batch_x, batch_y)
        pred = conv_net(batch_x)
        loss = cross_entropy_loss(pred, batch_y)
        acc = accuracy(pred, batch_y)
        ploty.append(loss)
        plotx.append(acc)

        if step % display_step == 0:
            acc = accuracy(pred, batch_y)
            print("step: %i, loss: %f, accuracy: %f" % (step, loss, acc))

    # Test model on validation set.
    pred = conv_net(x_test[:100])
    print("Test Accuracy: %f" % accuracy(pred, y_test[:100]))

    # Visualize predictions.
    import matplotlib.pyplot as plt


    # Predict 10 images from validation set.
    n_images = 10
    test_images = x_test[:n_images]
    predictions = conv_net(test_images)

    # Display image and model prediction.
    for i in range(n_images):
        plt.imshow(np.reshape(test_images[i], [IMGSHAPE, IMGSHAPE]), cmap='gray')
        plt.show()
        rawPredict = np.argmax(predictions.numpy()[i])
        rawReal = y_test[i]
        print(f"Model prediction: {NAMEDICT[rawPredict]}; real: {NAMEDICT[rawReal]}")

    conv_net.save_weights(f"./modelSave/{NET_SAVE_NAME}")
    plt.plot(ploty)
    plt.plot(plotx)
    plt.show()