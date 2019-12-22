#!/usr/bin/python
# coding:utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import tensorflow as tf
# The MNIST dataset has 10 classes, representing the digits 0 through 9.
NUM_CLASSES = 10 
# The MNIST images are always 28x28 pixels.
IMAGE_SIZE = 28
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE

def inference(images, hidden1_units, hidden2_units):
  # Hidden 1
  with tf.name_scope('hidden1'):
    weights = tf.Variable(
        tf.truncated_normal([IMAGE_PIXELS, hidden1_units],
                            stddev=1.0 / math.sqrt(float(IMAGE_PIXELS))),
        name='weights')
    biases = tf.Variable(tf.zeros([hidden1_units]),
                         name='biases')
    hidden1 = tf.nn.relu(tf.matmul(images, weights) + biases)
  # Hidden 2
  with tf.name_scope('hidden2'):
    weights = tf.Variable(
        tf.truncated_normal([hidden1_units, hidden2_units],
                            stddev=1.0 / math.sqrt(float(hidden1_units))),
        name='weights')
    biases = tf.Variable(tf.zeros([hidden2_units]),
                         name='biases')
    hidden2 = tf.nn.relu(tf.matmul(hidden1, weights) + biases)
  # Linear
  with tf.name_scope('softmax_linear'):
    weights = tf.Variable(
        tf.truncated_normal([hidden2_units, NUM_CLASSES],
                            stddev=1.0 / math.sqrt(float(hidden2_units))),
        name='weights')
    biases = tf.Variable(tf.zeros([NUM_CLASSES]),
                         name='biases')
    logits = tf.matmul(hidden2, weights) + biases
  return logits

def loss(logits, labels):
  #print logits
  #print labels
  batch_size = tf.size(labels)
  labels = tf.expand_dims(labels, 1)
  indices = tf.expand_dims(tf.range(0, batch_size, 1), 1)
  concated = tf.concat(1, [indices, labels])
  onehot_labels = tf.sparse_to_dense(
      concated, tf.pack([batch_size, NUM_CLASSES]), 1.0, 0.0)
  #labels = tf.to_float(labels)
  cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
      labels=onehot_labels, logits=logits, name='xentropy')
  return tf.reduce_mean(cross_entropy, name='xentropy_mean')

def training(loss, learning_rate):
  # Add a scalar summary for the snapshot loss.
  tf.scalar_summary('loss', loss)
  # Create the gradient descent optimizer with the given learning rate.
  optimizer = tf.train.GradientDescentOptimizer(learning_rate)
  # Create a variable to track the global step.
  global_step = tf.Variable(0, name='global_step', trainable=False)
  # Use the optimizer to apply the gradients that minimize the loss
  # (and also increment the global step counter) as a single training step.
  train_op = optimizer.minimize(loss, global_step=global_step)
  return train_op

def evaluation(logits, labels):
  correct = tf.nn.in_top_k(logits, labels, 1)
  return tf.reduce_sum(tf.cast(correct, tf.int32))
