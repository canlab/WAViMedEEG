#!/usr/bin/env python3
#!pip3 install tensorflow==2.0.0-alpha0

from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import csv
import tensorflow as tf
import sys, os, re

# make numpy values easier to read
np.set_printoptions(precision=3, suppress=True)

foldername=sys.argv[1]

LABELS = [0,1]
filenames = os.listdir(foldername)
n = len(filenames)

def printCSV(sub):
    with open(foldername+filenames[sub], newline='') as File:
        reader = csv.reader(File)
        for row in reader:
            print(row)

# def get_model(input_dim, hidden_units=[100]):
#     """
#     Create a Keras model with layers
#
#     Args:
#         input_dim: (int) The shape of an item in a batch
#         labels_dim: (int) The shape of a label
#         hidden_units: [int] the layer sizes of the DNN (input layer first)
#         learning_rate: (float) the learning rate for the optimizer
#
#     Returns:
#         A Keras model
#     """
#     inputs = tf.keras.Input(shape=(input_dim,))
#     x = inputs
#
#     for units in hidden_units:
#         x = tf.keras.layers.Dense(units, activation='relu')(x)
#     outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
#
#     model = tf.keras.Model(inputs, outputs)
#
#     return Model
#
# input_shape, output_shape = train_data.output_shapes
# input_dimension = input_shape.dims[1] # [0] is the batch size
#
# model = get_model(input_dimension)
# model.compile(
#     loss = 'binary_crossentropy',
#     optimizer = 'adam',
#     metrics = ['accuracy']
# )
# model.fit(train_data, epochs=20)
#
# test_loss, test_accuracy = model.evaluate(test_data)
#
# print('\n\nTest Loss {}, Test Accuracy {}'.format(test_loss, test_accuracy))
#
# predictions = model.predict(test_data)
