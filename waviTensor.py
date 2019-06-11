#!/usr/bin/env python3
#!pip3 install tensorflow==2.0.0-alpha0

from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import csv
import tensorflow as tf
import sys, os, re
import matplotlib as mpl

# make numpy values easier to read
np.set_printoptions(precision=3, suppress=True)

#dataDirectory=sys.argv[1]
os.chdir('PainStudyFiles/csv')
#os.listdir()
os.getcwd() #'../science/CANlab/WAViMedEEGScripts/PainStudyFiles/csv/'

def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    return sorted(data, key=alphanum_key)

# list of csv files to be loaded are those in the current directory
filenames=sorted_alphanumeric(os.listdir())
n=len(filenames)
print(filenames)

def get_dataset(filenames):
    dataset = tf.data.experimental.make_csv_dataset(
        filenames,
        251,
        column_names=['group','C1','C2','C3','C4','C5','C6','C7','C8','C9','C10','C11','C12','C13','C14','C15','C16','C17','C18','C19'],
        column_defaults=None,
        label_name='group',
        select_columns=None,
        field_delim=',',
        use_quote_delim=True,
        na_value='',
        header=False,
        num_epochs=None,
        shuffle=False,
        shuffle_buffer_size=10000,
        shuffle_seed=None,
        prefetch_buffer_size=tf.data.experimental.AUTOTUNE,
        num_parallel_reads=12,
        sloppy=False,
        num_rows_for_inference=100,
        compression_type=""
    )
    return dataset

get_dataset(filenames)

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
