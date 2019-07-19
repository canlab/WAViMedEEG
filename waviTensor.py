d#!/usr/bin/env python3

# Usage: ./waviTensor.py <folder containing CSV files> <test data folder>

from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plot
import numpy as np
import csv
import sys, os, re

# trying new vis package
import mne

# make numpy values easier to read
np.set_printoptions(precision=3, suppress=True)

train_file_path=sys.argv[1]
train_filenames_glob = tf.io.gfile.glob(train_file_path+"*.csv")

test_file_path=sys.argv[2]
test_filenames_glob = tf.io.gfile.glob(test_file_path+"*.csv")

# functions to help us plot our results
def plot_image(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
    plot.grid(False)
    plot.xticks([])
    plot.yticks([])

    plot.imshow(img, cmap=plot.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plot.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
    100*np.max(predictions_array),
    class_names[true_label]),
    color=color)

def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array[i], true_label[i]
    plot.grid(False)
    plot.xticks([])
    plot.yticks([])
    thisplot = plot.bar(range(10), predictions_array, color="#777777")
    plot.ylim([0, 1])
    predicted_label = lnp.argmax(predictions_array)

    thisplot[predicted_label].set_color('blue')

# creation of actual tensor dataset
def get_dataset(file_pattern):
    dataset = tf.data.experimental.make_csv_dataset(
        file_pattern,
        batch_size = 64,
        column_names=['group','C1','C2','C3','C4','C5','C6','C7','C8','C9','C10','C11','C12','C13','C14','C15','C16','C17','C18','C19'],
        column_defaults=None,
        label_name='group',
        select_columns=None,
        field_delim=',',
        use_quote_delim=False,
        na_value='',
        header=False,
        num_epochs=5,
        shuffle=True,
        shuffle_buffer_size=128,
        shuffle_seed=None,
        prefetch_buffer_size=tf.data.experimental.AUTOTUNE,
        num_parallel_reads=12,
        sloppy=False,
        num_rows_for_inference=50,
        compression_type=""
    )
    return dataset

# declare class names
LABELS = [0, 1]

train_data = get_dataset(train_filenames_glob)
test_data = get_dataset(test_filenames_glob)



# print to see basic TF feature / label tuple
# examples, labels = next(iter(train_data))
# print("EXAMPLES: \n", examples, "\n")
# print("LABELS: \n", labels)

def get_model(input_dim, hidden_units=[100]):
    """
    Create a Keras model with layers

    Args:
        input_dim: (int) The shape of an item in a batch
        labels_dim: (int) The shape of a label
        hidden_units: [int] the layer sizes of the DNN (input layer first)
        learning_rate: (float) the learning rate for the optimizer

    Returns:
        A Keras model
    """
    inputs = tf.keras.Input(shape=(input_dim,))
    x = inputs

    for units in hidden_units:
        x = tf.keras.layers.Dense(units, activation='relu')(x)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)

    model = tf.keras.Model(inputs, outputs)

    return model

input_shape, output_shape = tf.data.get_output_shapes(train_data)
input_dimension = input_shape.dims(1) # [0] is the batch size

model = get_model(input_dimension)
model.compile(
    loss = 'binary_crossentropy',
    optimizer = 'adam',
    metrics = ['accuracy']
)
model.fit(train_data, epochs=5)

# check accuracy on test_data set
test_loss, test_accuracy = model.evaluate(test_data)
print('\n\nTest Loss {}, Test Accuracy {}'.format(test_loss, test_accuracy))

#infer labels on a batch or a dataset of batches
predictions = model.predict(test_data)
