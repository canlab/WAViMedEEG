
from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model

import matplotlib.pyplot as plot
import numpy as np
from numpy import genfromtxt
import csv
import sys, os, re
import pathlib
import config
from tqdm import tqdm
import random

print("Using Tensorflow version", tf.__version__)

# get all subject numbers in use
def get_avail_subjects():
    subs = []
    for file in os.listdir(train_path):
        if file[:3] not in subs:
            if file[0] != "0":
                subs.append(file[:3])
    return(subs)

def generate_paths_and_labels():
    image_paths = os.listdir(train_path)
    image_paths = [path for path in image_paths if path[0] != "0"]
    train_image_paths = [str(path) for path in image_paths]

    random.shuffle(train_image_paths)

    train_count = len(train_image_paths)
    print("You have", train_count, "files.")

    #list the available labels
    label_names = ['pain', 'ctrl']
    #label_names = sorted(item.name for item in train_path.glob('*/') if item.is_dir())
    print("Labels discovered:", label_names)

    #assign an index to each label
    label_to_index = dict((name, index) for index, name in enumerate(label_names))
    print("Label indices:", label_to_index)

    #create a list of every file and its integer label
    train_image_groups = [config.subjectKeys.get(int(path[0]), "none") for path in train_image_paths]

    train_image_labels = [label_to_index[group] for group in train_image_groups]

    # force list of labels to numpy array
    train_image_labels = np.array(train_image_labels)

    return(train_image_paths, train_image_labels)

def reshape_paths(path_list, path_labels):
    fnames_stacked = []
    paths_and_labels = list(zip(path_list, path_labels))
    for band in config.frequency_bands:
        band_names = [pair for pair in paths_and_labels if band[0] in pair[0]]
        band_names = sorted(band_names)
        fnames_stacked.append(band_names)
    return(fnames_stacked)

def normalize(array):
    nom = (array - array.min())*(1 - 0)
    denom = array.max() - array.min()
    denom+=(10**-10)
    return 0 + nom/denom

def load_numpy_stack(lead, paths):
    numpy_stack = []
    label_stack = []
    i=0
    # once on every contig we have in paths
    print("\nLoading", config.selectedTask, "contigs into Tensor objects")
    print("==========\n")
    pbar_len = len(paths[0])*len(paths)
    pbar = tqdm(total=pbar_len)
    while i < len(paths[0]):
        contigs_each_band = []
        for band in paths:
            fpath = str(lead)+"/"+str(band[i][0])
            array = genfromtxt(fpath, delimiter=",")
            array = normalize(array)
            contigs_each_band.append(array)
            pbar.update(1)
        contigs_each_band = np.stack(contigs_each_band, axis=2)
        numpy_stack.append(contigs_each_band)
        if band[i][0][0] == "1":
            label_stack.append(0)
        elif band[i][0][0] == "2":
            label_stack.append(1)
        i+=1
    pbar.close()
    # labels as array, for each contig
    label_stack = np.array(label_stack)
    # make samples 1st-read axis
    numpy_dataset = np.stack(numpy_stack, axis=0)

    # randomly shuffle them, with same permutation
    idx = np.random.permutation(len(numpy_dataset))
    numpy_dataset, label_stack = numpy_dataset[idx], label_stack[idx]

    return(numpy_dataset, label_stack)

def createModel(train_arrays, train_image_labels, learn, num_epochs, betaOne, betaTwo):
    # Introduce sequential Set
    model = tf.keras.models.Sequential()

    # Create a convolutional base
    model.add(tf.keras.layers.Conv2D(5, kernel_size=6, strides=6, padding='same', activation='relu', data_format='channels_last', use_bias=False))
    model.add(tf.keras.layers.Conv2D(5, kernel_size=6, strides=6, padding='same', activation='relu', data_format='channels_last', use_bias=False))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=None, padding='same', data_format=None))
    model.add(tf.keras.layers.Dropout(0.2))

    model.add(tf.keras.layers.Conv2D(5, kernel_size=6, strides=6, padding='same', activation='relu', data_format='channels_last', use_bias=False))
    model.add(tf.keras.layers.Conv2D(5, kernel_size=6, strides=6, padding='same', activation='relu', data_format='channels_last', use_bias=False))
    #model.add(tf.keras.layers.Conv2D(5, kernel_size=5, strides=5, padding='same', activation='relu', data_format='channels_last', use_bias=False))
    model.add(tf.keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None))

    # Layers
    #model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=None, padding='same', data_format=None))
    model.add(tf.keras.layers.Flatten(data_format="channels_last"))

    # Hidden layers
    model.add(tf.keras.layers.Dense(2, activation='softmax', use_bias=True))

    model.build(train_arrays.shape)
    model.summary()

    # Model compilation
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learn, beta_1=betaOne, beta_2=betaTwo),
                 loss='sparse_categorical_crossentropy',
                 metrics=['accuracy'])

    # Include the epoch in the file name (uses `str.format`)
    checkpoint_path = "pain_model/cp-{epoch:04d}.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)

    # Create a callback that saves the model's weights every 5 epochs
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        verbose=1,
        save_weights_only=True,
        period=50)

    # Save the weights using the `checkpoint_path` format
    model.save_weights(checkpoint_path.format(epoch=0))

    # Train the model with the new callback
    history = model.fit(train_arrays, train_image_labels, epochs=num_epochs,
                       validation_split=0.33, callbacks=[cp_callback])

    return(history, model)

try:
    rate = config.learningRate
except:
    rate = float(input("What's my learning rate? \n"))

try:
    beta1 = config.betaOne
except:
    beta1 = float(input("What's my beta1? \n"))

try:
    beta2 = config.betaTwo
except:
    beta2 = float(input("What's my beta2? \n"))

try:
    epochs = config.numEpochs
except:
    epochs = int(input("How many epochs? \n"))

try:
    os.mkdir(config.resultsPath)
except:
    print("Results dir already made.")

import matplotlib.pyplot as plt

train_path = pathlib.Path(config.source)
subject_list = get_avail_subjects()
print("List of available subjects:")
for sub in subject_list:
    print(sub)

train_paths, train_labels = generate_paths_and_labels()

train_paths_and_labels = reshape_paths(train_paths, train_labels)

train_data, train_labels = load_numpy_stack(train_path, train_paths_and_labels)

fitted, modelvar = createModel(train_data, train_labels, rate, epochs, beta1, beta2)
modelvar.save('pain_model/MyModel')
