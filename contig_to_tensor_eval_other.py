
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

print("Using Tensorflow version", tf.__version__)

import random

def generate_subjects(contig_path):
    fnames = os.listdir(contig_path)
    subs = []
    for fname in fnames:
        if fname[:config.participantNumLen] not in subs:
            if fname[0] != "0":
                subs.append(fname[:config.participantNumLen])
    return(subs)

def generate_paths_and_labels(contigs_path, omission=[]):
    fnames = os.listdir(contigs_path)

    image_paths = [str(path) for path in fnames if path[0] != "0"]
    image_paths = [path for path in image_paths if path[:3] not in omission]
    #train_image_paths = list(train_path.glob('*/*'))
    random.shuffle(image_paths)

    image_count = len(image_paths)
    print("You have loaded", image_count, "path / label pairs.")

    #list the available labels
    label_names = list(config.subjectKeys.values())
    label_names = [label for label in label_names if label != "pilt"]
    print("Labels discovered:", label_names)

    #assign an index to each label
    label_to_index = dict((name, index) for index, name in enumerate(label_names))
    print("Label indices:", label_to_index)

    #create a list of every file and its integer label

    # this \/ gets labels from first 4 characters of filename
    #train_image_labels = [label_to_index[path[:4]] for path in train_image_paths]
    # this \/ gets labels from parent folder
    #train_image_labels = [label_to_index[pathlib.Path(path).parent.name]
     #                   for path in train_image_paths]

    image_groups = [config.subjectKeys.get(int(path[0]), "none") for path in image_paths]

    image_labels = [label_to_index[group] for group in image_groups]

    # force list of labels to numpy array
    image_labels = np.array(image_labels)

    return(image_paths, image_labels)

def normalize(array):
    nom = (array - array.min())*(1 - 0)
    denom = array.max() - array.min()
    denom+=(10**-10)
    return 0 + nom/denom

def load_numpy_stack(lead, paths):
    numpy_stack = []
    for path in paths:
        path = str(lead)+"/"+str(path)
        array = genfromtxt(path, delimiter=",")
        array = normalize(array)
        array = array.reshape(array.shape +(1,))
        numpy_stack.append(array)
    numpy_dataset = np.rollaxis(np.block(numpy_stack), 2, 0)
    numpy_dataset = numpy_dataset.reshape(numpy_dataset.shape+(1,))
    print("Original Shape of Dataset:", numpy_dataset.shape, "\n")
    return(numpy_dataset)

# def filter_my_channels(dataset, keep_channels, axisNum):
#     filter_indeces = []
#
#     # for each channel in my channel list
#     for keep in keep_channels:
#         filter_indeces.append(config.channel_names.index(keep))
#     #   get the index of that channel in the master channel list
#     #   add that index number to a new list filter_indeces
#
#     # iter over the rows of axis 2 (in 0, 1, 2, 3 4-dimensional dataset)
#     filtered_dataset = np.take(dataset, filter_indeces, axisNum)
#     print("New Shape of Dataset:", filtered_dataset.shape, "\n")
#     return(filtered_dataset)

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
    model.add(tf.keras.layers.Dense(2, activation='softmax'))

    model.build(train_arrays.shape)
    model.summary()

    # Model compilation
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learn, beta_1=betaOne, beta_2=betaTwo),
                 loss='sparse_categorical_crossentropy',
                 metrics=['accuracy'])

    history = model.fit(train_arrays, train_image_labels, epochs=num_epochs,
                       validation_split=0.33)

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
    os.mkdir(config.resultsDir)
except:
    print("Results dir already made.")

subject_list = generate_subjects(config.evalPath)
print("List of subjects being evaluated:")
for sub in subject_list:
    print(sub)

train_path = pathlib.Path(config.source)
eval_path = pathlib.Path(config.evalPath)

train_paths, train_labels = generate_paths_and_labels(train_path)

train_data = load_numpy_stack(train_path, train_paths)

fitted, modelvar = createModel(train_data, train_labels, rate, epochs, beta1, beta2)

for sub in tqdm(subject_list):
    eval_paths, eval_labels = generate_paths_and_labels(eval_path, omission=[omit for omit in subject_list if omit != sub])
    eval_data = load_numpy_stack(eval_path, eval_paths)

    f = open(config.resultsDir+"/"+sub+".txt", 'w')
    f.write("Subject accuracy:\n")

    score = modelvar.evaluate(eval_data, eval_labels)

    f.write(repr(score[1])+"\n")
    f.close()
