
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

def generate_paths_and_labels(omission=None):
    image_paths = os.listdir(train_path)
    image_paths = [path for path in image_paths if path[0] != "0"]
    #train_image_paths = list(train_path.glob('*/*'))
    train_image_paths = [str(path) for path in image_paths if omission != path[:3]]
    test_image_paths = [str(path) for path in image_paths if omission == path[:3]]

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
    test_image_groups = [config.subjectKeys.get(int(path[0]), "none") for path in test_image_paths]

    train_image_labels = [label_to_index[group] for group in train_image_groups]
    test_image_labels = [label_to_index[group] for group in test_image_groups]
    #train_image_labels = [int(path[0]) for path in train_image_paths]

    # null tests, shuffle labels and randomize test
    if config.permuteLabels == True:
        random.shuffle(train_image_labels)
        rand_group = random.randint(0, 1)
        test_image_labels = [rand_group for label in test_image_labels]

    # force list of labels to numpy array
    train_image_labels = np.array(train_image_labels)
    test_image_labels = np.array(test_image_labels)

    return(train_image_paths, train_image_labels, test_image_paths, test_image_labels)

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
    os.mkdir(config.resultsPath)
except:
    print("Results dir already made.")

import matplotlib.pyplot as plt

train_path = pathlib.Path(config.source)
subject_list = get_avail_subjects()
print("List of available subjects:")
for sub in subject_list:
    print(sub)

# for iter in tqdm(range(1000)):
    # os.mkdir(config.resultsPath+"/iter"+str(iter))
for sub in tqdm(subject_list):
    train_paths, train_labels, test_paths, test_labels = generate_paths_and_labels(sub)

    train_data = load_numpy_stack(train_path, train_paths)
    test_data = load_numpy_stack(train_path, test_paths)

    fitted, modelvar = createModel(train_data, train_labels, rate, epochs, beta1, beta2)

    f = open(config.resultsPath+"/"+sub+".txt", 'w')

    score = modelvar.evaluate(test_data, test_labels)
    f.write("Group: " + repr(test_labels[0])+"\n")
    f.write("Loss: " + repr(score[0]) + "\n" + "Accuracy: " + repr(score[1]) + "\n")
    f.close()
