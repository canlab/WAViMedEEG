from __future__ import absolute_import, division, print_function, unicode_literals
import sys, os, re
import numpy as np
from numpy import genfromtxt
import csv
from tqdm import tqdm
# tensorflow imports
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Dense, Flatten, Conv2D, LeakyReLU, MaxPooling2D, Dropout, BatchNormalization
import datetime

print("Using Tensorflow version", tf.__version__)

# get all subject numbers in use
def get_avail_subjects(train_path, subLen=4):
    subs = []
    for file in os.listdir(train_path):
        if file[:subLen] not in subs:
            if file[0] != "0":
                subs.append(file[:subLen])
    return(subs)

def get_pilot_subjects(train_path):
    subs = []
    for file in os.listdir(train_path):
        if file[:subLen] not in subs:
            if file[0] == "0":
                subs.append(file[:subLen])
    return(subs)

# generate structure of train paths for contigs
# if omissions are given, assumed to be test_image_paths,
# as seen in convnet.cross_val
def generate_paths_and_labels(train_path, subList, omission=None, subLen=4):
    image_paths = os.listdir(train_path)
    image_paths = [path for path in image_paths if path[:subLen] in subList]
    train_image_paths = [str(path) for path in image_paths if omission != path[:subLen]]
    test_image_paths = [str(path) for path in image_paths if omission == path[:subLen]]

    train_count = len(train_image_paths)
    print("You have", train_count, "files.")

    return(train_image_paths, test_image_paths)

# this will reshape paths to accomodate a new dimension for bandpassed data
def reshape_paths_with_bands(path_list, frequency_bands):
    fnames_stacked = []
    for band in frequency_bands:
        band_names = [path for path in path_list if band[0] in path]
        band_names = sorted(band_names)
        fnames_stacked.append(band_names)
    return(fnames_stacked) # list of paths for each band, fnames_stacked is list equal length to frequency_bands
    # each item in list contains paths for that band, equal to total number of contigs collected

# linearly normalizes a contig array to be between 1 and 0
def normalize(array):
    nom = (array - array.min())*(1 - 0)
    denom = array.max() - array.min()
    denom+=(10**-10)
    return 0 + nom/denom

# loads in list of paths
# returns paths as desired data shape:
# (samples, contig length, frequency bands, electrodes)
# and labels, as 1-dimensional list
def load_numpy_stack(lead, paths, permuteLabels=False):
    numpy_stack = []
    index_stack = []
    label_stack = []
    i=0
    # once on every contig we have in paths
    print("\nLoading contigs into Tensor objects")
    print("==========\n")
    pbar_len = len(paths[0])*len(paths)
    pbar = tqdm(total=pbar_len)
    while i < len(paths[0]): # while i less than number of contigs
        contigs_each_band = []
        index_stack.append(paths[0][i].split('_')[2][:-4])
        for band in paths: # for each frequency band
            fpath = str(lead)+"/"+str(band[i]) # set path to contig i in that band
            array = genfromtxt(fpath, delimiter=",") # read that array
            array = normalize(array)
            contigs_each_band.append(array) # add it to list of arrays for that contig
            pbar.update(1)
        contigs_each_band = np.block(contigs_each_band) # collapse list into last np axis
        # shape of contigs_each_band: (contig length, n electrodes, n freq bands)
        numpy_stack.append(contigs_each_band)
        # if fname index label pain
        if band[i][0][0] == "1":
            label_stack.append(0)
        # if fname index label control
        elif band[i][0][0] == "2":
            label_stack.append(1)
        # if pilot assume control
        elif band[i][0][0] == "0":
            label_stack.append(1)
        i+=1
    pbar.close()
    # start_indexes as array, for each contig
    index_stack = np.array(index_stack)

    # labels as array, for each contig
    label_stack = np.array(label_stack)
    # make samples 1st-read axis
    numpy_dataset = np.stack(numpy_stack, axis=0) # collapse list into np axis
    numpy_dataset = np.expand_dims(numpy_dataset,-1)
    # new shape of structure: (n contigs, contig length, n electrodes, n freq bands)
    # numpy_dataset = numpy_dataset.transpose(0, 1, 3, 2) # reshape to:
    # (n contigs, contig length, n frequency bands, n electrodes)

    # if permute labels trigger on, random sort train labels
    if permuteLabels == True:
        idy = np.random.permutation(len(train_labels))
        train_labels = train_labels[idy]

    return(numpy_dataset, index_stack, label_stack)

def load_numpy_stack_age(lead, paths, permuteLabels=False):
    numpy_stack = []
    index_stack = []
    label_stack = []
    i=0
    # once on every contig we have in paths
    print("\nLoading contigs into Tensor objects")
    print("==========\n")
    pbar_len = len(paths[0])*len(paths)
    pbar = tqdm(total=pbar_len)
    while i < len(paths[0]): # while i less than number of contigs
        contigs_each_band = []
        index_stack.append(paths[0][i].split('_')[2][:-4])
        for band in paths: # for each frequency band
            fpath = str(lead)+"/"+str(band[i]) # set path to contig i in that band
            array = genfromtxt(fpath, delimiter=",") # read that array
            array = normalize(array)
            contigs_each_band.append(array) # add it to list of arrays for that contig
            pbar.update(1)
        contigs_each_band = np.block(contigs_each_band) # collapse list into last np axis
        # shape of contigs_each_band: (contig length, n electrodes, n freq bands)
        numpy_stack.append(contigs_each_band)


        if "ref 24-30" in str(lead):
            label_stack.append(0)
        elif "ref 31-40" in str(lead):
            label_stack.append(1)
        elif "ref 41-50" in str(lead):
            label_stack.append(2)
        elif "ref 51-60" in str(lead):
            label_stack.append(3)
        elif "ref 61-70" in str(lead):
            label_stack.append(4)
        elif "ref 71-80" in str(lead):
            label_stack.append(5)
        elif "ref 81+" in str(lead):
            label_stack.append(6)

        i+=1
    pbar.close()
    # start_indexes as array, for each contig
    index_stack = np.array(index_stack)

    # labels as array, for each contig
    label_stack = np.array(label_stack)
    # make samples 1st-read axis
    numpy_dataset = np.stack(numpy_stack, axis=0) # collapse list into np axis
    numpy_dataset = np.expand_dims(numpy_dataset,-1)
    # new shape of structure: (n contigs, contig length, n electrodes, n freq bands)

    # numpy_dataset = numpy_dataset.transpose(0, 1, 3, 2) # reshape to:
    # (n contigs, contig length, n frequency bands, n electrodes)

    # if permute labels trigger on, random sort train labels
    if permuteLabels == True:
        idy = np.random.permutation(len(train_labels))
        train_labels = train_labels[idy]

    return(numpy_dataset, index_stack, label_stack)

# randomly shuffle them, with same permutation
def shuffle_same_perm(data, start_indeces, labels):
    idx = np.random.permutation(len(data))
    data, start_indeces, labels = data[idx], start_indeces[idx], labels[idx]
    return(data, start_indeces, labels)

# input shape: (# timepoints, (3))
def createModel(train_arrays, train_image_labels, test_arrays, test_image_labels, learn, num_epochs, betaOne, betaTwo):
    # Introduce sequential Set
    model = tf.keras.models.Sequential()

    # temporal convolution
    model.add(Conv2D(16, kernel_size=(16, 3), strides=3, padding='same', activation='relu', input_shape=train_arrays[0].shape, data_format='channels_last', kernel_initializer='glorot_uniform'))

    # cross-channel convolution
    model.add(Conv2D(16, kernel_size=(16, 3), strides=3, padding='same', activation='relu', data_format='channels_last', kernel_initializer='glorot_uniform'))

    # pooling & dropout
    model.add(MaxPooling2D(pool_size=(3,3), strides=None, padding='same', data_format='channels_last'))

    # deep conv layers, cross-bandpass
    model.add(Conv2D(16, kernel_size=(16,3), strides=3, padding='same', data_format='channels_last', activation='relu', kernel_initializer='glorot_uniform'))

    model.add(Conv2D(16, kernel_size=(16,3), strides=3, padding='same', data_format='channels_last', activation='relu', kernel_initializer='glorot_uniform'))

    # model.add(Dropout(0.5))

    # model.add(Conv2D(64, kernel_size=3, strides=3, padding='same', data_format='channels_last', kernel_initializer='glorot_uniform'))

    model.add(Dropout(0.5))

    # flatten
    model.add(Flatten(data_format="channels_last"))

    # batch normalize
    # model.add(BatchNormalization())

    model.add(Dense(25, activation='relu', kernel_initializer='glorot_uniform', kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4)))
    model.add(Dense(2, activation='#logit/relu', kernel_initializer='glorot_uniform', kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4)))

    model.build(train_arrays.shape)
    model.summary()
    print("Input shape:", train_arrays.shape)

    # adaptive learning rate
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=0.001,
        decay_steps=100,
        decay_rate=0.96)

    # Model compilation
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learn, beta_1=betaOne, beta_2=betaTwo),
                 loss='sparse_categorical_crossentropy',
                 metrics=['accuracy'])
    # model.compile(optimizer = tf.keras.optimizers.SGD(learning_rate=learn, momentum=0.01, nesterov=True),
    #                  loss='binary_crossentropy',
    #                  metrics=['accuracy'])

    # tensorboard setup
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    history = model.fit(
        train_arrays,
        train_image_labels,
        epochs=num_epochs,
        # validation_split=0.33,
        validation_data=(test_arrays, test_image_labels),
        batch_size=32,
        callbacks=[tensorboard_callback]
    )

    return(history, model)

def createAgeModel(train_arrays, train_image_labels, test_arrays, test_image_labels, learn, num_epochs, betaOne, betaTwo):
    # tf.keras.backend.set_floatx('float64')
    # Introduce sequential Set
    model = tf.keras.models.Sequential()

    # time-series
    model.add(tf.keras.layers.Conv2D(25, kernel_size=(10, 1), strides=1, padding='same', activation='relu', data_format='channels_last')) #, use_bias=True, bias_initializer=tf.keras.initializers.Constant(0.6574402652980739)))
    model.add(tf.keras.layers.Conv2D(15, kernel_size=(100, 1), strides=50, padding='same', activation='relu', data_format='channels_last')) #, use_bias=True, bias_initializer=tf.keras.initializers.Constant(0.6574402652980739)))
    model.add(tf.keras.layers.Conv2D(10, kernel_size=(1250, 1), strides=250, padding='same', activation='relu', data_format='channels_last')) #, use_bias=True, bias_initializer=tf.keras.initializers.Constant(0.6574402652980739)))
    model.add(tf.keras.layers.BatchNormalization())


    # time-pooling
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(10,1), strides=None, padding='same', data_format='channels_last'))
    model.add(tf.keras.layers.Dropout(0.4))

    # cross-channel
    model.add(tf.keras.layers.Conv2D(25, kernel_size=(10, 3), strides=1, padding='same', activation='relu', data_format='channels_last')) #, use_bias=True, bias_initializer=tf.keras.initializers.Constant(0.6574402652980739)))
    model.add(tf.keras.layers.Conv2D(15, kernel_size=(100, 3), strides=1, padding='same', activation='relu', data_format='channels_last')) #, use_bias=True, bias_initializer=tf.keras.initializers.Constant(0.6574402652980739)))
    model.add(tf.keras.layers.Conv2D(10, kernel_size=(1250, 3), strides=1, padding='same', activation='relu', data_format='channels_last')) #, use_bias=True, bias_initializer=tf.keras.initializers.Constant(0.6574402652980739)))
    model.add(tf.keras.layers.BatchNormalization())

    # channel-pooling
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=None, padding='same', data_format='channels_last'))
    # model.add(tf.keras.layers.Dropout(0.2))

    # cross-bandpass
    model.add(tf.keras.layers.Conv2D(15, kernel_size=6, strides=3, padding='same', activation='relu', data_format='channels_last')) #, use_bias=True, bias_initializer=tf.keras.initializers.Constant(0.6574402652980739)))
    model.add(tf.keras.layers.Conv2D(15, kernel_size=6, strides=3, padding='same', activation='relu', data_format='channels_last')) #, use_bias=True, bias_initializer=tf.keras.initializers.Constant(0.6574402652980739)))

    # model.add(tf.keras.layers.MaxPooling2D(pool_size=(6,6), strides=None, padding='same', data_format='channels_last'))

    model.add(tf.keras.layers.Flatten(data_format="channels_last"))

    # model.add(tf.keras.layers.Dense(100, activation='softmax'))
    model.add(tf.keras.layers.Dense(7, activation='softmax'))

    model.build(train_arrays.shape)
    model.summary()
    print("Input shape:", train_arrays.shape)

    # adaptive learning rate
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-2,
    decay_steps=10000,
    decay_rate=0.2)

    # Model compilation
    # model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learn, beta_1=betaOne, beta_2=betaTwo),
    #              loss='sparse_categorical_crossentropy',
    #              metrics=['accuracy'])
    model.compile(optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule, momentum=0.01, nesterov=True),
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])

    # tensorboard setup
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    history = model.fit(
        train_arrays,
        train_image_labels,
        epochs=num_epochs,
        # validation_split=0.33,
        validation_data=(test_arrays, test_image_labels),
        batch_size=64,
        callbacks=[tensorboard_callback]
    )

    return(history, model)
