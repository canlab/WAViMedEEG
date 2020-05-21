
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
    for file in os.listdir(test_path):
        if file[:3] not in subs:
            if file[0] != "0":
                subs.append(file[:3])
    return(subs)

def generate_paths_and_labels(omission=None):
    image_paths = os.listdir(test_path)
    image_paths = [path for path in image_paths if path[0] != "0"]
    #train_image_paths = list(train_path.glob('*/*'))
    test_image_paths = [str(path) for path in image_paths if omission == path[:3]]

    random.shuffle(test_image_paths)

    test_count = len(test_image_paths)
    print("You have", test_count, "files.")

    #list the available labels
    label_names = ['pain', 'ctrl']
    #label_names = sorted(item.name for item in train_path.glob('*/') if item.is_dir())
    print("Labels discovered:", label_names)

    #assign an index to each label
    label_to_index = dict((name, index) for index, name in enumerate(label_names))
    print("Label indices:", label_to_index)

    #create a list of every file and its integer label
    test_image_groups = [config.subjectKeys.get(int(path[0]), "none") for path in test_image_paths]

    test_image_labels = [label_to_index[group] for group in test_image_groups]
    #train_image_labels = [int(path[0]) for path in train_image_paths]

    # null tests, shuffle labels and randomize test
    if config.permuteLabels == True:
        random.shuffle(train_image_labels)
        rand_group = random.randint(0, 1)
        test_image_labels = [rand_group for label in test_image_labels]

    # force list of labels to numpy array
    test_image_labels = np.array(test_image_labels)

    return(test_image_paths, test_image_labels)

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
    numpy_dataset = np.swapaxes(numpy_dataset, 2, 3)

    # randomly shuffle them, with same permutation
    idx = np.random.permutation(len(numpy_dataset))
    numpy_dataset, label_stack = numpy_dataset[idx], label_stack[idx]

    return(numpy_dataset, label_stack)


modelvar = tf.keras.models.load_model('pain_model/MyModel')

# Check its architecture
modelvar.summary()

import matplotlib.pyplot as plt

test_path = pathlib.Path(config.evalPath)
subject_list = get_avail_subjects()
print("List of available subjects:")
for sub in subject_list:
    print(sub)

if not os.path.isdir(config.resultsBaseDir):
    os.mkdir(config.resultsBaseDir)

if os.path.isdir(config.resultsPath):
    print("Specified results path already exists. Go back and clean up.")
else:
    os.mkdir(config.resultsPath)

for sub in tqdm(subject_list):
    test_paths, test_labels = generate_paths_and_labels(sub)

    test_paths_and_labels = reshape_paths(test_paths, test_labels)

    test_data, test_labels = load_numpy_stack(test_path, test_paths_and_labels)

    f = open(config.resultsPath+"/"+sub+".txt", 'w')

    score = modelvar.evaluate(test_data, test_labels)
    f.write("Group: " + repr(test_labels[0])+"\n")
    f.write("Loss: " + repr(score[0]) + "\n" + "Accuracy: " + repr(score[1]) + "\n")
    f.close()
