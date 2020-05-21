
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

import convnet

print("Using Tensorflow version", tf.__version__)

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

modelvar = tf.keras.models.load_model('pain_model/MyModel')

# Check its architecture
modelvar.summary()

import matplotlib.pyplot as plt

test_path = pathlib.Path(config.evalPath)
subject_list = convnet.get_avail_subjects()
print("List of available subjects:")
for sub in subject_list:
    print(sub)

if not os.path.isdir(config.resultsBaseDir):
    os.mkdir(config.resultsBaseDir)

if os.path.isdir(config.resultsPath):
    print("Specified results path already exists. Go back and clean up.")
    quit()
else:
    os.mkdir(config.resultsPath)

for sub in tqdm(subject_list):
    , , test_paths, test_labels = convnet.generate_paths_and_labels(sub)

    test_paths_and_labels = reshape_paths(test_paths, test_labels)

    test_data, test_labels = load_numpy_stack(test_path, test_paths_and_labels)

    i = 0
    for contig in zip(test_data, test_labels):
        f = open(config.resultsPath+"/"+sub+"-"+str(i)+".txt", 'w')
        array = np.expand_dims(contig[0], axis=0)
        array = np.asarray(contig[0][np.newaxis,:])
        labels = np.asarray(contig[1][np.newaxis])
        score = modelvar.predict(array)
        print(score)
        f.write("Group: " + repr(contig[1])+"\n")
        f.write("Pain Prediction: " + repr(score[0][1]) + "\n")
        f.close()
        i+=1
