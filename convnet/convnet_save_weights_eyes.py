import pathlib
from tqdm import tqdm
import config
import os
import random
import numpy as np

import convnet

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

folders = [config.myStudies+"/"+folder+"/contigs/"+config.selectedTask+"_"+str(config.contigLength) for folder in os.listdir(config.myStudies) if "eyes" in folder]
paths_to_train = []
for folder in folders:
    paths_to_train.append(pathlib.Path(folder))

train_paths = []
test_paths = []

train_data = []
test_data = []

train_indeces = []
test_indeces = []

train_labels = []
test_labels = []

for path in paths_to_train:
    subject_list = convnet.get_avail_subjects(path)
    random.shuffle(subject_list)
    subject_list = subject_list[:200]
    split = len(subject_list)//3
    train_subjects = subject_list[:split*-1]
    print("Number of train subjects:", len(train_subjects))
    test_subjects = subject_list[split*-1:]
    print("Number of test subjects:", len(test_subjects))

    # train
    paths, _ = convnet.generate_paths_and_labels(path, train_subjects, subLen=config.participantNumLen)
    paths = convnet.reshape_paths_with_bands(paths, config.frequency_bands)
    train_paths.extend(paths)
    data, indeces, labels = convnet.load_numpy_stack(path, paths, permuteLabels=config.wumbo)
    data, indeces, labels = convnet.shuffle_same_perm(data, indeces, labels)
    train_data.extend(data)
    train_indeces.extend(indeces)
    train_labels.extend(labels)
    # validation
    paths, _ = convnet.generate_paths_and_labels(path, test_subjects, subLen=config.participantNumLen)
    paths = convnet.reshape_paths_with_bands(paths, config.frequency_bands)
    test_paths.extend(paths)
    data, indeces, labels = convnet.load_numpy_stack(path, paths, permuteLabels=config.wumbo)
    data, indeces, labels = convnet.shuffle_same_perm(data, indeces, labels)
    test_data.extend(data)
    test_indeces.extend(indeces)
    test_labels.extend(labels)

train_paths = np.array(train_paths)
test_paths  = np.array(test_paths)
train_data = np.array(train_data)
test_data = np.array(test_data)
train_indeces = np.array(train_indeces)
test_indeces = np.array(test_indeces)
train_labels = np.array(train_labels)
test_labels = np.array(test_labels)

fitted, modelvar = convnet.createModel(train_data, train_labels, test_data, test_labels, rate, epochs, beta1, beta2)
modelvar.save(config.model_file)
