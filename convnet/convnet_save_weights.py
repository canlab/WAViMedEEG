import pathlib
from tqdm import tqdm
import config
import os
import random

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

path_to_train = pathlib.Path(config.train_source)
subject_list = convnet.get_avail_subjects(path_to_train)
# shuffle subject list randomly and then
random.shuffle(subject_list)


split = len(subject_list)//3

train_subjects = subject_list[:split*-1]
test_subjects = subject_list[split*-1:]

# print("List of available subjects:")
# for sub in subject_list:
#     print(sub)

print("Train subjects:", train_subjects)
print("Validation subjects:", test_subjects)

train_paths, _= convnet.generate_paths_and_labels(path_to_train, train_subjects, subLen=config.participantNumLen)
test_paths, _= convnet.generate_paths_and_labels(path_to_train, test_subjects, subLen=config.participantNumLen)
train_paths = convnet.reshape_paths_with_bands(train_paths, config.frequency_bands)
test_paths = convnet.reshape_paths_with_bands(test_paths, config.frequency_bands)
train_data, train_indeces, train_labels = convnet.load_numpy_stack(path_to_train, train_paths, permuteLabels=config.wumbo)
test_data, test_indeces, test_labels = convnet.load_numpy_stack(path_to_train, test_paths, permuteLabels=config.wumbo)
train_data, train_indeces, train_labels = convnet.shuffle_same_perm(train_data, train_indeces, train_labels)
test_data, test_indeces, test_labels = convnet.shuffle_same_perm(test_data, test_indeces, test_labels)

input_shape = train_data[0].shape
print("Input shape:", input_shape)
fitted, modelvar = convnet.createModel(train_data, train_labels, test_data, test_labels, rate, epochs, beta1, beta2)
modelvar.save(config.model_file)
