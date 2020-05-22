import pathlib
from tqdm import tqdm
import config
import os

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

try:
    os.mkdir(config.resultsPath)
except:
    print("Results dir already made.")

path_to_train = pathlib.Path(config.source)
subject_list = convnet.get_avail_subjects()
print("List of available subjects:")
for sub in subject_list:
    print(sub)

train_paths, = convnet.generate_paths_and_labels(path_to_train)
train_paths = convnet.reshape_paths_with_bands(train_paths, config.frequency_bands)
train_data, train_labels = convnet.load_numpy_stack(path_to_train, train_paths, permuteLabels=config.permuteLabels)
train_data, train_labels = convnet.shuffle_same_perm(train_data, train_labels)

fitted, modelvar = convnet.createModel(train_data, train_labels, rate, epochs, beta1, beta2)
modelvar.save(config.model_file)
)
