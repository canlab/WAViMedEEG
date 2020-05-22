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
    os.mkdir(config.resultsBaseDir)
except:
    print("Results base dir was already created.")

try:
    os.mkdir(config.resultsPath)
except:
    print("Specified results directory already exists. Go back and clean up.")
    quit()

path_to_train = pathlib.Path(config.train_source)

subject_list = convnet.get_avail_subjects(path_to_train)
print("List of available subjects:")
for sub in subject_list:
    print(sub)

for sub in tqdm(subject_list):
    train_paths, test_paths = convnet.generate_paths_and_labels(path_to_train, omission=sub)
    train_paths = convnet.reshape_paths_with_bands(train_paths, config.frequency_bands)
    test_paths = convnet.reshape_paths_with_bands(test_paths, config.frequency_bands)
    train_data, train_labels = convnet.load_numpy_stack(path_to_train, train_paths, permuteLabels=config.wumbo)
    test_data, test_labels = convnet.load_numpy_stack(path_to_train, test_paths, permuteLabels=config.wumbo)
    train_data, train_labels = convnet.shuffle_same_perm(train_data, train_labels)
    test_data, test_labels = convnet.shuffle_same_perm(test_data, test_labels)

    fitted, modelvar = convnet.createModel(train_data, train_labels, rate, epochs, beta1, beta2)

    f = open(config.resultsPath+"/"+sub+".txt", 'w')

    score = modelvar.evaluate(test_data, test_labels)
    f.write("Group: " + repr(test_labels[0])+"\n")
    f.write("Loss: " + repr(score[0]) + "\n" + "Accuracy: " + repr(score[1]) + "\n")
    f.close()
