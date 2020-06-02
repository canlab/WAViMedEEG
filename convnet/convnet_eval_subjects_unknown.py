import pathlib
from tqdm import tqdm
import config
import os
import tensorflow as tf

import convnet

modelvar = tf.keras.models.load_model(config.model_file)

# Check its architecture
modelvar.summary()

path_to_test = pathlib.Path(config.eval_source)
subject_list = convnet.get_pilot_subjects(path_to_test)
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
    _, test_paths = convnet.generate_paths_and_labels(path_to_test, omission=sub)
    test_paths = convnet.reshape_paths_with_bands(test_paths, config.frequency_bands)
    test_data, test_indeces, test_labels = convnet.load_numpy_stack(path_to_test, test_paths, permuteLabels=config.wumbo)
    test_data, test_indeces, test_labels = convnet.shuffle_same_perm(test_data, test_indeces, test_labels)

    f = open(config.resultsPath+"/"+sub+".txt", 'w')

    score = modelvar.predict(test_data)

    f.write("Loss: " + repr(score[0]) + "\n" + "Accuracy: " + repr(score[1]) + "\n")
    f.close()
