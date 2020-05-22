import pathlib
from tqdm import tqdm
import config
import os
import numpy as np

import convnet

modelvar = tf.keras.models.load_model(config.model_file)

# Check its architecture
modelvar.summary()

path_to_test = pathlib.Path(config.eval_source)
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
    , test_paths = convnet.generate_paths_and_labels(sub)
    test_paths = convnet.reshape_paths_with_bands(test_paths, config.frequency_bands)
    test_data, test_labels = convnet.load_numpy_stack(path_to_test, test_paths, permuteLabels=config.permuteLabels)
    test_data, test_labels = convnet.shuffle_same_perm(test_data, test_labels)

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
