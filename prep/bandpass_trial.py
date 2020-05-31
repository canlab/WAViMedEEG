import scipy
import matplotlib.pyplot as plt
import os, sys
import config
import numpy as np
from tqdm import tqdm
import random
from random import randint
import scipy.signal
from scipy.signal import butter, lfilter
# from scipy.signal import freqs

bands = config.frequency_bands

taskfolders = os.listdir(config.studyDirectory)
taskfolders = [folder for folder in taskfolders if folder in ["chronic", "p300", "flanker", "rest"]]

for band in bands:
    print("Filter range:", band[0])
    range = band[1]
    for task in taskfolders:
        print("Filtering task:", task)
        fnames = [fname for fname in os.listdir(config.studyDirectory+"/"+task) if "eeg_nofilter" in fname]
        for fname in tqdm(fnames):
            arr = np.genfromtxt(config.studyDirectory+"/"+task+"/"+fname, delimiter=",")
            j = 0
            post = np.ndarray(shape=(len(arr), len(config.channel_names))).T
            for sig in arr.T:
                sos = scipy.signal.butter(4, [range[0], range[1]], btype='band', fs=config.sampleRate, output='sos')
                filtered = scipy.signal.sosfilt(sos, sig)
                post[j] = filtered
                j+=1
            np.savetxt(config.studyDirectory+"/"+task+"/"+fname[:7]+"_"+band[0]+".csv", post.T, delimiter=",", fmt="%2.1f")
