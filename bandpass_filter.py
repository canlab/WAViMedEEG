import scipy
from scipy import signal
import matplotlib.pyplot as plt
import os, sys
import config
import numpy as np
from tqdm import tqdm
# from scipy.signal import freqs

source = config.bandpassSource
letter = config.bandpassName
range = config.bandpassBounds

outputDir = source+"_"+letter

try:
    os.mkdir(outputDir)
except:
    print("Couldn't make output directory.")

# from scipy.signal import butter, lfilter

for fname in tqdm(os.listdir(source)):
    # fnames = os.listdir(source)
    # fname = fnames[0]
    arr = np.genfromtxt(source+"/"+fname, delimiter=",")
    sos = scipy.signal.butter(4, [range[0], range[1]], btype='band', fs=config.sampleRate, output='sos')
    np.savetxt(outputDir+"/"+fname, sos, delimiter=",", fmt="%2.0f")
