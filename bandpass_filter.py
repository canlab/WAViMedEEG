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

source = config.bandpassSource
letter = config.bandpassName
range = config.bandpassBounds
fnames = os.listdir(source)

outputDir = source+"_"+letter

try:
    os.mkdir(outputDir)
except:
    print("Couldn't make output directory.")

# x = np.arange(0, len(fnames), 1)
# random_plot_selection = random.sample(x, config.numberExamples)
random_plot_selection = [0, 10, 20, 30, 40, 50]

i = 0
t = np.linspace(0, (config.contigLength / config.sampleRate), config.sampleRate, False)
for fname in tqdm(fnames):
    arr = np.genfromtxt(source+"/"+fname, delimiter=",")
    j = 0
    post = np.ndarray(shape=(config.contigLength, len(config.network_channels))).T
    for sig in arr.T:
        sos = scipy.signal.butter(4, [range[0], range[1]], btype='band', fs=config.sampleRate, output='sos')
        filtered = scipy.signal.sosfilt(sos, sig)
        if (i in random_plot_selection) & (j == 0):
            fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
            ax1.plot(t, sig)
            ax1.set_title('Channel 1 Signal')
            ax1.axis([0, 1, np.min(sig)-5, np.max(sig)+5])
            ax2.plot(t, filtered)
            ax2.set_title('After '+ config.bandpassName + ' filter')
            ax2.axis([0, 1, np.min(sig)-5, np.max(sig)+5])
            ax2.set_xlabel('Time [seconds]')
            plt.tight_layout()
            plt.show()
        post[j] = filtered
        j+=1
    np.savetxt(outputDir+"/"+fname, post.T, delimiter=",", fmt="%2.0f")
    i+=1
