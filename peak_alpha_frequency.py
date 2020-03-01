import config
import scipy
from scipy import signal
from scipy import stats
import os
import numpy as np
import matplotlib.pyplot as plt

sourceDir = config.resultsBaseDir+"/spectral"
resultsDir = config.resultsBaseDir+"/spectral_norm"

lowestBin = int(1/(config.sampleRate / config.contigLength) * config.freqRange[0])
highestBin = int(1/(config.sampleRate / config.contigLength) * config.freqRange[1])

frequencies = np.arange(config.freqRange[0], config.freqRange[1], config.sampleRate/config.contigLength)

fnames = [fname for fname in os.listdir(sourceDir)]

for fname in fnames:
fname = fnames[0]

PSD = np.genfromtxt(sourceDir+"/"+fname, delimiter=",")[1][lowestBin:highestBin]
meanP = sum(PSD) / len(PSD)
normPSD = np.true_divide(PSD, meanP)
logNormPSD = np.log(normPSD)
slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(logNormPSD, frequencies)
print(slope)
print(intercept)
print(r_value)
print(p_value)
print(std_err)
