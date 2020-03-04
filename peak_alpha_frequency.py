import config
import scipy
from scipy import signal
from scipy import stats
import os
import numpy as np
import matplotlib.pyplot as plt

sourceDir = config.resultsBaseDir+"/spectral"
resultsDir = config.resultsBaseDir+"/spectral_norm"

lowestBin = int(1/(config.sampleRate / config.contigLength) * config.alphaRange[0])
highestBin = int(1/(config.sampleRate / config.contigLength) * config.alphaRange[1])

frequencies = np.arange(config.alphaRange[0], config.alphaRange[1], config.sampleRate/config.contigLength)
freq_res = config.sampleRate /

fnames = [fname for fname in os.listdir(sourceDir)]
#
# class pChans:
#     def __init__(self):
#         self.pxx = [],
#         self.minPow = [],
#         self.d0 = [],
#         self.d1 = [],
#         self.d2 = [],
#         self.peaks = [],
#         self.pos1 = [],
#         self.pos2 = [],
#         self.f1 = [],
#         self.f2 = [],
#         self.inf1 = [],
#         self.inf2 = [],
#         self.Q = [],
#         self.Qf = [],
#         self.gravs = [],
#         self.selP = [],
#         self.selG = []
#
# class pSpec:
#     def __init__(self):
#         self.sub = 0
#         self.contig = 0
#         self.channel = 'e'
#         self.sums = {
#             paf = 0
#             pafStd = 0
#             muSpec = np.empty(len(frequencies))
#         }

def next_power_of_2(x):
    return 1 if x == 0 else 2**(x - 1).bit_length()

# calculate frequency resolution
exp_tlen = next_power_of_2(config.contigLength)
frequency_res = config.sampleRate / 2**(exp_tlen)

def peakBounds(d0, d1, d2, f, w, minp, mdiff, fres):
    # set upper and lower bounds for alpha bands
    lower_alpha =

for fname in fnames:
    # delimit range of freq bins to be included in analysis
    PSD = np.genfromtxt(sourceDir+"/"+fname, delimiter=",")[1][lowestBin:highestBin]

    # normalise truncated PSD t
    normPSD = np.true_divide(PSD, np.mean(PSD))

    # slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(frequencies, logNormPSD)
    # calculate minPower vector
    p, cov = np.polyfit(frequencies, np.log(normPSD), cov=True) # fit regression line to normalized log-scaled spectra
    yvalues = np.polyval(p, frequencies) # derive yval coefficients of fitted polynomial
    mu1 = np.mean(values) # population mean
    mu2 = np.sqrt(np.diag(cov)) # standard deviation estimates for each coefficient
    minPow = yvalues + (1 * mu2)

    # apply Savitzky-Golay filter to fit curves to spectra & estimate 1st and 2nd derivatives
    d0 = scipy.signal.savgol_filter(PSD, config.window_length, config.poly_order, 0)
    d1 = scipy.signal.savgol_filter(PSD, config.window_length, config.poly_order, 1)
    d2 = scipy.signal.savgol_filter(PSD, config.window_length, config.poly_order, 2)

    # take derivatives, find peak(s) and boundaries of alpha band
    peaks =
    pos1 =
    pos2 =
    f1 =
    f2 =
    inf1 =
    inf2 =
    Q =
    Qf =
