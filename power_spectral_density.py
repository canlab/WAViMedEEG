import config
import scipy
from scipy import signal
from scipy import stats
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

sourceDir = config.studyDirectory+"/contigs_"+config.selectedTask
resultsDir = config.resultsBaseDir+"/spectral"

try:
    os.mkdir(config.resultsBaseDir)
except:
    print("Base results directory couldn't be made")
try:
    os.mkdir(resultsDir)
except:
    print("Spectral results directory couldn't be made")

subject_list = [fname[:3] for fname in os.listdir(sourceDir) if fname[:0]!="0"]
subject_list = set(subject_list)

print("Analyzing spectral density of subjects:", subject_list)

for sub in tqdm(subject_list):
    subject_files = [fname for fname in os.listdir(sourceDir) if fname[:3]==sub]
    for contig in subject_files:
        array = np.genfromtxt(sourceDir+"/"+contig, delimiter=',')
        channel_number = 0
        for sensor_waveform in array.T:
            electrode = config.network_channels[channel_number]
            # perform pwelch routine to extract PSD estimates by channel
            f, Pxx_den = scipy.signal.periodogram(
                sensor_waveform,
                fs=float(config.sampleRate),
                window='hann'
                )
            spectral = np.array((f, Pxx_den))
            np.savetxt(resultsDir+"/"+sub+"_"+contig[4:-4]+"_"+electrode+".csv", spectral, delimiter=",")
            channel_number+=1
