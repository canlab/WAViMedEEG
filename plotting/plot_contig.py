import config
# from plotly import tools
# from plotly.graph_objs import Layout, layout, Scatter, Annotation, Annotations, Data, Figure, Marker, Font
import numpy as np
import matplotlib.pyplot as plt

array = np.genfromtxt(config.studyDirectory+"/contigs/"+config.selectedTask+"_"+config.contigLength+"/"+config.plot_subject+"_nofilter_"+config.plot_contig, delimiter=",")
n_channels = len(config.network_channels)
ch_names = config.network_channels

import os
import mne
import matplotlib.pyplot as plt
import numpy as np
from numpy import array
import pandas as pd
import pathlib
import config

hasStim = False
# if config.selectedTask in ['p300', 'flanker', 'chronic']:
#     hasStim = True

# eeg_data = np.genfromtxt(data_folder[0]+"/100_eeg.csv", delimiter=',').transpose()
eeg_data = np.genfromtxt(config.plotSource, delimiter=',').T

#print(eeg_data)
eeg_data = eeg_data

# if hasStim:
#     event_data = np.genfromtxt(data_folder[0]+"/100_evt.csv", delimiter=',').transpose()
#     #print(event_data)
#
#     trial_data = np.concatenate((eeg_data, event_data[None, :]))
#     #print(trial_data)
trial_data = eeg_data / 1000

channels = config.network_channels

channel_types = ['eeg' for channel in channels]

if hasStim:
    channels.append(config.mneTask)
    channel_types.append('stim')

sfreq = config.sampleRate
info = mne.create_info(channels, sfreq, ch_types=channel_types)
raw = mne.io.RawArray(trial_data, info)

print(raw.info)

print(mne.channels.get_builtin_montages())
montage = mne.channels.read_montage("standard_alphabetic")
print(montage.plot())

raw.set_montage(montage, set_dig=True)
raw.set_eeg_reference("average")

scalings = {'eeg': 0.01}
raw.plot(scalings=scalings, show=True, block=True)
