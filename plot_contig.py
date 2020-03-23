import config
# from plotly import tools
# from plotly.graph_objs import Layout, layout, Scatter, Annotation, Annotations, Data, Figure, Marker, Font
import numpy as np
import matplotlib.pyplot as plt

array = np.genfromtxt(config.plotSource, delimiter=",")
n_channels = len(config.network_channels)
ch_names = config.network_channels
#
# step = 1. / n_channels
# kwargs = dict(domain=[1 - step, 1], showticklabels=False, zeroline=False, showgrid=False)
# times = [i for i in range(250)]
#
# # create objects for layout and traces
# layout = Layout(yaxis=layout.YAxis(kwargs), showlegend=False)
# traces = [Scatter(x=times, y=array.T[:, 0])]
#
# # loop over the channels
# for ii in range(1, n_channels):
#         kwargs.update(domain=[1 - (ii + 1) * step, 1 - ii * step])
#         layout.update({'yaxis%d' % (ii + 1): layout.YAxis(kwargs), 'showlegend': False})
#         traces.append(Scatter(x=times, y=data.T[:, ii], yaxis='y%d' % (ii + 1)))
#
# # add channel names using Annotations
# annotations = Annotations([Annotation(x=-0.06, y=0, xref='paper', yref='y%d' % (ii + 1),
#                                       text=ch_name, font=Font(size=9), showarrow=False)
#                           for ii, ch_name in enumerate(ch_names)])
# layout.update(annotations=annotations)
#
# # set the size of the figure and plot it
# layout.update(autosize=False, width=1000, height=600)
# fig = Figure(data=array, layout=layout)
# py.iplot(fig, filename='shared xaxis')

#%matplotlib qt5
import os
import mne
import matplotlib.pyplot as plt
import numpy as np
from numpy import array
import pandas as pd
import pathlib
import config

#data_folder = config.studyDirectory+"/"+config.mneTask
#data_folder = [os.path.join(data_folder, group) for group in os.listdir(data_folder)]

hasStim = False
# if config.selectedTask in ['p300', 'flanker', 'chronic']:
#     hasStim = True

# eeg_data = np.genfromtxt(data_folder[0]+"/100_eeg.csv", delimiter=',').transpose()
eeg_data = np.genfromtxt(config.plotSource, delimiter=',').transpose()

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
