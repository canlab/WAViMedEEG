#!/usr/bin/env python3
# coding: utf-8

# In[11]:


import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
mne.__version__


# In[12]:


# data = pd.read_csv("/home/clayton/science/CANlab/WAViMedEEG/CANlabStudy/p300/1/110_eeg.csv", index_col=0).transpose()
# print(data)
# data = data / 1000
# print(data)
data = np.genfromtxt('/home/clayton/science/CANlab/WAViMedEEG/CANlabStudy/p300/1/test.csv', delimiter=',').transpose()
print(data)
data = data / 1000

# In[37]:


channel_names = [
    'Fp1',
    'Fp2',
    'F3',
    'F4',
    'F7',
    'F8',
    'C3',
    'C4',
    'P3',
    'P4',
    'O1',
    'O2',
    'T3',
    'T4',
    'T5',
    'T6',
    'Fz',
    'Cz',
    'Pz'
]

sfreq = 250

info = mne.create_info(channel_names, sfreq, ch_types="eeg", montage="standard_alphabetic")

raw = mne.io.RawArray(data, info)


# In[35]:


raw.info


# In[36]:


raw.plot()


# In[28]:


print(mne.channels.get_builtin_montages())
montage = mne.channels.read_montage("standard_alphabetic")
montage.plot()


# In[29]:


raw.info["ch_names"]


# In[30]:


# raw.set_montage(montage, set_dig=True)

raw.set_eeg_reference("average")
# In[33]:


raw.plot()

# raw.plot_sensors(kind='3d', ch_groups='position')

plt.show(block=True)
#
# # In[102]:
#
#
# event_data = pd.read_csv("/home/clayton/science/CANlab/WAViMedEEG/CANlabStudy/p300/1/110_evt.csv", index_col=False, usecols=[1], skiprows=[0], names=["stim"])
# event_data
#
#
# # In[111]:
#
#
# sfreq = 250
#
# event_info = mne.create_info(['stim'], sfreq, ch_types="stim")
#
# events = mne.io.RawArray(event_data.transpose(), event_info)
#
#
# # In[112]:
#
#
# mne_events = mne.find_events(events, consecutive=True)
#
#
# # In[113]:
#
#
# import matplotlib.pyplot as plt
# plt.plot(mne_events.data[1])


# In[ ]:
