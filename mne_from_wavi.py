#!/usr/bin/env python3
# coding: utf-8

# In[1]:


# get_ipython().run_line_magic('matplotlib', 'qt5')
import mne
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
mne.__version__

plt.show(block=True)
# In[2]:


eeg_data = np.genfromtxt('/home/clayton/science/CANlab/WAViMedEEG/CANlabStudy/p300/1/test.csv', delimiter=',').transpose()
print(eeg_data)
eeg_data = eeg_data / 1000


# In[3]:


event_data = np.genfromtxt('/home/clayton/science/CANlab/WAViMedEEG/CANlabStudy/p300/1/test_events.csv', delimiter=',').transpose()
print(event_data)


# In[4]:


p300_data = np.concatenate((eeg_data, event_data[None, :]))
print(p300_data)


# In[5]:


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
    'Pz',
    'p300'
]

channel_types = [
    'eeg',
    'eeg',
    'eeg',
    'eeg',
    'eeg',
    'eeg',
    'eeg',
    'eeg',
    'eeg',
    'eeg',
    'eeg',
    'eeg',
    'eeg',
    'eeg',
    'eeg',
    'eeg',
    'eeg',
    'eeg',
    'eeg',
    'stim'
]


# In[6]:


sfreq = 250

info = mne.create_info(channel_names, sfreq, ch_types=channel_types)

raw = mne.io.RawArray(p300_data, info)


# In[7]:


raw.info


# In[8]:


print(mne.channels.get_builtin_montages())
montage = mne.channels.read_montage("standard_alphabetic")
montage.plot()


# In[9]:


raw.set_montage(montage, set_dig=True)

raw.set_eeg_reference("average")
# In[10]:


scalings = {'eeg': 0.1}
raw.plot(scalings=scalings)


# In[11]:


mne_events = mne.find_events(raw, consecutive=True)


# In[15]:


import plotly.offline as py
from plotly import tools
from plotly.graph_objs import Layout, layout, Scatter, Annotation, Annotations, Data, Figure, Marker, Font


# In[16]:


picks = mne.pick_types(raw.info, eeg=True, exclude=[])
start, stop = raw.time_as_index([0, 10])

n_channels = 19
data, times = raw[picks[:n_channels], start:stop]
ch_names = [raw.info['ch_names'][p] for p in picks[:n_channels]]


# In[17]:


step = 1. / n_channels
kwargs = dict(domain=[1 - step, 1], showticklabels=False, zeroline=False, showgrid=False)

# create objects for layout and traces
layout = Layout(yaxis=layout.YAxis(kwargs), showlegend=False)
traces = [Scatter(x=times, y=data.T[:, 0])]

# loop over the channels
for ii in range(1, n_channels):
        kwargs.update(domain=[1 - (ii + 1) * step, 1 - ii * step])
        layout.update({'yaxis%d' % (ii + 1): layout.YAxis(kwargs), 'showlegend': False})
        traces.append(Scatter(x=times, y=data.T[:, ii], yaxis='y%d' % (ii + 1)))

# add channel names using Annotations
annotations = Annotations([Annotation(x=-0.06, y=0, xref='paper', yref='y%d' % (ii + 1),
                                      text=ch_name, font=Font(size=9), showarrow=False)
                          for ii, ch_name in enumerate(ch_names)])
layout.update(annotations=annotations)

# set the size of the figure and plot it
layout.update(autosize=False, width=1000, height=600)
fig = Figure(data=Data(traces), layout=layout)
py.iplot(fig, filename='shared xaxis')


# In[20]:


picks = mne.pick_types(raw.info, eeg=True, meg=False, stim=True, exclude=[])
raw.save('sample_vis_raw.fif', tmin=0., tmax=150., picks=picks, overwrite=True)


# In[23]:


raw_beta = mne.io.Raw("sample_vis_raw.fif", preload=True)  # reload data with preload for filtering

# keep beta band
raw_beta.filter(13.0, 30.0, method='iir', n_jobs=-1)

# save the result
raw_beta.save('sample_vis_raw.fif', overwrite=True)

# check if the info dictionary got updated
print(raw_beta.info['highpass'], raw_beta.info['lowpass'])


# In[24]:


events = mne.find_events(raw, stim_channel='p300')
print(events[:5])  # events is a 2d array


# In[30]:


print(len(events[events[:, 2] == 1]))
print(len(events[events[:, 2] == 2]))


# In[31]:


len(events)


# In[32]:


raw.ch_names.index('p300')


# In[33]:


raw = mne.io.Raw("sample_vis_raw.fif", preload=True)  # reload data with preload for filtering
raw.filter(1, 40, method='iir')


# In[35]:


d, t = raw[raw.ch_names.index('p300'), :]
plt.plot(d[0,:10000])
py.iplot_mpl(plt.gcf())


# In[37]:


event_ids = ['1', '2']
fig = mne.viz.plot_events(events, raw.info['sfreq'], raw.first_samp, show=False)

# convert plot to plotly
update = dict(layout=dict(showlegend=True), data=[dict(name=e) for e in event_ids])
py.iplot_mpl(plt.gcf())


# In[39]:


event_id = dict(reg=1, odd=2)  # event trigger and conditions
tmin = -0.2  # start of each epoch (200ms before the trigger)
tmax = 0.5  # end of each epoch (500ms after the trigger)


# In[40]:


event_id


# In[41]:


picks = mne.pick_types(raw.info, eeg=True, stim=False, exclude='bads')


# In[42]:


baseline = (None, 0)  # means from the first instant to t = 0


# In[43]:


# reject = dict(grad=4000e-13, mag=4e-12, eog=150e-6)


# In[45]:


# epochs = mne.Epochs(raw, events, event_id, tmin, tmax, proj=True, picks=picks, baseline=baseline, reject=reject)
epochs = mne.Epochs(raw, events, event_id, tmin, tmax, proj=True, picks=picks, baseline=baseline)


# In[47]:


# from mne.fixes import Counter

# # drop bad epochs
# epochs.drop_bad_epochs()
# drop_log = epochs.drop_log

# # calculate percentage of epochs dropped for each channel
# perc = 100 * np.mean([len(d) > 0 for d in drop_log if not any(r in ['IGNORED'] for r in d)])
# scores = Counter([ch for d in drop_log for ch in d if ch not in ['IGNORED']])
# ch_names = np.array(list(scores.keys()))
# counts = 100 * np.array(list(scores.values()), dtype=float) / len(drop_log)
# order = np.flipud(np.argsort(counts))


# In[48]:


# from plotly.graph_objs import Data, Layout, Bar, YAxis, Figure

# data = Data([
#     Bar(
#         x=ch_names[order],
#         y=counts[order]
#     )
# ])
# layout = Layout(title='Drop log statistics', yaxis=YAxis(title='% of epochs rejected'))

# fig = Figure(data=data, layout=layout)
# py.iplot(fig)


# In[49]:


# epochs.save('sample-epo.fif')


# In[50]:


evoked = epochs.average()


# In[52]:


fig = evoked.plot(show=False)  # butterfly plots
update = dict(layout=dict(showlegend=False), data=[dict(name=raw.info['ch_names'][p]) for p in picks[:10]])
py.iplot_mpl(fig)


# In[57]:


# topography plots
evoked.plot_topomap(times=np.linspace(0.05, 0.15, 5), ch_type='eeg');


# In[58]:


epochs_data = epochs['odd'].get_data()
print(epochs_data.shape)


# In[62]:


evokeds = [epochs[k].average() for k in event_id]
from mne.viz import plot_topo
layout = mne.find_layout(epochs.info)
plot_topo(evokeds, layout=layout, color=['blue', 'orange']);


# In[ ]:
