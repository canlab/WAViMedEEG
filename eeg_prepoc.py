#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!/usr/bin/env python3

import os
if(os.getcwd()) != '/home/clayton/science/CANlab/WAViMedEEG/CANlabStudy':
    os.chdir('/home/clayton/science/CANlab/WAViMedEEG/CANlabStudy')


# In[2]:


import numpy as np
import pandas as pd
import tensorflow as tf

os.chdir('raw')
fnames = os.listdir('/home/clayton/science/CANlab/WAViMedEEG/CANlabStudy/raw')

column_names = ['subject', 'group', 'p300', 'flanker', 'chronic', 'rest']

def generateSubjects(fnames):
    return([fname[:3] for fname in fnames])

subs = list(dict.fromkeys(generateSubjects(fnames)))

data = pd.DataFrame(columns=column_names)
data['subject'] = subs
data.set_index('subject', inplace=True)
for sub in subs:
    if (int(sub) in range(100, 199)):
        data.loc[sub]['group']=1
    elif (int(sub) in range(200, 299)):
        data.loc[sub]['group']=2
    else:
        data.loc[sub]['group']=0

ends=[
    '_P300_Eyes_Closed.eeg',
    '_Flanker_Test.eeg',
    '_EO_Baseline_12.eeg',
    '_EO_Baseline_8.eeg']

print(data)


# In[3]:


def populateTasks(df, fnames):
    for sub in subs:
        if (sub + ends[0]) in fnames:
            df.loc[sub]['p300']=[['eeg','NaN'],['art','NaN'],['evt','NaN']]
        else:
            df.loc[sub]['p300']="none"
        if (sub + ends[1]) in fnames:
            df.loc[sub]['flanker']=[['eeg','NaN'],['art','NaN'],['evt','NaN']]
        else:
            df.loc[sub]['flanker']="none"
        if (sub + ends[2]) in fnames:
            df.loc[sub]['chronic']=[['eeg','NaN'],['art','NaN']]
        else:
            df.loc[sub]['chronic']="none"
        if (sub + ends[3]) in fnames:
            df.loc[sub]['rest']=[['eeg','NaN'],['art','NaN']]
        else:
            df.loc[sub]['rest']="none"
    return(df)

multidata = populateTasks(data, fnames)
print(multidata)


# In[5]:


def loadEEGdata(df):
    channel_names = [
        'FP1',
        'FP2',
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
        'FZ',
        'CZ',
        'PZ'
    ]
    for sub in df.index.values:
        for task in df.columns:
            if df.loc[sub][task]!="none":
                if task=="p300":
                    df.loc[sub][task][0][1]=pd.read_csv(sub+ends[0][:-4]+'.eeg', delimiter=' ', names=channel_names, index_col=False)
                    df.loc[sub][task][1][1]=pd.read_csv(sub+ends[0][:-4]+'.art', delimiter=' ', names=channel_names, index_col=False)
                    df.loc[sub][task][2][1]=pd.read_csv(sub+ends[0][:-4]+'.evt', delimiter=' ', names=channel_names, index_col=False)
                if task=="flanker":
                    df.loc[sub][task][0][1]=pd.read_csv(sub+ends[1][:-4]+'.eeg', delimiter=' ', names=channel_names, index_col=False)
                    df.loc[sub][task][1][1]=pd.read_csv(sub+ends[1][:-4]+'.art', delimiter=' ', names=channel_names, index_col=False)
                    df.loc[sub][task][2][1]=pd.read_csv(sub+ends[1][:-4]+'.evt', delimiter=' ', names=channel_names, index_col=False)
                if task=="chronic":
                    df.loc[sub][task][0][1]=pd.read_csv(sub+ends[2][:-4]+'.eeg', delimiter=' ', names=channel_names, index_col=False)
                    df.loc[sub][task][1][1]=pd.read_csv(sub+ends[2][:-4]+'.art', delimiter=' ', names=channel_names, index_col=False)
                if task=="rest":
                    df.loc[sub][task][0][1]=pd.read_csv(sub+ends[3][:-4]+'.eeg', delimiter=' ', names=channel_names, index_col=False)
                    df.loc[sub][task][1][1]=pd.read_csv(sub+ends[3][:-4]+'.art', delimiter=' ', names=channel_names, index_col=False)
    return(df)

EEG = loadEEGdata(multidata)
print(EEG.loc['110']['p300'][0][1])


# In[7]:


# EEG.loc['110']['p300'][0][1].transpose()


# In[32]:


# os.chdir('/home/clayton/science/CANlab/WAViMedEEG/CANlabStudy')
# for task in EEG.columns[1:]:
#     os.mkdir(task)
#     os.chdir(task)
#     os.mkdir('0')
#     os.mkdir('1')
#     os.mkdir('2')
#     os.chdir('..')
#     for sub in EEG.index.values:
#         if (EEG.loc[sub][task]!="none"):
#             if len(EEG.loc[sub][task])>=2:
#                 EEG.loc[sub][task][0][1].to_csv(task+'/'+str(EEG.loc[sub]['group'])+'/'+sub+'_eeg.csv')
#                 EEG.loc[sub][task][1][1].to_csv(task+'/'+str(EEG.loc[sub]['group'])+'/'+sub+'_art.csv')
#             if len(EEG.loc[sub][task])==3:
#                 EEG.loc[sub][task][2][1].to_csv(task+'/'+str(EEG.loc[sub]['group'])+'/'+sub+'_evt.csv')


# In[ ]:


EEG.to_pickle("/home/clayton/science/CANlab/WAViMedEEG/EEG.pkl")


# In[ ]:
