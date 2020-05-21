import numpy as np
import pandas as pd
import config
import os
from tqdm import tqdm

# Takes user-supplied study directory, adds all files in /raw ending with "eeg", "art", "evt" into fnames list
def getRawFnames(raw_path):
    filenames = sorted([f for f in os.listdir(raw_path) if f.rpartition('.')[2] in ['eeg','art','evt']])
    ends = []
    print("Using the following filename endings:")
    for e in filenames:
        if((".eeg" in e) & (e[4:] not in ends)):
            ends.append(e[4:])
            print("\t",e[4:])
    print("\n")
    return(filenames, ends)

# Takes list of study files, length of subject order numbers
def getSubjects(filenames, places):
    return([f[:places] for f in filenames])

def getGroupNumber(subject):
    if (int(subject) in range(100, 199)):
        return(1)
    elif (int(subject) in range(200, 299)):
        return(2)
    else:
        return(0)

def initializeDataframe(column_names, subjects):
    df = pd.DataFrame(columns=column_names)
    df['subject'] = subjects
    df.set_index('subject', inplace=True)
    for sub in subs:
        df.loc[sub]['group']=getGroupNumber(sub)
    return(df)

def populateTasks(df, filenames):
    for sub in subs:
        if (sub + "_" + "p300.eeg") in filenames:
            df.loc[sub]['p300']=[['eeg','NaN'],['art','NaN'],['evt','NaN']]
        else:
            df.loc[sub]['p300']="none"
        if (sub + "_" + "flanker.eeg") in filenames:
            df.loc[sub]['flanker']=[['eeg','NaN'],['art','NaN'],['evt','NaN']]
        else:
            df.loc[sub]['flanker']="none"
        if (sub + "_" + "chronic.eeg") in filenames:
            df.loc[sub]['chronic']=[['eeg','NaN'],['art','NaN']]
        else:
            df.loc[sub]['chronic']="none"
        if (sub + "_" + "rest.eeg") in filenames:
            df.loc[sub]['rest']=[['eeg','NaN'],['art','NaN']]
        else:
            df.loc[sub]['rest']="none"
        if (sub + "_" + "thumper.eeg") in filenames:
            df.loc[sub]['thumper']=[['eeg','NaN'],['art','NaN'],['evt','NaN']]
        else:
            df.loc[sub]['thumper']="none"
    return(df)

def loadEEGdataNumpy(df):
    print("Importing data")
    for sub in tqdm(df.index.values):
        for task in df.columns:
            if task!="group":
                if df.loc[sub][task]!="none":
                    if task=="p300":
                        df.loc[sub][task][0][1]=np.genfromtxt(raw_folder+"/"+sub+"_"+'p300.eeg', delimiter=' ')
                        df.loc[sub][task][1][1]=np.genfromtxt(raw_folder+"/"+sub+"_"+'p300.art', delimiter=' ')
                        df.loc[sub][task][2][1]=np.genfromtxt(raw_folder+"/"+sub+"_"+'p300.evt', delimiter=' ')
                    if task=="flanker":
                        df.loc[sub][task][0][1]=np.genfromtxt(raw_folder+"/"+sub+"_"+'flanker.eeg', delimiter=' ')
                        df.loc[sub][task][1][1]=np.genfromtxt(raw_folder+"/"+sub+"_"+'flanker.art', delimiter=' ')
                        df.loc[sub][task][2][1]=np.genfromtxt(raw_folder+"/"+sub+"_"+'flanker.evt', delimiter=' ')
                    if task=="chronic":
                        df.loc[sub][task][1][1]=np.genfromtxt(raw_folder+"/"+sub+"_"+'chronic.art', delimiter=' ')
                        df.loc[sub][task][0][1]=np.genfromtxt(raw_folder+"/"+sub+"_"+'chronic.eeg', delimiter=' ')
                    if task=="rest":
                        df.loc[sub][task][0][1]=np.genfromtxt(raw_folder+"/"+sub+"_"+'rest.eeg', delimiter=' ')
                        df.loc[sub][task][1][1]=np.genfromtxt(raw_folder+"/"+sub+"_"+'rest.art', delimiter=' ')
                    if task=="thumper":
                        df.loc[sub][task][0][1]=np.genfromtxt(raw_folder+"/"+sub+"_"+'thumper.eeg', delimiter=' ')
                        df.loc[sub][task][1][1]=np.genfromtxt(raw_folder+"/"+sub+"_"+'thumper.art', delimiter=' ')
                        df.loc[sub][task][2][1]=np.genfromtxt(raw_folder+"/"+sub+"_"+'thumper.evt', delimiter=' ')
    return(df)

def writeCSVs(df):
    for task in df.columns[1:]:
        os.mkdir(config.studyDirectory+"/"+task)
        print("Writing files for task:", task, "\n")
        for sub in tqdm(df.index.values):
            if (df.loc[sub][task]!="none"):
                if len(df.loc[sub][task])>=2:
                    np.savetxt(config.studyDirectory+"/"+task+'/'+sub+'_art.csv', df.loc[sub][task][1][1], delimiter=",", fmt="%2.0f")
                    np.savetxt(config.studyDirectory+"/"+task+'/'+sub+'_eeg.csv', df.loc[sub][task][0][1], delimiter=",", fmt="%2.0f")
                if len(df.loc[sub][task])==3:
                    np.savetxt(config.studyDirectory+"/"+task+'/'+sub+'_evt.csv', df.loc[sub][task][2][1], delimiter=",", fmt="%2.0f")

raw_folder = config.studyDirectory+"/raw"
fnames, ends = getRawFnames(raw_folder)
subs = list(dict.fromkeys(getSubjects(fnames, config.participantNumLen)))
print("Using the following subjects:")
print("\t",subs, "\n")
cnames = ['subject', 'group']
for task in ends:
    cnames.append(task[0:-4])
data = initializeDataframe(cnames, subs)
print(data, "\n")
data = populateTasks(data, fnames)
print(data, "\n")

EEG = loadEEGdataNumpy(data)

writeCSVs(EEG)
