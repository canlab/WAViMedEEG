import os
import numpy as np
from numpy import array, ma, genfromtxt
import pandas as pd
import config
from tqdm import tqdm

try:
    contigsFolder = config.studyDirectory+"/contigs"+"_"+config.selectedTask+"_"+str(config.contigLength)
    os.mkdir(contigsFolder)
except:
    print("You already had the specified contigs folder. Go back and clean up.\n")
    quit()

def load_csv(prefix, sub):
    eeg = genfromtxt(prefix+"/"+sub+"_eeg.csv", delimiter=",")
    art = genfromtxt(prefix+"/"+sub+"_art.csv", delimiter=",")
    return(eeg, art)

def apply_art_mask_nan(data, artifact):
    mx = ma.masked_array(data, mask=artifact)
    mx = ma.filled(mx.astype(float), np.nan)
    return(mx)

def apply_art_mask_zero(data, artifact):
    mx = ma.masked_array(data, mask=artifact)
    mx = ma.filled(mx.astype(float), 0)
    return(mx)

def filter_my_channels(dataset, keep_channels, axisNum):
    filter_indeces = []
    # print("Old Shape of Dataset:", dataset.shape, "\n")
    # for each channel in my channel list
    for keep in keep_channels:
        filter_indeces.append(config.channel_names.index(keep))
    #   get the index of that channel in the master channel list
    #   add that index number to a new list filter_indeces

    # iter over the rows of axis 2 (in 0, 1, 2, 3 4-dimensional dataset)
    filtered_dataset = np.take(dataset, filter_indeces, axisNum)
    # print("New Shape of Dataset:", filtered_dataset.shape, "\n")
    return(filtered_dataset)

def loadTaskCSVs(task_folder):
    print("\nLoading", config.selectedTask, "CSVs for contigification")
    print("==========\n")
    arrays = []
    subjectWriteOrder = []
    for sub in tqdm(os.listdir(task_folder)):
        subNum = sub[:config.participantNumLen]
        if subNum not in subjectWriteOrder:
            subjectWriteOrder.append(subNum)
    print("\nApplying Artifact Masks")
    print("==========\n")
    for num in tqdm(subjectWriteOrder):
        eeg, art = load_csv(task_folder+"/", num)
        masked = apply_art_mask_nan(eeg, art)
        arrays.append(filter_my_channels(masked, config.network_channels, 1))
    return(arrays, subjectWriteOrder)

def sec_to_cyc(seconds):
    return(seconds * 250)

def generate_sparse_contigs(run, length):
    i = 0
    contigs = []
    startindexes = []
    while i < run.shape[0]-length:
        stk = run[i:(i+length),:]
        if not np.any(np.isnan(stk)):
            contigs.append(stk)
            startindexes.append(i)
            i+=length
        else:
            i+=1
    return contigs

def generate_all_contigs(run, length):
    i = 0
    contigs = []
    startindexes = []
    while i < run.shape[0]-length:
        stk = run[i:(i+length),:]
        contigs.append(stk)
        startindexes.append(i)
        i+=length
    return contigs, startindexes

def get_contigs_from_trials(trial):
    cycles = config.contigLength
    sub_contigs = generate_sparse_contigs(trial, cycles)
    return(sub_contigs)

def filter_amplitude(contig, bads):
    amp = np.sum(np.square(contig))
    if amp in bads:
        return(True)
    else:
        return(False)

def contigs_to_csv(batch, prefix, subject_number):
    num_tossed = 0
    i = 0
    sub_step = 0
    for contig in batch:
        if (len(contig)>0):
            if filter_amplitude(contig, bads=[0]):
                num_tossed+=1
            else:
                np.savetxt(contigsFolder+"/"+subject_number+"_"+str(i)+".csv", contig, delimiter=",", fmt="%2.0f")
                i+=1
        sub_step+=1

working_path = config.studyDirectory+"/"+config.selectedTask
trials_array, subject_array = loadTaskCSVs(working_path)
print("\n Contigifying CSV data")
print("==========\n")
pbar = tqdm(total=len(trials_array))
for sub_trial in zip(trials_array, subject_array):
    contigs = get_contigs_from_trials(sub_trial[0])
    contigs_to_csv(contigs, config.selectedTask, sub_trial[1])
    pbar.update(1)
pbar.close()
