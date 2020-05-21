import os
import numpy as np
from numpy import array, ma, genfromtxt
import pandas as pd
import config
from tqdm import tqdm

try:
    os.mkdir(config.studyDirectory+"/contigs")
except:
    print("Contigs folder already exists.")

try:
    contigsFolder = config.studyDirectory+"/contigs"+"/"+config.selectedTask+"_"+str(config.contigLength)
    os.mkdir(contigsFolder)
except:
    if len(os.listdir(contigsFolder)) > 0:
        print("You already had the specified contigs folder. Go back and clean up.\n")
        quit()

def load_csv(task_folder, sub):
    eeg = []
    fnames_eeg = [fname for fname in os.listdir(task_folder) if sub in fname]
    fnames_eeg = [fname for fname in fnames_eeg if "_eeg_" in fname]
    for band_eeg in fnames_eeg:
        eeg.append(genfromtxt(task_folder+"/"+band_eeg, delimiter=","))
    art = genfromtxt(task_folder+"/"+sub+"_art.csv", delimiter=",")
    return(eeg, art, fnames_eeg)

def apply_art_mask_nan(data, artifact):
    mx = []
    for band_eeg in data:
        mxi = ma.masked_array(band_eeg, mask=artifact)
        mxi = ma.filled(mxi.astype(float), np.nan)
        mx.append(mxi)
    return(mx)

def apply_art_mask_zero(data, artifact):
    mx = []
    for band_eeg in data:
        mxi = ma.masked_array(band_eeg, mask=artifact)
        mxi = ma.filled(mxi.astype(float), 0)
        mx.append(mxi)
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
    network_dataset = []
    for band_eeg in dataset:
        network_dataset.append(np.take(band_eeg, filter_indeces, axisNum))
    # print("New Shape of Dataset:", filtered_dataset.shape, "\n")
    return(network_dataset)

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

    fnamesBySubject = []
    for num in tqdm(subjectWriteOrder):
        eeg, art, fnames_eeg = load_csv(task_folder+"/", num)
        fnamesBySubject.append((num, fnames_eeg))
        masked = apply_art_mask_nan(eeg, art)
        arrays.append(filter_my_channels(masked, config.network_channels, 1))

    return(arrays, fnamesBySubject)

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

def contigs_to_csv(batch, trial_fname):
    num_tossed = 0
    i = 0
    sub_step = 0
    for contig in batch:
        if (len(contig)>0):
            if filter_amplitude(contig, bads=[0]):
                num_tossed+=1
            else:
                np.savetxt(contigsFolder+"/"+str(trial_fname[:3])+"_"+str(trial_fname[8:-4])+"_"+str(i)+".csv", contig, delimiter=",", fmt="%2.0f")
                i+=1
        sub_step+=1

working_path = config.studyDirectory+"/"+config.selectedTask
trials_array, subject_filenames = loadTaskCSVs(working_path)
print("\n Contigifying CSV data")
print("==========\n")
pbar_len = len(trials_array)*len(trials_array[0])
pbar = tqdm(total=pbar_len)
subject_filenames = [set[1] for set in subject_filenames]
for sub_trial in zip(trials_array, subject_filenames):
    i = 0
    for band_eeg in sub_trial[0]:
        contigs = get_contigs_from_trials(band_eeg)
        contigs_to_csv(contigs, sub_trial[1][i])
        pbar.update(1)
        i+=1
pbar.close()
