import os
import numpy as np
from numpy import array, ma, genfromtxt
import pandas as pd
import config

# class taskInfo():
#     def __init__(self):
#         self.fnames = os.listdir()
#         self.ctrl_fnames = os.listdir("2")
#         self.pain_subs = []
#         self.ctrl_subs = []
#         for fname in self.pain_fnames:
#             if(fname[:3] not in self.pain_subs):
#                 self.pain_subs.append(fname[:3])
#         for fname in self.ctrl_fnames:
#             if(fname[:3] not in self.ctrl_subs):
#                 self.ctrl_subs.append(fname[:3])

os.mkdir(config.studyDirectory+"/contigs")
print("I made the contigs folder")

def load_csv(prefix, sub):
    eeg = genfromtxt(prefix+"/"+sub+"_eeg.csv", delimiter=",")
    art = genfromtxt(prefix+"/"+sub+"_art.csv", delimiter=",")
    return(eeg, art)

def apply_art_mask_nan(data, artifact):
    mx = ma.masked_array(data, mask=artifact)
    mx = ma.filled(mx.astype(float), np.nan)
    return(mx)

# def apply_art_mask_zero(data, artifact):
#     mx = ma.masked_array(data, mask=artifact)
#     mx = ma.filled(mx.astype(float), 0)
#     return(mx)

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

def loadTaskCSVs(group_folder):
    arrays = []
    for sub in os.listdir(group_folder):
        eeg, art = load_csv(group_folder+"/", sub[:config.participantNumLen])
        masked = apply_art_mask_nan(eeg, art)
        arrays.append(filter_my_channels(masked, config.network_channels, 1))
    return(arrays)

# def load_flanker():
#     go_to_task("flanker")
#     flanker_pain = []
#     flanker_ctrl = []
#     info = taskInfo()
#     os.chdir("1")
#     for sub in info.pain_subs:
#         eeg, art = load_csv(sub)
#         flanker_pain.append(apply_art_mask_zero(eeg, art))
#     os.chdir("..")
#     os.chdir("2")
#     for sub in info.ctrl_subs:
#         eeg, art = load_csv(sub)
#         flanker_ctrl.append(apply_art_mask_zero(eeg, art))
#     os.chdir("..")
#     return(flanker_pain, flanker_ctrl)

# def load_chronic():
#     go_to_task("chronic")
#     chronic_pain = []
#     chronic_ctrl = []
#     info = taskInfo()
#     os.chdir("1")
#     for sub in info.pain_subs:
#         eeg, art = load_csv(sub)
#         chronic_pain.append(apply_art_mask_zero(eeg, art))
#     os.chdir("..")
#     os.chdir("2")
#     for sub in info.ctrl_subs:
#         eeg, art = load_csv(sub)
#         chronic_ctrl.append(apply_art_mask_zero(eeg, art))
#     os.chdir("..")
#     return(chronic_pain, chronic_ctrl)

# def load_rest():
#     go_to_task("rest")
#     rest_pain = []
#     rest_ctrl = []
#     info = taskInfo()
#     os.chdir("1")
#     for sub in info.pain_subs:
#         eeg, art = load_csv(sub)
#         rest_pain.append(apply_art_mask_zero(eeg, art))
#     os.chdir("..")
#     os.chdir("2")
#     for sub in info.ctrl_subs:
#         eeg, art = load_csv(sub)
#         rest_ctrl.append(apply_art_mask_zero(eeg, art))
#     os.chdir("..")
#     return(rest_pain, rest_ctrl)

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
    return contigs, startindexes

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

def get_contigs_from_trials(trials):
    all_contigs = []
    cycles = config.contigLength
    for run in trials:
        all_contigs.append(generate_sparse_contigs(run, cycles))
    return(all_contigs)

# def amps_by_subject(contigs):
#     amps_by_subject=[]
#     for sub in contigs:
#         amps = [np.sum(np.square(contig)) for contig in sub[0]]
#         amps_by_subject.append(amps)
#     return(amps_by_subject)
#
# def amps_by_group(contigs):
#     amps_by_group=[]
#     for sub in contigs:
#         for contig in sub[0]:
#             amp = np.sum(np.square(contig))
#             amps_by_group.append(amp)
#     return(amps_by_group)
#
# print((p300_pain_contigs[0][0][1]))
#
# p300_pain_amps = amps_by_group(p300_pain_contigs)
# p300_ctrl_amps = amps_by_group(p300_ctrl_contigs)
#
# chronic_pain_amps = amps_by_group(chronic_pain_contigs)
# chronic_ctrl_amps = amps_by_group(chronic_ctrl_contigs)

# import matplotlib.pyplot as plt

# n, bins, patches = plt.hist(x=chronic_ctrl_amps, bins='auto', color='blue', alpha=0.7, rwidth=0.85)
# plt.grid(axis='y', alpha=0.75)
# plt.xlabel('Amplitude')
# plt.xticks(rotation=315)
# plt.ylabel('Frequency')
# plt.title('Voltage Amplitude Distribution Across Ctrl Subjects')
# plt.text(23,45,r'$\\mu=15, b=3$')
# maxfreq = n.max()
#
# plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
#
# n, bins, patches = plt.hist(x=p300_ctrl_amps, bins='auto', color='blue', alpha=0.7, rwidth=0.85)
# plt.grid(axis='y', alpha=0.75)
# plt.xlabel('Amplitude')
# plt.xticks(rotation=315)
# plt.ylabel('Frequency')
# plt.title('Voltage Amplitude Distribution Across Ctrl Subjects')
# plt.text(23,45,r'$\\mu=15, b=3$')
# maxfreq = n.max()
#
# plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
#
# p300_pain_std = np.std(p300_pain_amps)
# p300_ctrl_std = np.std(p300_ctrl_amps)
#
# print(p300_pain_std)
# print(p300_ctrl_std)

# def load_per_task(task):
#     switcher = {
#         p300: load_p300,
#         flanker: load_flanker,
#         chronic: load_chronic,
#         rest: load_rest
#     }
#     func = switcher.get(argument, lambda: \"Invalid task\")

def filter_amplitude(contig, bads=[]):
    amp = np.sum(np.square(contig))
    if amp in bads:
        return(True)
    else:
        return(False)

def contigs_to_csv(batch, group, prefix):
    num_tossed = 0
    i = 0
    for sub in batch:
        if (len(sub[0])>0):
            for contig in sub[0]:
                if filter_amplitude(contig, bads=[0]):
                    num_tossed+=1
                else:
                    np.savetxt(config.studyDirectory+"/contigs"+"/"+group+"_"+prefix+"_"+str(i)+".csv", contig, delimiter=",", fmt="%2.0f")
                    i+=1
    print("I tossed", num_tossed, "contigs, due to specified amplitude filters.")

# def contigs_to_csv_split(group, task, prefix):
#     if(\"train\" not in os.listdir()):
#         os.mkdir("train")
#         os.mkdir("test")
#     if(task not in os.listdir("train")):
#         os.chdir("train")
#         os.mkdir(task)
#         os.chdir("../test")
#         os.mkdir(task)
#         os.chdir("..")
#     i = 0
#     for sub in group:
#         if(len(sub[0])>0):
#             for contig in sub[0]:
#                 if(i%3==0):
#                     np.savetxt(directory+"/contigs/test/"+task+"/"+prefix+"_"+str(i)+".csv", contig, delimiter=",", fmt="%2.0f")
#                 else:
#                     np.savetxt(directory+"/contigs/train/"+task+"/"+prefix+"_"+str(i)+".csv", contig, delimiter=",", fmt="%2.0f")
#                 i+=1
working_path = config.studyDirectory+"/"+config.selectedTask
for group in os.listdir(working_path):
    if group != "0":
        data_array = loadTaskCSVs(working_path+"/"+group)
        contigs = get_contigs_from_trials(data_array)
        contigs_to_csv(contigs, group, config.selectedTask)
