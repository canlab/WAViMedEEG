import config
import os, sys

# if you generated contigs without doing a bandpass
# this will append "_nofilter" in place of the raw contigs, for integration with
# the optional multi-band dimension, allowing for refinement of components

taskfolders = os.listdir(config.studyDirectory)
taskfolders = [folder for folder in taskfolders if folder in ["chronic", "flanker", "p300", "rest", "SMS"]]

for folder in taskfolders:
    fnames = os.listdir(config.studyDirectory+"/"+folder)
    fnames = [fname for fname in fnames if "eeg" in fname]
    for fname in fnames:
        new_fname = fname[:-4]+"_nofilter"+".csv"
        os.rename(config.studyDirectory+"/"+folder+"/"+fname, config.studyDirectory+"/"+folder+"/"+new_fname)
