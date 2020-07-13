# Welcome to the WAVi EEG Analysis Toolbox
# Before running this code, set your configuration variables below
# then import config
import os

# INITIALIZING STUDY FILES
# ====================
# before beginning with new data, format your
# new study directory as follows:
# /path/to/mystudies/newstudydir
# ----------> /raw
# --------------------> *.eeg
# --------------------> *.evt
# --------------------> *.art

myStudies = "/wavi/EEGstudies" # if you are working with multiple studies at once, set the parent directory where you will keep them all
studyDirectory = myStudies+"/eyes closed" # specific study, for functions that only deal with 1 at a time

selectedTask = "fcsd" # in general, the task which will be used for triggered analysis step

# dictionary of first-index subject number and a respective 4-character name for the group
# subjectKeys = {
#     0: "pilt", # pilot
#     1: "pain",
#     2: "ctrl"
# }

subjectKeys = {
    0: "pilt", # pilot
    1: "clsd",
    2: "open"
}

# WAVi to CSV CONVERSIONS / HEADSET CONFIG
# ====================
# this package expects a naming convention for raw EEG files:
# n-digit participant number, underscore, task name, .art / .eeg / .evt
# Ex: 104_p300.eeg
# if you want to use a different length participant identifier, specify it here
participantNumLen = 4 # default 4
sampleRate = 250 # in Hz

# default channel names, customize if using non-WAVi headset
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

artDegree = 1 # highest number of WAVi-supplied artifact
# still accepted by the program, 0 (strict), 1 (loose), or 2 (none)

# current supported tasks are
# p300
# flanker / flnk
# chronic
# rest
# fcsd (eyes open focused)
# clsd (eyes closed rest)

# if you need to add a new one, you currently have to specify
# whether it will use .evt in the loadEEGdataNumpy function of wavi_to_csv.py
# once you do, please make a pull request so others can use it as well


# CONTIG SETTINGS
# ====================
contigLength = 1250 # length of segmented epochs, in cycles, at 250 Hz


# MACHINE LEARNING SETTINGS
# ====================
# train and eval sources for various ML functions
train_source = studyDirectory+"/contigs/"+selectedTask+"_"+str(contigLength)
eval_source = studyDirectory+"/contigs/"+selectedTask+"_"+str(contigLength)
model_file = "/home/clayton/science/CANlab/WAViMedEEG/saved_models/convnet/eye_model_latest/MyModel" # path where model will be saved to / loaded from, will overwrite!
wumbo = False # set to True if you want to permute labels during convnet.load_numpy_stack or other similar functions


# RESULTS FOLDER SETUP
# ====================
resultsBaseDir = studyDirectory+"/results" # change this if you want to rename your study's base results folder
resultsPath = resultsBaseDir+"/model_evaluation"+"_"+selectedTask+"_"+str(contigLength)+"_subjects" # path to which current analysis results will be written
# will break if tries to write on existing folder

# for accurate sensors in spectral analysis,
# keep these in the same order
# as the default list above (channel_names)
network_channels = [
    'Fp1',
    'Fp2',
    'P3',
    'P4',
    'Pz'
]


# CONVOLUTIONAL NEURAL NETWORK
# ====================
# network hyperparameters
learningRate = 0.001
betaOne = 0.9
betaTwo = 0.999
numEpochs = 100

# SUPPORT VECTOR MACHINE
# ====================
kernel_type = 'rbf' # one of ['linear', 'poly', 'rbf']


# PLOTTING
# ====================
plot_waves = studyDirectory+"/contigs/SMS_1250/002_alpha_17472.csv"
plot_art = studyDirectory+"/raw/001_p300.art"

# for contig plotting
plot_subject = "001" # just for title if not defined, just chooses random, and assumes path from studyDirectory and selectedTask
plot_contig = "43534" # same as above

# for results plotting / PDFs and ROCs
plot_req_results_keyword = "eyes" # optional to require roc/pdf plot study folders to contain a keyword
plot_req_results_path = "" # optional to require path of specific evaluation within 1 or many study folder(s)


# BANDPASS FILTER
# ====================
# you can comment out different bands to mute them from being admitted to the network training / evaluation
# format is ("name", [low-end, high-end]) tuple, in Hz
frequency_bands = [
    # ("delta", [0.1, 4]),
    # ("theta", [4, 8]),
    # ("alpha", [8, 12]),
    # ("beta", [16, 31]),
    # ("gamma", [32, 60]),
    ("nofilter", []),
    ]


# MISCELLANEOUS
# ====================
roc_type = "filter" if wumbo==False else "shuffle" # this determines whether plot colors are generated or hard-coded, hard-coded by default
max_tree_depth = 2 # max depth traversed by study trees printed by the program template 'master.py'
import help
help.viewStudyTree(studyDirectory, max_tree_depth)
