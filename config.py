import os

# INITIALIZING STUDY FILES
# ====================

# full path of study directory
# Ex. /StudyDirectory
# ----------> /raw
# --------------------> *.eeg
# --------------------> *.evt
# --------------------> *.art
studyDirectory = "/home/clayton/science/CANlab/EEGstudies/ref pain"
resultsBaseDir = studyDirectory+"/results"
max_tree_depth = 3

sampleRate = 250 # in Hz

# dictionary of first-index subject number and a respective 4-character name for the group
subjectKeys = {
    0: "pilt", # pilot
    1: "pain",
    2: "ctrl"
}

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

selectedTask = "p300" # in general, the task which will be used for triggered analysis step


# I. WAVi to CSV CONVERSIONS
# ====================
stepOneTrigger = "no" # enter 'yes' or 'no' to skip command line prompt

# this package expects a naming convention for raw EEG files:
# 3-digit participant number, underscore, task name, .art / .eeg / .evt
# Ex: 104_p300.eeg
# if you want to use a different length participant identifier, specify it here
participantNumLen = 3

# current supported tasks are
# p300
# flanker
# chronic
# rest

# if you need to add a new one, you currently have to specify
# whether it will use .evt in the loadEEGdataNumpy function of wavi_to_csv.py
# once you do, please make a pull request so others can use it as well


# II. MNE CONFIGURATION
# ====================
stepTwoTrigger = "no" # enter 'yes' or 'no' to skip command line prompt
# numChannels = 19 # default 19 for WAVi headset


# III. CONTIG GENERATION
# ====================
stepThreeTrigger = "no" # enter 'yes' or 'no' to skip command line prompt
contigLength = 1250 # length of segmented epochs, in cycles, at 250 Hz

# for accurate sensors in spectral analysis,
# keep these in the same order
# as the default list above (channel_names)
# network_channels = [
#     'P3',
#     'P4',
#     'Pz'
# ]
network_channels = [
    'P3',
    'P4',
    'Pz'
]

# IV. Convolutional Neural Network
# ====================
source = "/home/clayton/science/CANlab/EEGstudies/CANlabStudy"+"/contigs/"+selectedTask+"_"+str(contigLength)
evalPath = studyDirectory+"/contigs/"+selectedTask+"_"+str(contigLength)
resultsPath = resultsBaseDir+"/model_evaluation"+"_"+selectedTask+"_"+str(contigLength)+"_ab_2"

permuteLabels = False

# network hyperparameters
learningRate = 0.001
betaOne = 0.99
betaTwo = 0.999
numEpochs = 1000


# IVa. TRAIN / SAVE NEURAL NETWORK WEIGHTS
# convnet_save_weights
# ====================
stepFourATrigger = "no" # enter 'yes' or 'no' to skip command line prompt


# IVb. NEURAL NETWORK DIFFERENTIATION JACKNIFE
# ====================
stepFourBTrigger = "no" # enter 'yes' or 'no' to skip command line prompt


# IVc. EVALUATE A SAVED MODEL
# ====================
stepFourCTrigger = "no"


# Va. POWER SPECTRAL DENSITY CURVES
# ====================
stepFiveATrigger = "no" # enter 'yes' or 'no' to skip command line prompt
# alphaRange = [7.0, 13.0] # bounds of alpha peak search windows frequencies
# Savitzky-Golay filter
# window_length = 11
# poly_order = 5
# mdiff = 0.2 # minimal height difference distinguishing a primary peak from competitors

# Vb. CEPSTRUMS
# ====================
stepFiveBTrigger = "no" # enter 'yes' or 'no' to skip command line prompt

# Vs. PLOT POWER SPECTRAL DENSITY CURVES
# ====================
stepFiveSTrigger = "no"
plot_contig_lead = "209_1"


# VIa. ROC CURVE
# ====================
stepSixATrigger = "no" # enter 'yes' or 'no' to skip command line prompt
roc_source = resultsBaseDir+"/jacknife_evaluation_p300_1250"
req_results_keyword = "jacknife" # optional to require roc source folders to contain a keyword
roc_type = "filter" # 'shuffle' or 'filter'

# VIb. Plot Many Probability Distribution Functions
# ====================
stepSixBTrigger = "no" # enter 'yes' or 'no' to skip command line prompt
roc_sourceDir_many = "/home/clayton/science/CANlab/EEGstudies"
roc_source_keyword_many = "ref"
req_many_eval_path = "/results/model_evaluation_p300_1250_ab_2" # path of specific evaluation within each study folder

# VII. BANDPASS FILTER
# ====================
stepSevenTrigger = "no" # enter 'yes' or 'no' to skip command line prompt
# delta, 0.1-4, theta: 4-8, alpha: 8-12, beta: 16-31, gamma: 32-60
# you can comment out different bands to mute them from being admitted to the network training / evaluation
frequency_bands = [
    # ("delta", [0.1, 4]),
    # ("theta", [4, 8]),
    ("alpha", [8, 12]),
    ("beta", [16, 31]),
    # ("gamma", [32, 60])
    ]


# VIIs. FILTER PLOTS
# ====================
stepSevenSuppTrigger = "no" # enter 'yes' or 'no' to skip command line prompt
filterPlotContig = "104_77"


# VIII. MNE PLOT
# ====================
stepEightTrigger = "no" # enter 'yes' or 'no' to skip command line prompt
plotSource = studyDirectory+"/contigs_p300_250_no/101_29.csv"

# IX. SUPPORT VECTOR MACHINE
# ====================
stepNineTrigger = "no" # enter 'yes' or 'no' to skip command line prompt
svm_source = studyDirectory+"/spectral/"+selectedTask+"_"+str(contigLength)
kernel_type = 'rbf' # one of ['linear', 'poly', 'rbf']
