import os

# INITIALIZING STUDY FILES
# ====================

# full path of study directory
# Ex. /StudyDirectory
# ----------> /raw
# --------------------> *.eeg
# --------------------> *.evt
# --------------------> *.art
studyDirectory = "/home/clayton/science/CANlab/EEGstudies/CANlabStudy"
resultsBaseDir = studyDirectory+"/results"
sampleRate = 250 # in Hz

selectedTask = "p300" # in general, the task which will be used for triggered analysis step

# I. WAVi to CSV CONVERSIONS
# ====================
stepOneTrigger = "no" # enter 'yes' or 'no' to skip command line prompt

# this package expects a naming convention for raw EEG files:
# 3-digit participant number, underscore, task name, .art / .eeg / .evt
# Ex: 104_p300.eeg
# if you want to use a different length participant identifier, specify it here
participantNumLen = 3

# subjectsTasksKeys={}
#
# for task in os.listdir(studyDirectory):
#     if task != "raw":
#         subjectsTasksKeys[task]=[[os.listdir(studyDirectory+"/"+task)]:2]
#
# for key in subjectsTasksKeys:
#     print(key, " : ", subjectsTasksKeys[key])

# current supported tasks are
# p300
# flanker
# chronic
# rest

# if you need to add a new one, you currently have to specify
# whether it will use .evt in the loadEEGdataNumpy function of wavi_to_csv
# once you do, please make a pull request so others can use it as well

# II. MNE CONFIGURATION
# ====================
stepTwoTrigger = "no" # enter 'yes' or 'no' to skip command line prompt
# numChannels = 19 # default 19 for WAVi headset

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

# III. CONTIG GENERATION
# ====================
stepThreeTrigger = "no" # enter 'yes' or 'no' to skip command line prompt
contigLength = 250 # length of segmented epochs, in cycles, at 250 Hz

# for accurate sensors in spectral analysis,
# keep these in the same order
# as the default list above (channel_names)
# it will not affect contigs themselves
# and you can change the order after you've run step 3
network_channels = [
    'P3',
    'P4',
    'Pz'
]

# IV. NEURAL NETWORK DIFFERENTIATION
# ====================
stepFourTrigger = "no" # enter 'yes' or 'no' to skip command line prompt

source = studyDirectory+"/contigs_p300_250_alpha"
evalPath = studyDirectory+"/contigs_p300_250_alpha"
resultsPath = studyDirectory+"/results/jacknife_evaluation_alpha"

permuteLabels = False

# dictionary of first-index subject number and a respective 4-character name for the group
subjectKeys = {
    0: "pilt", # pilot
    1: "pain",
    2: "ctrl"
}

# network hyperparameters
learningRate = 0.001
betaOne = 0.99
betaTwo = 0.999
numEpochs = 100

# Supplement
# SCORE DISTRIBUTIONS
# ====================
stepFourSuppTrigger = "no" # enter 'yes' or 'no' to skip command line prompt

# V. FREQUENCY DECOMPOSITION
# ====================
stepFiveTrigger = "no" # enter 'yes' or 'no' to skip command line prompt
alphaRange = [7.0, 13.0] # bounds of alpha peak search windows frequencies
# Savitzky-Golay filter
window_length = 11
poly_order = 5
mdiff = 0.2 # minimal height difference distinguishing a primary peak from competitors

# VI. ROC CURVE
# ====================
stepSixTrigger = "yes" # enter 'yes' or 'no' to skip command line prompt
roc_source = studyDirectory+"/results/"
roc_type = "shuffle" # 'shuffle' or 'filter'

# VII. BANDPASS FILTER
# ====================
stepSevenTrigger = "no" # enter 'yes' or 'no' to skip command line prompt
bandpassSource = source
#   delta, 0.1-4, theta: 4-8, alpha: 8-12, beta: 16-31, gamma: 32-60
bandpassBounds = [32, 60]
bandpassName = "gamma"

numberExamples = 5

# VIIs. FILTER PLOTS
# ====================
stepSevenSuppTrigger = "no" # enter 'yes' or 'no' to skip command line prompt
filterPlotContig = "104_77"

# VIII. CONTIG PLOT
# ====================
stepEightTrigger = "no" # enter 'yes' or 'no' to skip command line prompt
plotSource = studyDirectory+"/contigs_p300_250/101_29.csv"
