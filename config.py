import os

# INITIALIZING STUDY FILES
# ====================

# full path of study directory
# Ex. /StudyDirectory
# ----------> /raw
# --------------------> *.eeg
# --------------------> *.evt
# --------------------> *.art
studyDirectory = "/home/claytonjschneider/science/CANlab/EEGstudies/CANlabStudy"
sampleRate = 250 # in Hz

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
mneTask = "p300"

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
selectedTask = "thumper" # one of the supported tasks listed above, to be exported to contigs
contigLength = 250 # in cycles, at 250 Hz

network_channels = [
    'Pz',
    'P3',
    'P4'
]

# IV. NEURAL NETWORK DIFFERENTIATION
# ====================
stepFourTrigger = "no" # enter 'yes' or 'no' to skip command line prompt

source = studyDirectory+"/contigs_p300"
evalPath = "/home/claytonjschneider/science/CANlab/EEGstudies/WAViPainStudy/contigs_thumper"

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


resultsDir = "/home/claytonjschneider/science/CANlab/EEGstudies/CANlabStudy/results_jacknife_shuffle"

# Supplement
# SCORE DISTRIBUTIONS
# ====================
stepFourTriggerDistributions = "yes" # enter 'yes' or 'no' to skip command line prompt
