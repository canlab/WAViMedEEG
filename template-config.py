# INITIALIZING STUDY FILES
# ====================

# full path of study directory
# Ex. /StudyDirectory
# ----------> /raw
# --------------------> *.eeg
# --------------------> *.evt
# --------------------> *.art
studyDirectory = "" # ex. /home/user/studyaboutstudying
sampleRate = 250 # in Hz

# I. WAVi to CSV CONVERSIONS
# ====================
stepOneTrigger = "" # 'yes' or 'no' to skip command line prompt

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
# whether it will use .evt in the loadEEGdataNumpy function of wavi_to_csv
# once you do, please make a pull request so others can use it as well

# II. MNE CONFIGURATION
# ====================
stepTwoTrigger = "" # 'yes' or 'no' to skip command line prompt
numChannels = 19 # default 19 for WAVi headset

# III. CONTIG GENERATION
# ====================
stepThreeTrigger = "" # 'yes' or 'no' to skip command line prompt
selectedTask = "" # one of the supported tasks listed above, to be exported to contigs
contigLength = 750 # in cycles, at 250 Hz, default 3 seconds

# IV. NEURAL NETWORK DIFFERENTIATION
# ====================
stepFourTrigger = "" # 'yes' or 'no' to skip command line prompt
