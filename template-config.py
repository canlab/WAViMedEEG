# ===================== WAVi Analysis Toolbox Config Overview ===================== #
# Welcome to the WAVi EEG Analysis Toolbox. Before running this code, set your      #
# configuration variables below. Most classes, scripts, etc., will reference it.    #

# INITIALIZING STUDY FILES
# ================================================================================= #
# before beginning with new data, format your new study directory as follows:       #
# ~/.../path/to/mystudies/newstudydir
# ----------> /raw
# --------------------> *.eeg
# --------------------> *.evt
# --------------------> *.art

# ----------------------
myStudies = "/wavi/EEGstudies" # if you are working with multiple studies at once, set the parent directory where you will keep them all
studyDirectory = myStudies+"/BrainStudy" # specific study, for functions that only deal with 1 at a time

tasks = {
    "P300": ["p300", "P300_Eyes_Closed", "P300-Sync_Blink", "P300_Eye_Closed", "P300s", "P300-2"],
    "FLNK": ["Flanker", "flanker"],
    "CLSD": ["Eyes_Closed_Resting", "CLSD-2"],
    "OPEN": ["Eyes_Open_Focused"],
    "REST": ["Rest", "rest"],
    "CRNC": ["Chronic", "chronic"],
    "SOMA": ["SMS"]
}


# this package expects a naming convention for raw EEG files:
# n-digit participant number, underscore, task name, .art / .eeg / .evt
# Ex: 104_p300.eeg
# if you want to use a different length participant identifier, specify it here
participantNumLen = 4 # default length
sampleRate = 250 # in Hz
artDegree = 2 # highest number of WAVi-supplied artifact
# still accepted by the program, 0 (strict), 1 (loose), or 2 (none)


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

# channels to be used for artifacting, contigification
# and ultimately used in neural net analyses
# for accurate sensors in spectral analysis,
# keep these in the same order
# as the default list above (channel_names)
network_channels = [
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

# kernel type for SVM analysis
kernel_type = "linear"
