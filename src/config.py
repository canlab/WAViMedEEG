"""
=== WAVi Analysis Toolbox Config Overview ===

Welcome to the WAVi EEG Analysis Toolbox.
Before running this code, set your configuration variables below.
Most classes, scripts, etc., will reference it.

INITIALIZING STUDY FILES
========================
before beginning with new data, format your new study directory as follows:
~/.../path/to/my_studies/newstudydir
----------> /raw
--------------------> <subjID>_<taskID>.eeg
--------------------> <subjID>_<taskID>.evt
--------------------> <subjID>_<taskID>.art
--------------------> *...

========================
"""

# this package expects a naming convention for raw EEG files:
# n-digit participant number, underscore, task name, .art / .eeg / .evt
# Ex: 104_p300.eeg
# if you want to use a different length participant identifier, specify it here
participantNumLen = 4  # default length

# if you are working with multiple studies at once,
# set the parent directory where you will keep them all
my_studies = "/wavi/EEGstudies"
study_directory = 'canlab pain'

group_names = {
    0: "Pilot",
    1: "Control",
    2: "Pain",
    3: "Rehab"
}

group_colors = {
    0: "black",
    1: "green",
    2: "purple",
    3: "orange"
}

ref_folders = [
    "ref 24-30",
    "ref 31-40",
    "ref 41-50",
    "ref 51-60",
    # "ref 61-70",
    # "ref 71-80",
    # "ref 81+"
    ]

# dictionary where key==task names, = 'unclean' task names
# so that filenames can be cleaned automatically
tasks = {
    "P300": [
        "p300",
        "P300_Eyes_Closed",
        "P300-Sync_Blink",
        "P300_Eye_Closed",
        "P300s",
        "P300-2",
        "P300_EC_4_Min",
        "P300"],
    "FLNK": [
        "FLNK",
        "Flanker",
        "flanker"],
    "CLSD": [
        "CLSD",
        "Eyes_Closed_Resting",
        "CLSD-2"],
    "OPEN": [
        "OPEN",
        "Eyes_Open_Focused"],
    "REST": [
        "REST",
        "Rest",
        "rest"],
    "CRNC": [
        "CRNC",
        "Chronic",
        "chronic"],
    "SOMA": [
        "SOMA",
        "SMS"]}


sample_rate = 250  # in Hz

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
    'Pz']

networkx_positions = [
    (25, 60), # Fp1
    (35, 60), # Fp2
    (20, 50), # F3
    (40, 50), # F4
    (15, 52), # F7
    (45, 52), # F8
    (20, 35), # C3
    (40, 35), # C4
    (20, 20), # P3
    (40, 20), # P4
    (23, 10), # O1
    (37, 10), # O2
    (10, 35), # T3
    (50, 35), # T4
    (13, 20), # T5
    (47, 20), # T6
    (30, 48), # Fz
    (30, 32), # Cz
    (30, 17)] # Pz

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
    'Pz']

custom_art_map = [
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    1,
    1,
    0,
    0,
    0,
    0,
    0,
    0,
    0
]

frequency_bands = {
    "delta": [0, 4],
    "theta": [2, 6],
    "alpha": [6, 13],
    "beta": [13, 30],
    "gamma": [30, 45]
}
