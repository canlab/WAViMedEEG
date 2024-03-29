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

exclude_subs = [
    '1000',
    '1004',
    '1020',
    '2007',
    '2017',
    '2013',
    '2001',
    '2010',
    '2024']

group_names = {
    0: "Pilot",
    1: "Control",
    2: "Pain",
    3: "Rehab"
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
        "P300_EC_4_Min"],
    "FLNK": [
        "Flanker",
        "flanker"],
    "CLSD": [
        "Eyes_Closed_Resting",
        "CLSD-2"],
    "OPEN": [
        "Eyes_Open_Focused"],
    "REST": [
        "Rest",
        "rest"],
    "CRNC": [
        "Chronic",
        "chronic"],
    "SOMA": [
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
    "delta": [0, 2],
    "theta": [2, 6],
    "alpha": [6, 13],
}
