import os
import re
import shutil
import config

def fixStandardNames(directory):
    oldWorkingDir = os.getcwd()
    os.chdir(directory)
    for filename in os.listdir():
        if "Flanker" in filename:
            os.rename(filename, filename[:4]+"flanker"+filename[-4:])
        elif "P300" in filename:
            os.rename(filename, filename[:4]+"p300"+filename[-4:])
        elif "EO_Baseline_12" in filename:
            os.rename(filename, filename[:4]+"chronic"+filename[-4:])
        elif "EO_Baseline_8" in filename:
            os.rename(filename, filename[:4]+"rest"+filename[-4:])
    os.chdir(oldWorkingDir)

fixStandardNames(config.studyDirectory+"/raw")

