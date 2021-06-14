import sys
sys.path.append('..')
from src import Prep
from src import Standard
from src import config
import os
import numpy as np
from tqdm import tqdm
import argParser


def main():

    args = argParser.main([
        'studies_folder',
        'study_names',
        'task',
        'frequency_band'
    ])

    studies_folder = args.studies_folder
    study_names = args.study_names
    task = args.task
    frequency_band = args.frequency_band

    if study_names is None:
        my_studies = os.listdir(studies_folder)
    else:
        my_studies = study_names

    for study_folder in my_studies:
        print("Processing:", study_folder)
        try:
            os.mkdir(studies_folder+"/"+study_folder+"/"+"coherences")
        except FileExistsError:
            print("Destination coherence folder already exists.")
        try:
            os.mkdir(studies_folder+"/"+study_folder+"/"+"coherences"+"/"+task)
        except FileExistsError:
            print("Destination coherence subfolder already exists.")
            sys.exit(1)

        for fname in tqdm([fname for fname in os.listdir(
            studies_folder+"/"+study_folder+"/"+task) if (".eeg" in fname)\
            and (filter_band in fname)]):

            map = Standard.CoherenceMap(
                np.genfromtxt(
                    studies_folder\
                    + "/"\
                    + study_folder\
                    + "/"\
                    + task\
                    + "/"\
                    + fname))
            map.write(
                studies_folder\
                + "/"\
                + study_folder\
                + "/"\
                + "coherences"\
                + "/"\
                + task\
                + "/"\
                + fname[:-4]\
                + ".coh")

if __name__ == '__main__':
    main()
