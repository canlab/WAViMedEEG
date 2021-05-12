import sys
sys.path.append('..')
from src import Prep
from src import Standard
from src import config
import os
from tqdm import tqdm
import argParser


def main():

    args = argParser.main([
        'studies_folder',
        'study_names',
        'task',
        'filter_type',
        'frequency_band'
    ])

    studies_folder = args.studies_folder
    study_names = args.study_names
    task = args.task
    filter_type = args.filter_type
    frequency_band = args.frequency_band

    for study_name in study_names:
        print("Processing:", study_name)
        mytask = Standard.BandFilter(
            studies_folder+"/"+study_name,
            task,
            type=filter_type)
        mytask.gen_taskdata(frequency_band)
        mytask.write_taskdata()


if __name__ == '__main__':
    main()
