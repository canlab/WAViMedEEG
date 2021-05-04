import sys
sys.path.append('..')
from src import Prep
from src import Standard
from src import config
import os
import numpy as np
from tqdm import tqdm
import argparse


def main():

    parser = argparse.ArgumentParser(
        description='Options for Standard.CoherenceMap method')

    parser.add_argument('--studies_folder',
                        dest='studies_folder',
                        type=str,
                        default=config.my_studies,
                        help="(Default: " + config.my_studies + ") Path to "
                        + "parent folder containing study folders")

    parser.add_argument('--study_name',
                        dest='study_name',
                        type=str,
                        default=None,
                        nargs='+',
                        help='(Default: None) Study folder containing '
                        + 'dataset. '
                        + 'If None, performed on all studies available.')

    parser.add_argument('--task',
                        dest='task',
                        type=str,
                        default="P300",
                        help="(Default: P300) Task to use from config.py: "
                        + str([val for val in config.tasks]))

    parser.add_argument('--filter_band',
                        dest='filter_band',
                        type=str,
                        default='nofilter',
                        help="(Default: nofilter) Bandfilter to be used in "
                        + "analysis steps, such "
                        + "as: 'noalpha', 'delta', or 'nofilter'")

    # save the variables in 'args'
    args = parser.parse_args()

    studies_folder = args.studies_folder
    study_name = args.study_name
    task = args.task
    filter_band = args.filter_band

    # ERROR HANDLING
    if not os.path.isdir(studies_folder):
        print(
            "Invalid entry for studies_folder, "
            + "path does not exist as directory.")
        raise FileNotFoundError
        sys.exit(3)

    if study_name is not None:
        if not os.path.isdir(os.path.join(studies_folder, study_name)):
            print(
                "Invalid entry for study_name, "
                + "path does not exist as directory.")
            raise FileNotFoundError
            sys.exit(3)

    if task not in config.tasks:
        print(
            "Invalid entry for task, "
            + "not accepted as regular task name in config.")
        raise ValueError
        sys.exit(3)

    if filter_band == "nofilter":
        pass
    elif any(band == filter_band for band in config.frequency_bands):
        pass
    elif any("no"+band == filter_band for band in config.frequency_bands):
        pass
    elif any("lo"+band == filter_band for band in config.frequency_bands):
        pass
    elif any("hi"+band == filter_band for band in config.frequency_bands):
        pass
    else:
        print("That is not a valid filterband option.")
        raise ValueError
        sys.exit(3)

    if study_name is None:
        my_studies = os.listdir(studies_folder)
    else:
        my_studies = [study_name]

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
