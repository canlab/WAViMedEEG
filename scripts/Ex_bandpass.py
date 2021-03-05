import Prep
import Standard
import sys
import os
from tqdm import tqdm
import config
import argparse


def main():

    parser = argparse.ArgumentParser(
        description='Options for Standard.BandFilter method')

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
                        help='(Default: None) Study folder containing '
                        + 'dataset. '
                        + 'If None, performed on all studies available.')

    parser.add_argument('--task',
                        dest='task',
                        type=str,
                        default="P300",
                        help="(Default: P300) Task to use from config.py: "
                        + str([val for val in config.tasks]))

    parser.add_argument('--type',
                        dest='type',
                        type=str,
                        default="bandpass",
                        help="(Default: bandpass) Which band filter method "
                        + "should be applied: "
                        + "lowpass, highpass, bandstop, bandpass")

    parser.add_argument('--band',
                        dest='band',
                        type=str,
                        default="delta",
                        help="(Default: delta) "
                        + "Frequency band used for band ranges: "
                        + str([val for val in config.frequency_bands]))

    # save the variables in 'args'
    args = parser.parse_args()

    studies_folder = args.studies_folder
    study_name = args.study_name
    task = args.task
    type = args.type
    band = args.band

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

    if type not in ["lowpass", "highpass", "bandpass", "bandstop"]:
        print(
            "Invalid entry for type, "
            + "must be one of: lowpass, highpass, bandpass, bandstop")
        raise ValueError
        sys.exit(3)

    if band not in config.frequency_bands:
        print(
            "Invalid entry for band, "
            + "must be one of: " + [val for val in config.frequency_bands])
        raise ValueError
        sys.exit(3)

    if study_name is None:
        my_studies = os.listdir(studies_folder)
    else:
        my_studies = [study_name]

    for study_folder in my_studies:
        print("Processing:", study_folder)
        mytask = Standard.BandFilter(
            studies_folder+"/"+study_folder,
            task,
            type=type)
        mytask.gen_taskdata(band)
        mytask.write_taskdata()


if __name__ == '__main__':
    main()
