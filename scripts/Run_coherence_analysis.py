import sys
sys.path.append('..')
from src import Prep
from src import Standard
from src import Networks
from src import config
import os
from tqdm import tqdm
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

    parser.add_argument('--reference_study',
                        dest='reference_study',
                        type=str,
                        default=None,
                        nargs='+',
                        help='(Default: None) Study folder containing '
                        + 'reference dataset. ')

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
                        default="alpha",
                        help="(Default: alpha) "
                        + "Frequency band used for band ranges: "
                        + str([val for val in config.frequency_bands]))

    # save the variables in 'args'
    args = parser.parse_args()

    studies_folder = args.studies_folder
    reference_study = args.reference_study
    study_name = args.study_name
    task = args.task
    type = args.type
    band = args.band

    # if reference_study is None:
    #     ref_studies = os.listdir(studies_folder)
    # else:
    ref_studies = [study+"/coherences/"+task for study in reference_study]

    # if study_name is None:
    #     my_studies = os.listdir(studies_folder)
    # else:
    my_studies = [study+"/coherences/"+task for study in study_name]

    # ERROR HANDLING
    if not os.path.isdir(studies_folder):
        print(
            "Invalid entry for studies_folder, "
            + "path does not exist as directory.")
        raise FileNotFoundError
        sys.exit(3)

    if ref_studies is not None:
        for study in reference_study:
            if not os.path.isdir(os.path.join(studies_folder, study)):
                print(
                    "Invalid entry for reference_study, "
                    + "path does not exist as directory.")
                raise FileNotFoundError
                sys.exit(3)

    if my_studies is not None:
        for study in study_name:
            if not os.path.isdir(os.path.join(studies_folder, study)):
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

    ref_coh = Networks.Coherence()
    for study in ref_studies:
        ref_coh.load_data(studies_folder+"/"+study)
    ref_coh.score(band=band)
    # ref_coh.draw(weighting=True)

    study_coh = Networks.Coherence()
    for study in my_studies:
        study_coh.load_data(studies_folder+"/"+study)
    study_coh.score(band=band)
    study_coh.threshold(reference=ref_coh.G, z_score=1)
    study_coh.draw(weighting=True, threshold=True)

if __name__ == '__main__':
    main()
