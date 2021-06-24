import sys
sys.path.append('..')
from src import ML
from src import config
from src.Standard import SpectralAverage
import os
import argParser
from tqdm import tqdm


def main():

    args = argParser.main([
        "studies_folder",
        "study_names",
        "task",
        "length",
        "channels",
        "artifact",
        "erp_degree",
        "filter_band",
        "fig_fname"
    ])

    studies_folder = args.studies_folder
    study_names = args.study_names
    task = args.task
    length = args.length
    channels = args.channels
    artifact = args.artifact
    erp_degree = args.erp_degree
    filter_band = args.filter_band
    fig_fname = args.fig_fname

    # patient_path points to our 'condition-positive' dataset
    # ex. patient_path =
    # "/wavi/EEGstudies/CANlab/spectra/P300_250_1111111111111111111_0_1"
    patient_paths = []
    # patient_path points to our 'condition-positive' dataset
    # ex. patient_path =
    # "/wavi/EEGstudies/CANlab/spectra/P300_250_1111111111111111111_0_1"
    for study_name in study_names:

        patient_path = studies_folder\
            + '/'\
            + study_name\
            + '/'\
            + 'spectra'\
            + '/'\
            + task\
            + '_'\
            + str(length)\
            + '_'\
            + channels\
            + '_'\
            + str(artifact)

        if erp_degree is not None:
            patient_path += ("_" + str(erp_degree))

        if not os.path.isdir(patient_path):
            print("Configuration supplied was not found in study folder data.")
            print("Failed:", patient_path)
            raise FileNotFoundError
            sys.exit(3)

        patient_paths.append(patient_path)

    if any(
        ["_"+filter_band in fname
            for fname in os.listdir(patient_path)]):
        pass
    else:
        print(
            "No files with " + filter_band + " were found in " + patient_path)
        raise FileNotFoundError
        sys.exit(3)

    # Instantiate a 'Classifier' Object
    myclf = ML.Classifier('spectra')

    # ============== Load All Studies' Data ==============
    for patient_path in patient_paths:
        for fname in os.listdir(patient_path):
            if "_"+filter_band in fname:
                myclf.LoadData(patient_path+"/"+fname)

    specavgObj = SpectralAverage(myclf)
    specavgObj.plot(fig_fname=fig_fname)


if __name__ == '__main__':
    main()
