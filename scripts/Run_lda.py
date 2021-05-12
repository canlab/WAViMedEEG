import sys
sys.path.append('..')
from src import ML
from src import config
import os
from tqdm import tqdm
import argParser


def main():

    args = argParser.main([
        'studies_folder',
        'study_names',
        'task',
        'data_type',
        'length',
        'channels',
        'artifact',
        'erp_degree',
        'filter_band',
        'balance',
        'normalize',
        'tt_split',
        'k_folds'
        # 'plot'
    ])

    data_type = args.data_type
    studies_folder = args.studies_folder
    study_name = args.study_name
    task = args.task
    length = args.length
    channels = args.channels
    artifact = args.artifact
    erp_degree = args.erp_degree
    filter_band = args.filter_band
    balance = args.balance
    normalize = args.normalize
    tt_split = args.tt_split
    k_folds = args.k_folds
    # plot = args.plot

    # patient_path points to our 'condition-positive' dataset
    # ex. patient_path =
    # "/wavi/EEGstudies/CANlab/spectra/P300_250_1111111111111111111_0_1"
    patient_path = studies_folder\
        + '/'\
        + study_name\
        + '/'\
        + data_type\
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

    for folder in balance:
        if not os.path.isdir(studies_folder + "/" + folder):
            print(
                "Invalid path in balance. ",
                studies_folder + "/" + folder + "  does not exist.")
            raise FileNotFoundError
            sys.exit(3)

        if not os.path.isdir(patient_path.replace(study_name, folder)):
            print(
                "Invalid path in balance. "
                + patient_path.replace(study_name, folder)
                + "does not exist.")
            raise FileNotFoundError
            sys.exit(3)

    # Instantiate a 'Classifier' Object
    myclf = ML.Classifier(data_type)

    # ============== Load Patient (Condition-Positive) Data ==============

    for fname in os.listdir(patient_path):
        if fname[:config.participantNumLen] not in config.exclude_subs:
            if "_"+filter_band in fname:
                myclf.LoadData(patient_path+"/"+fname)

    # ============== Load Control (Condition-Negative) Data ==============
    # the dataset will automatically add healthy control data
    # found in the reference folders
    myclf.Balance(studies_folder, filter_band=filter_band, ref_folders=balance)

    if k_folds == 1:
        myclf.Prepare(tt_split=tt_split)

        myclf.LDA(
            plot_data=plot)

    if k_folds > 1:
        myclf.KfoldCrossVal(
            myclf.LDA,
            k=k_folds,
            plot_data=plot)


if __name__ == '__main__':
    main()
