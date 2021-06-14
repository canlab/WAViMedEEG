import sys
sys.path.append('..')
from src import ML
from src import config
from src.Standard import SpectralAverage
import os
from tqdm import tqdm
import argparse
from datetime import datetime


def main():

    parser = argparse.ArgumentParser(
        description="Options for Linear Regression "
        + "method of ML.Classifier")

    parser.add_argument('data_type',
                        type=str,
                        help="Input data type: contigs, erps, or spectra")

    parser.add_argument('--studies_folder',
                        dest='studies_folder',
                        type=str,
                        default=config.my_studies,
                        help="(Default: " + config.my_studies + ") Path to "
                        + "parent folder containing study folders")

    parser.add_argument('--study_names',
                        dest='study_names',
                        nargs='+',
                        default=config.study_directory,
                        help="(Default: " + config.study_directory + ") "
                        + "Study folder containing dataset")

    parser.add_argument('--balance',
                        dest='balance',
                        type=bool,
                        default=False,
                        help="(Default: False) If True, then will pop data "
                        + "from the larger class datasets until balanced.")

    parser.add_argument('--task',
                        dest='task',
                        type=str,
                        default='P300',
                        help="(Default: P300) Four-character task name. "
                        + "Options: " + str([key for key in config.tasks]))

    parser.add_argument('--length',
                        dest='length',
                        type=int,
                        default=250,
                        help="(Default: 250) Duration of input data, in "
                        + "number of samples @ "
                        + str(config.sample_rate) + " Hz")

    parser.add_argument('--channels',
                        dest='channels',
                        type=str,
                        default='1111111111111111111',
                        help="(Default: 1111111111111111111) Binary string "
                        + "specifying which of the "
                        + "following EEG channels will be included "
                        + "in analysis: " + str(config.channel_names))

    parser.add_argument('--artifact',
                        dest='artifact',
                        type=str,
                        default=''.join(map(str, config.custom_art_map)),
                        help="(Default: (custom) "
                        + ''.join(map(str, config.custom_art_map))
                        + ") Strictness of artifacting "
                        + "algorithm to be used: 0=strict, 1=some, 2=raw")

    parser.add_argument('--erp_degree',
                        dest='erp_degree',
                        type=int,
                        default=None,
                        help="(Default: None) If not None, lowest number in "
                        + ".evt files which will be accepted as an erp event. "
                        + "Only contigs falling immediately after erp event, "
                        + "i.e. evoked responses, are handled.")

    parser.add_argument('--filter_band',
                        dest='filter_band',
                        type=str,
                        default='nofilter',
                        help="(Default: nofilter) Bandfilter to be used in "
                        + "analysis steps, such "
                        + "as: 'noalpha', 'delta', or 'nofilter'")

    # ============== CNN args ==============

    parser.add_argument('--normalize',
                        dest='normalize',
                        type=str,
                        default=None,
                        help="(Default: None) Which normalization technique "
                        + "to use. One of "
                        + "the following: standard, minmax, None")

    parser.add_argument('--sample_weight',
                        dest='sample_weight',
                        type=bool,
                        default=False,
                        help="(Default: None) If 'auto', uses auto sample "
                        + "weighting to try to resolve class imbalances")

    parser.add_argument('--plot_data',
                        dest='plot_data',
                        type=bool,
                        default=False,
                        help="(Default: False) If True, plots coefficients "
                        + "of linear regression model.")

    parser.add_argument('--plot_ROC',
                        dest='plot_ROC',
                        type=bool,
                        default=False,
                        help="(Default: False) Plot sensitivity-specificity "
                        + "curve on validation dataset")

    parser.add_argument('--plot_conf',
                        dest='plot_conf',
                        type=bool,
                        default=False,
                        help="(Default: False) Plot confusion matrix "
                        + "on validation dataset")

    parser.add_argument('--plot_spectra',
                        dest='plot_spectra',
                        type=bool,
                        default=False,
                        help="(Default: False) Plot spectra by group for "
                        + "training data")

    parser.add_argument('--tt_split',
                        dest='tt_split',
                        type=float,
                        default=0.33,
                        help="(Default: 0.33) Ratio of test samples "
                        + "to train samples. Note: not applicable if using "
                        + "k_folds.")

    parser.add_argument('--k_folds',
                        dest='k_folds',
                        type=int,
                        default=1,
                        help="(Default: 1) If you want to perform "
                        + "cross evaluation, set equal to number of k-folds.")

    parser.add_argument('--repetitions',
                        dest='repetitions',
                        type=int,
                        default=1,
                        help="(Default: 1) Unlike k-fold, trains the "
                        + "model n times without mixing around subjects. "
                        + "Can still be used within each k-fold.")

    parser.add_argument('--regularizer',
                        dest='regularizer',
                        type=str,
                        default=None,
                        help="(Default: l1_l2) Regularizer to be used in dense "
                        + "layers. One of: ['l1', 'l2', 'l1_l2']")

    parser.add_argument('--regularizer_param',
                        dest='regularizer_param',
                        type=float,
                        default=0.01,
                        help="(Default: 0.01) Regularization parameter ")

    # save the variables in 'args'
    args = parser.parse_args()

    data_type = args.data_type
    studies_folder = args.studies_folder
    study_names = args.study_names
    task = args.task
    length = args.length
    channels = args.channels
    artifact = args.artifact
    erp_degree = args.erp_degree
    filter_band = args.filter_band
    normalize = args.normalize
    sample_weight = args.sample_weight
    plot_data = args.plot_data
    plot_ROC = args.plot_ROC
    plot_conf = args.plot_conf
    plot_spectra = args.plot_spectra
    tt_split = args.tt_split
    k_folds = args.k_folds
    repetitions = args.repetitions
    regularizer = args.regularizer
    regularizer_param = args.regularizer_param
    balance = args.balance

    # ERROR HANDLING
    if data_type not in ["erps", "spectra", "contigs"]:
        print(
            "Invalid entry for data_type. "
            + "Must be one of ['erps', 'contigs', 'spectra']")
        raise ValueError
        sys.exit(3)

    if not os.path.isdir(studies_folder):
        print(
            "Invalid entry for studies_folder, "
            + "path does not exist as directory.")
        raise FileNotFoundError
        sys.exit(3)

    for study_name in study_names:
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

    if type(length) is int is False:
        print("Length must be an integer (in Hz).")
        raise ValueError
        sys.exit(3)

    try:
        if (length <= 0) or (length > 10000):
            print("Invalid entry for length, must be between 0 and 10000.")
            raise ValueError
            sys.exit(3)
    except TypeError:
        print(
            "Invalid entry for length, "
            + "must be integer value between 0 and 10000.")
        raise ValueError
        sys.exit(3)

    try:
        str(channels)
    except ValueError:
        print(
            "Invalid entry for channels. Must be 19-char long string of "
            + "1s and 0s")
        raise ValueError
        sys.exit(3)

    if len(channels) != 19:
        print(
            "Invalid entry for channels. Must be 19-char long string of "
            + "1s and 0s")
        raise ValueError
        sys.exit(3)

    for char in channels:
        if char != '0' and char != '1':
            print(
                "Invalid entry for channels. Must be 19-char long string of "
                + "1s and 0s")
            raise ValueError
            sys.exit(3)

    if normalize not in ["standard", "minmax", None]:
        print(
            "Invalid entry for normalize. "
            + "Must be one of ['standard', 'minmax', 'None'].")
        raise ValueError
        sys.exit(3)

    if tt_split < 0 or tt_split > 0.999:
        print(
            "Invalid entry for tt_split. Must be float between "
            + "0 and 0.999.")
        raise ValueError
        sys.exit(3)

    try:
        if len(str(artifact)) == 19:
            for char in artifact:
                if int(char) < 0 or int(char) > 2:
                    raise ValueError

        elif artifact in ["0", "1", "2"]:
            artifact = int(artifact)

        else:
            raise ValueError

    except ValueError:
        print(
            "Invalid entry for artifact. Must be str with length 19, "
            + "or int between 0 and 2.")
        raise ValueError
        sys.exit(3)

    if erp_degree not in [1, 2, None]:
        print("Invalid entry for erp_degree. Must be None, 1, or 2.")
        raise ValueError
        sys.exit(3)

    if k_folds <= 0:
        print("Invalid entry for k_folds. Must be int 1 or greater.")
        raise ValueError
        sys.exit(3)

    if regularizer is not None:
        if regularizer not in ['l1', 'l2', 'l1_l2']:
            print("Invalid entry for regularizer. Must be l1, l2, or l1_l2.")
            raise ValueError
            sys.exit(3)

    if (regularizer_param <= 0) or (regularizer_param >= 1):
        print(
            "Invalid entry for regularizer param. Must be float between "
            + "0 and 1.")
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

    patient_paths = []
    # patient_path points to our 'condition-positive' dataset
    # ex. patient_path =
    # "/wavi/EEGstudies/CANlab/spectra/P300_250_1111111111111111111_0_1"
    for study_name in study_names:

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

        patient_paths.append(patient_path)

    # Instantiate a 'Classifier' Object
    myclf = ML.Classifier(data_type)

    # ============== Load All Studies' Data ==============
    patient_paths = []
    # patient_path points to our 'condition-positive' dataset
    # ex. patient_path =
    # "/wavi/EEGstudies/CANlab/spectra/P300_250_1111111111111111111_0_1"
    for study_name in study_names:

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

        patient_paths.append(patient_path)

    for patient_path in patient_paths:
        for fname in sorted(os.listdir(patient_path)):
            if "_"+filter_band in fname:
                myclf.LoadData(patient_path+"/"+fname)

    # ============== Balance Class Data Sizes ==============
    # pops data off from the larger class until class sizes are equal
    # found in the reference folders
    if balance is True:
        myclf.Balance()

    if k_folds == 1:
        myclf.Prepare(tt_split=tt_split, normalize=normalize)

        for i in range(repetitions):

            model, y_pred, y_labels = myclf.LinearRegression(
                plot_data=plot_data,
                plot_ROC=plot_ROC,
                plot_conf=plot_conf)

            if data_type == 'spectra':
                if plot_spectra is True:
                    specavgObj = SpectralAverage(myclf)
                    specavgObj.plot(
                        fig_fname=myclf.checkpoint_dir
                        + "/specavg_"
                        + os.path.basename(myclf.trial_name)
                        + "_train_"
                        + str(datetime.now().strftime("%H-%M-%S")))

    if k_folds > 1:
        myclf.KfoldCrossVal(
            myclf.LDA,
            normalize=normalize,
            regularizer=regularizer,
            regularizer_param=regularizer_param,
            repetitions=repetitions,
            learning_rate=learning_rate,
            lr_decay=lr_decay,
            plot_ROC=plot_ROC,
            plot_conf=plot_conf,
            plot_3d_preds=plot_3d_preds,
            k=k_folds,
            plot_spec_avgs=plot_spectra,
            sample_weight=sample_weight)


if __name__ == '__main__':
    main()
