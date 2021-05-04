import sys
sys.path.append('..')
from src import ML
from src import config
from src.Standard import SpectralAverage
import os
from tqdm import tqdm
import argparse
from datetime import datetime
import numpy as np


def main():

    parser = argparse.ArgumentParser(
        description="Options for CNN "
        + "(convoluional neural network) method of ML.Classifier")

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

    parser.add_argument('--log_dir',
                        dest='log_dir',
                        type=str,
                        default="logs/fit",
                        help="(Default: logs/fit) Parent directory for "
                        + "checkpoints.")

    parser.add_argument('--checkpoint_dir',
                        dest='checkpoint_dir',
                        type=str,
                        default=None,
                        help="(Default: None) Checkpoint directory (most "
                        + "likely found in logs/fit) containing saved model.")

    parser.add_argument('--pred_level',
                        dest='pred_level',
                        type=str,
                        default='all',
                        help="(Default: 'all') If 'all', returns a prediction "
                        + "for every individual piece of data. If 'subject', "
                        + "then returns 1 averaged prediction per subject.")

    parser.add_argument('--combine',
                        dest='combine',
                        type=bool,
                        default=False,
                        help="(Default: False) If True, will load all of the "
                        + "data at one time for a combined set of predictions.")

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

    parser.add_argument('--plot_hist',
                        dest='plot_hist',
                        type=bool,
                        default=False,
                        help="(Default: False) Plot histogram of predictions.")

    parser.add_argument('--plot_conf',
                        dest='plot_conf',
                        type=bool,
                        default=False,
                        help="(Default: False) Plot confusion matrix "
                        + "on validation dataset")

    parser.add_argument('--plot_3d_preds',
                        dest='plot_3d_preds',
                        type=bool,
                        default=False,
                        help="(Default: False) Plot 3-dimensional scatter "
                        + "plot of validation dataset predictions")

    parser.add_argument('--plot_spectra',
                        dest='plot_spectra',
                        type=bool,
                        default=False,
                        help="(Default: False) Plot spectra by group for "
                        + "training data")

    # save the variables in 'args'
    args = parser.parse_args()

    data_type = args.data_type
    studies_folder = args.studies_folder
    study_names = args.study_names
    log_dir = args.log_dir
    checkpoint_dir = args.checkpoint_dir
    task = args.task
    length = args.length
    channels = args.channels
    artifact = args.artifact
    erp_degree = args.erp_degree
    filter_band = args.filter_band
    normalize = args.normalize
    plot_spectra = args.plot_spectra
    plot_hist = args.plot_hist
    plot_conf = args.plot_conf
    plot_3d_preds = args.plot_3d_preds
    pred_level = args.pred_level
    combine = args.combine

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

    if checkpoint_dir is not None:
        if not os.path.isdir(checkpoint_dir):
            if not os.path.isdir(log_dir+checkpoint_dir):
                print(
                    "Invalid entry for checkpoint directory, "
                    + "path does not exist as directory.")
                raise FileNotFoundError
                sys.exit(3)
            else:
                checkpoint_dir = log_dir+checkpoint_dir

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

    if pred_level not in ["all", "subject"]:
        print(
            "Invalid entry for pred_level. Must be either "
            + "'all' or 'subject'")
        raise ValueError
        sys.exit(3)

    # patient_path points to our 'condition-positive' dataset
    # ex. patient_path =
    # "/wavi/EEGstudies/CANlab/spectra/P300_250_1111111111111111111_0_1"
    if checkpoint_dir is None:
        checkpoint_dirs = [
            log_dir + folder
            for folder in os.listdir(log_dir)
            if "_"+data_type in folder]
        checkpoint_dirs.sort()
    else:
        checkpoint_dirs = [checkpoint_dir]

    patient_paths = []
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

    if combine is True:
        # Instantiate a 'Classifier' object
        myclf = ML.Classifier(data_type)

    # ============== Load All Studies' Data ==============
    for study_name, patient_path in zip(study_names, patient_paths):

        if combine is not True:
            # Instantiate a 'Classifier' object
            myclf = ML.Classifier(data_type)

        for fname in os.listdir(patient_path):
            if "_"+filter_band in fname:
                myclf.LoadData(patient_path+"/"+fname)

        if combine is not True:

            for checkpoint_dir in checkpoint_dirs:

                label_names=checkpoint_dir.split('_')[7:]
                label_values=[]
                for group in label_names:
                    for key, value in config.group_names.items():
                        if group == value:
                            label_values.append(key)

                myclf.Prepare(
                    tt_split=1,
                    labels=label_values,
                    normalize=normalize)

                if data_type == 'spectra':
                    if plot_spectra is True:
                        specavgObj = SpectralAverage(myclf)
                        specavgObj.plot(
                            fig_fname=checkpoint_dir+"/"
                            + study_name
                            + "_true_"
                            + str(datetime.now().strftime("%H-%M-%S")))

                y_preds = myclf.eval_saved_CNN(
                    checkpoint_dir,
                    plot_hist=plot_hist,
                    plot_conf=plot_conf,
                    plot_3d_preds=plot_3d_preds,
                    fname=study_name,
                    pred_level=pred_level)

                for i, (pred, inputObj) in enumerate(
                    zip(np.rint(y_preds), myclf.data)):

                    inputObj.group = myclf.groups[int(np.argmax(pred))]

                if data_type == 'spectra':
                    if plot_spectra is True:
                        specavgObj = SpectralAverage(myclf)
                        specavgObj.plot(
                            fig_fname=checkpoint_dir+"/"
                            + study_name
                            + "_pred_"
                            + str(datetime.now().strftime("%H-%M-%S")))

    if combine is True:
        for checkpoint_dir in checkpoint_dirs:

            label_names=checkpoint_dir.split('_')[7:]
            label_values=[]
            for group in label_names:
                for key, value in config.group_names.items():
                    if group == value:
                        label_values.append(key)

            myclf.Prepare(
                tt_split=1,
                labels=label_values,
                normalize=normalize,
                eval=True)

            if data_type == 'spectra':
                if plot_spectra is True:
                    specavgObj = SpectralAverage(myclf)
                    specavgObj.plot(
                        fig_fname=checkpoint_dir+"/"
                        + study_name
                        + "_true_"
                        + str(datetime.now().strftime("%H-%M-%S")))

            y_preds = myclf.eval_saved_CNN(
                checkpoint_dir,
                plot_hist=plot_hist,
                plot_conf=plot_conf,
                plot_3d_preds=plot_3d_preds,
                fname=study_name,
                pred_level=pred_level,
                save_results=True)

            # TODO:
            # broken, can't change label here for combination runs
            # for i, (pred, inputObj) in enumerate(
            #     zip(np.rint(y_preds), myclf.data)):
            #
            #     inputObj.group = myclf.groups[int(np.argmax(pred))]

            # if data_type == 'spectra':
            #     if plot_spectra is True:
            #         specavgObj = SpectralAverage(myclf)
            #         specavgObj.plot(
            #             fig_fname=checkpoint_dir+"/"
            #             + study_name
            #             + "_pred_"
            #             + str(datetime.now().strftime("%H-%M-%S")))

if __name__ == '__main__':
    main()
