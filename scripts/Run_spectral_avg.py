import sys
sys.path.append('..')
import src.ML
import src.config
from src.Standard import SpectralAverage
import os
from tqdm import tqdm
import config


def main():

    parser = argparse.ArgumentParser(
        description="Options for Spectral Average Plotting ")

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

    parser.add_argument('--fig_fname',
                        dest='fig_fname',
                        type=str,
                        default=None,
                        help="If supplied, will save figure instead of "
                        + "showing it, and save to file at this path.")

    # save the variables in 'args'
    args = parser.parse_args()

    studies_folder = args.studies_folder
    study_names = args.study_names
    task = args.task
    length = args.length
    channels = args.channels
    artifact = args.artifact
    erp_degree = args.erp_degree
    filter_band = args.filter_band
    fig_fname = args.fig_fname

    # ERROR HANDLING
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

    if not isinstance(length, int):
        print("Length must be an integer.")
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
