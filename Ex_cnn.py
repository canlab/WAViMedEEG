import ML
import os
from tqdm import tqdm
import config
import argparse

def main():

    parser = argparse.ArgumentParser(description = "Options for CNN "\
        + "(convoluional neural network) method of ML.Classifier")

    parser.add_argument('data_type',
                        type = str,
                        help = "Input data type: contigs, erps, or spectra")

    parser.add_argument('--studies_folder',
                        dest = 'studies_folder',
                        type = str,
                        default = config.myStudies,
                        help = "(Default: " + config.myStudies + ") Path to"\
                            + "parent folder containing study folders")

    parser.add_argument('--study_name',
                        dest = 'study_name',
                        type = str,
                        default = config.studyDirectory,
                        help = "(Default: " + config.studyDirectory + ") "\
                            + "Study folder containing dataset")

    parser.add_argument('--task',
                        dest = 'task',
                        type = str,
                        default = 'P300',
                        help = "(Default: P300) Four-character task name. "\
                            + "Options: " + str([key for key in config.tasks]))

    parser.add_argument('--length',
                        dest = 'length',
                        type = int,
                        default = 250,
                        help = "(Default: 250) Duration of input data, in"\
                            + "number of samples @ "\
                            + str(config.sample_rate) + " Hz")

    parser.add_argument('--channels',
                        dest = 'channels',
                        type = str,
                        default = '1111111111111111111',
                        help = "(Default: 1111111111111111111) Binary string "\
                            + "specifying which of the "\
                            + "following EEG channels will be included "\
                            + "in analysis: " + str(config.channel_names))

    parser.add_argument('--artifact',
                        dest = 'artifact',
                        type = int,
                        default = 0,
                        help = "(Default: 0) Strictness of artifacting "\
                            + "algorithm to be used: 0=strict, 1=some, 2=raw")

    parser.add_argument('--erp_degree',
                        dest = 'erp_degree',
                        type = int,
                        default = None,
                        help = "Lowest number in .evt files which will "\
                            + "be accepted as an erp event")

    # ============== CNN args ==============

    parser.add_argument('--epochs',
                        dest = 'epochs',
                        type = int,
                        default = 100,
                        help = "(Default: 100) Number of training "\
                            + " iterations to be run")

    parser.add_argument('--normalize',
                        dest = 'normalize',
                        type = str,
                        default = None,
                        help = "(Default: None) Which normalization technique "\
                            + "to use. One of "\
                            + "the following: standard, minmax, None")

    parser.add_argument('--plot_ROC',
                        dest = 'plot_ROC',
                        type = bool,
                        default = False,
                        help = "(Default: False) Plot sensitivity-specificity "\
                            + "curve on validation dataset")

    parser.add_argument('--tt_split',
                        dest = 'tt_split',
                        type = float,
                        default = 0.33,
                        help = "(Default: 0.33) Ratio of test samples "\
                            + "to train samples")

    parser.add_argument('--learning_rate',
                        dest = 'learning_rate',
                        type = float,
                        default = 0.01,
                        help = "(Default: 0.01) CNN step size")

    parser.add_argument('--lr_decay',
                        dest = 'lr_decay',
                        type = bool,
                        default = False,
                        help = "(Default: False) Whether learning rate should "\
                            + "decay adhering to a 0.96 decay rate schedule")

    # save the variables in 'args'
    args = parser.parse_args()

    data_type = args.data_type
    studies_folder = args.studies_folder
    study_name = args.study_name
    task = args.task
    length = args.length
    channels = args.channels
    art_degree = args.artifact
    erp_degree = args.erp_degree
    epochs = args.epochs
    normalize = args.normalize
    plot_ROC = args.plot_ROC
    tt_split = args.tt_split
    learning_rate = args.learning_rate
    lr_decay = args.lr_decay

    if data_type not in ["erps", "spectra", "contigs"]:
        print("Invalid entry for data_type. "\
            + "Must be one of ['erps', 'contigs', 'spectra']")
        raise ValueError
        sys.exit(1)

    if not os.path.isdir(studies_folder):
        print("Invalid entry for studies_folder, "\
            + "path does not exist as directory.")
        raise FileNotFoundError
        sys.exit(1)

    if not os.path.isdir(os.path.join(studies_folder, study_name)):
        print("Invalid entry for study_name, "\
            + "path does not exist as directory.")
        raise FileNotFoundError
        sys.exit(1)

    if task not in config.tasks:
        print("Invalid entry for task, "\
            + "not accepted as regular task name in config.")
        raise ValueError
        sys.exit(1)

    try:
        if length <=0 or length > 10000:
            print("Invalid entry for length, must be between 0 and 10000.")
            raise ValueError
            sys.exit(1)
    except:
        print("Invalid entry for length, "\
            + "must be integer value between 0 and 10000.")
        raise ValueError
        sys.exit(1)

    try:
        int(channels)
    except:
        print("Invalid entry for channels. Must be 19-char long string of"\
            + "1s and 0s")
        raise ValueError
        sys.exit(1)

    if len(channels) != 19:
        print("Invalid entry for channels. Must be 19-char long string of"\
            + "1s and 0s")
        raise ValueError
        sys.exit(1)

    for char in channels:
        if char != '0' and char != '1':
            print("Invalid entry for channels. Must be 19-char long string of"\
                + "1s and 0s")
            raise ValueError
            sys.exit(1)

    try:
        if epochs <=0 or epochs > 10000:
            print("Invalid entry for epochs, must be between 0 and 10000.")
            sys.exit(1)
            raise ValueError
            sys.exit(1)
    except:
        print("Invalid entry for epochs, "\
            + "must be integer value between 0 and 10000.")
        raise ValueError
        sys.exit(1)

    if normalize not in ["standard", "minmax", None]:
        print("Invalid entry for normalize. "\
            + "Must be one of ['standard', 'minmax', 'None'].")
        raise ValueError
        sys.exit(1)

    if tt_split < 0.1 or tt_split > 0.9:
        print("Invalid entry for tt_split. Must be float between "\
            + "0.1 and 0.9.")
        raise ValueError
        sys.exit(1)

    if learning_rate < 0.00001 or learning_rate > 0.99999:
        print("Invalid entry for learning_rate. Must be float between "\
            + "0.00001 and 0.99999.")
        raise ValueError
        sy.exit(1)

    if art_degree not in [0, 1, 2]:
        print("Invalid entry for artifact. Must be int between 0 and 2.")
        raise ValueError
        sys.exit(1)

    if erp_degree not in [1, 2, None]:
        print("Invalid entry for erp_degree. Must be int between 1 and 2.")
        raise ValueError
        sys.exit(1)

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
        + str(art_degree)

    if erp_degree is not None:
        patient_path += ("_" + str(erp_degree))

    if not os.path.isdir(patient_path):
        print("Configuration supplied was not found in study folder data.")
        print("Failed:", patient_path)
        raise FileNotFoundError
        sys.exit(1)

    # Instantiate a 'Classifier' Object
    myclf = ML.Classifier(data_type)

    # ============== Load Patient (Condition-Positive) Data ==============

    for fname in os.listdir(patient_path):
        if fname[:config.participantNumLen] not in config.excludeSubs:
            myclf.LoadData(patient_path+"/"+fname)

    # ============== Load Control (Condition-Negative) Data ==============
    # the dataset will automatically add healthy control data found in the reference folders

    myclf.Balance(studies_folder)

    myclf.CNN(
        normalize=normalize,
        learning_rate=learning_rate,
        lr_decay=lr_decay,
        epochs=epochs,
        plot_ROC=plot_ROC,
        tt_split=tt_split)

if __name__ == '__main__':
    main()
