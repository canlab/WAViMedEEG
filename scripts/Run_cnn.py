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

    parser.add_argument('--epochs',
                        dest='epochs',
                        type=int,
                        default=100,
                        help="(Default: 100) Number of training "
                        + " iterations to be run")

    parser.add_argument('--normalize',
                        dest='normalize',
                        type=str,
                        default=None,
                        help="(Default: None) Which normalization technique "
                        + "to use. One of "
                        + "the following: standard, minmax, None")

    # parser.add_argument('--bias',
    #                     dest='bias',
    #                     type=str,
    #                     default=None,
    #                     help="(Default: None) If 'auto', uses bias "
    #                     + "initializer to try to resolve class imbalances")

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

    parser.add_argument('--tt_split',
                        dest='tt_split',
                        type=float,
                        default=0.33,
                        help="(Default: 0.33) Ratio of test samples "
                        + "to train samples. Note: not applicable if using "
                        + "k_folds.")

    parser.add_argument('--learning_rate',
                        dest='learning_rate',
                        type=float,
                        default=0.001,
                        help="(Default: 0.001) CNN step size")

    parser.add_argument('--lr_decay',
                        dest='lr_decay',
                        type=bool,
                        default=False,
                        help="(Default: False) Whether learning rate should "
                        + "decay adhering to a 0.96 decay rate schedule")

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

    parser.add_argument('--depth',
                        dest='depth',
                        type=int,
                        default=5,
                        help="(Default: 5) Number of sets of {convoutional, "
                        + "pooling, batch norm} to include in the model.")

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

    parser.add_argument('--focal_gamma',
                        dest='focal_loss_gamma',
                        type=float,
                        default=0,
                        help="(Default: 0) At zero, focal loss is exactly "
                        + "cross-entropy. At higher values, easier data "
                        + "contributes more weakly to loss function, and "
                        + "difficult classifications are stronger.")

    parser.add_argument('--dropout',
                        dest='dropout',
                        type=float,
                        default=None,
                        help="(Default: None) Dropout rate ")

    parser.add_argument('--hypertune',
                        dest='hypertune',
                        type=bool,
                        default=False,
                        help="(Default: False) If True, all args will be "
                        + "tuned in keras tuner and saved to logs/fit.")

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
    epochs = args.epochs
    normalize = args.normalize
    # bias = args.bias
    plot_ROC = args.plot_ROC
    plot_conf = args.plot_conf
    plot_3d_preds = args.plot_3d_preds
    plot_spectra = args.plot_spectra
    tt_split = args.tt_split
    learning_rate = args.learning_rate
    lr_decay = args.lr_decay
    k_folds = args.k_folds
    repetitions = args.repetitions
    depth = args.depth
    regularizer = args.regularizer
    regularizer_param = args.regularizer_param
    focal_loss_gamma = args.focal_loss_gamma
    dropout = args.dropout
    hypertune = args.hypertune
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

    try:
        if (epochs <= 0) or (epochs > 10000):
            print("Invalid entry for epochs, must be between 0 and 10000.")
            sys.exit(3)
            raise ValueError
            sys.exit(3)
    except TypeError:
        print(
            "Invalid entry for epochs, "
            + "must be integer value between 0 and 10000.")
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

    if learning_rate < 0.00001 or learning_rate > 0.99999:
        print(
            "Invalid entry for learning_rate. Must be float between "
            + "0.00001 and 0.99999.")
        raise ValueError
        sy.exit(3)

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

    if (focal_loss_gamma < 0):
        print("Invalid entry for focal gamma. Must be float >= 0.")
        raise ValueError
        sys.exit(3)

    if dropout is not None:
        if (dropout <= 0) or (dropout >= 1):
            print(
                "Invalid entry for dropout. Must be float between 0 and 1 "
                + "or None.")
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
            if hypertune is False:
                model, _, _, = myclf.CNN(
                    learning_rate=learning_rate,
                    lr_decay=lr_decay,
                    epochs=epochs,
                    plot_ROC=plot_ROC,
                    plot_conf=plot_conf,
                    plot_3d_preds=plot_3d_preds,
                    depth=depth,
                    regularizer=regularizer,
                    regularizer_param=regularizer_param,
                    focal_loss_gamma=focal_loss_gamma,
                    # initial_bias=bias,
                    dropout=dropout)

                if data_type == 'spectra':
                    if plot_spectra is True:
                        specavgObj = SpectralAverage(myclf)
                        specavgObj.plot(
                            fig_fname=myclf.checkpoint_dir
                            + "/specavg_"
                            + os.path.basename(myclf.trial_name)
                            + "_train_"
                            + str(datetime.now().strftime("%H-%M-%S")))

            else:
                import tensorflow as tf
                from kerastuner.tuners import Hyperband

                tuner = Hyperband(
                    myclf.hypertune_CNN,
                    objective='val_accuracy',
                    max_epochs=100,
                    directory='logs/fit/',
                    project_name='1second-spectra')

                tuner.search_space_summary()

                stop_early = tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=10)

                tuner.search(
                    myclf.train_dataset,
                    myclf.train_labels,
                    epochs=100,
                    validation_data=(myclf.test_dataset, myclf.test_labels),
                    callbacks=[stop_early])

                models = tuner.get_best_models(num_models=10)

                for i, model in enumerate(models):
                    try:
                        os.mkdir('logs/fit/'+str(i))

                        model.save('logs/fit/'+str(i)+"/my_model")

                    except:
                        print("Can't save models. :(")

                tuner.results_summary()

    if k_folds > 1:
        myclf.KfoldCrossVal(
            myclf.CNN,
            normalize=normalize,
            regularizer=regularizer,
            regularizer_param=regularizer_param,
            focal_loss_gamma=focal_loss_gamma,
            repetitions=repetitions,
            dropout=dropout,
            learning_rate=learning_rate,
            lr_decay=lr_decay,
            epochs=epochs,
            plot_ROC=plot_ROC,
            plot_conf=plot_conf,
            plot_3d_preds=plot_3d_preds,
            k=k_folds,
            plot_spec_avgs=plot_spectra)


if __name__ == '__main__':
    main()
