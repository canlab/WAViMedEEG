import sys
sys.path.append('..')
import os
import shutil
import argparse
from src import config

# supply with a tuple of arguments needed for the script to run
# unknown number of arguments
def main(query):
    parser = argparse.ArgumentParser(
        description="Options for scripting with the WAViMedEEG toolbox")

    # PATHS Setup
    # ==========================
    if 'studies_folder' in query:
        parser.add_argument('--studies_folder',
                            dest='studies_folder',
                            type=str,
                            default=config.my_studies,
                            help="(Default: " + config.my_studies + ") Path to "
                            + "parent folder containing study folders")

    if 'study_names' in query:
        parser.add_argument('--study_names',
                            dest='study_names',
                            nargs='+',
                            default=config.study_directory,
                            help="(Default: " + config.study_directory + ") "
                            + "Study folder(s) containing dataset")

    if 'reference_studies' in query:
        parser.add_argument('--reference_studies',
                            dest='reference_studies',
                            type=str,
                            default=None,
                            nargs='+',
                            help='(Default: None) Study folder(s) containing '
                            + 'reference dataset (for z-score analyses)')

    if 'task' in query:
        parser.add_argument('--task',
                            dest='task',
                            type=str,
                            default='P300',
                            help="(Default: P300) Four-character task name. "
                            + "Options: " + str([key for key in config.tasks]))

    # PATHS Setup - METADATA
    # ==========================
    if 'group_nums' in query:
        parser.add_argument('--group_nums',
                            dest='group_nums',
                            type=int,
                            default="[1 for study in study_names]",
                            nargs='+',
                            help='(Default: 1) Group number to be '
                            + 'assigned to dataset. If using multiple '
                            + 'study_names, match as list.')

    if 'data_type' in query:
        parser.add_argument('--data_type',
                            dest='data_type',
                            type=str,
                            default='erps',
                            help="(Default: erps) Input data type, one of "
                            + "{contigs, erps, spectra, coherences}")

    if 'length' in query:
        parser.add_argument('--length',
                            dest='length',
                            type=int,
                            default=250,
                            help="(Default: 250) Duration of input data, in "
                            + "number of samples @ "
                            + str(config.sample_rate) + " Hz")

    if 'channels' in query:
        parser.add_argument('--channels',
                            dest='channels',
                            type=str,
                            default='1111111111111111111',
                            help="(Default: 1111111111111111111) Binary "
                            + "string specifying which of the "
                            + "following EEG channels will be included "
                            + "in analysis: " + str(config.channel_names))

    if 'artifact' in query:
        parser.add_argument('--artifact',
                            dest='artifact',
                            type=str,
                            default=''.join(map(str, config.custom_art_map)),
                            help="(Default: (custom) "
                            + ''.join(map(str, config.custom_art_map))
                            + ") Strictness of artifacting "
                            + "algorithm to be used: 0=strict, 1=some, 2=raw")

    if 'erp_degree' in query:
        parser.add_argument('--erp_degree',
                            dest='erp_degree',
                            type=int,
                            default=None,
                            help="(Default: None) If not None, lowest number "
                            + "in .evt files which will be accepted as an erp "
                            + "event. Only contigs falling immediately after "
                            + "erp event, i.e. evoked responses, are handled.")

    if 'filter_band' in query:
        parser.add_argument('--filter_band',
                            dest='filter_band',
                            type=str,
                            default='nofilter',
                            help="(Default: nofilter) Bandfilter to be used in "
                            + "analysis steps, such "
                            + "as: 'noalpha', 'delta', or 'nofilter'")

    # DATA PREP
    # ==========================
    if 'limited_subjects' in query:
        parser.add_argument('--limited_subjects',
                            dest='limited_subjects',
                            nargs='+',
                            default=None,
                            help="(Default: None" + ") Used to only provide "
                            + "analysis or preprocessing step wiht a "
                            + "limited subset of the participants in it.")

    if 'balance' in query:
        parser.add_argument('--balance',
                            dest='balance',
                            type=bool,
                            default=False,
                            help="(Default: False) If True, then will pop data "
                            + "from the larger class datasets until balanced.")

    if 'normalize' in query:
        parser.add_argument('--normalize',
                            dest='normalize',
                            type=str,
                            default=None,
                            help="(Default: None) Which normalization "
                            + "technique to use. One of "
                            + "{standard, minmax, None}")

    if 'filter_type' in query:
        parser.add_argument('--filter_type',
                            dest='filter_type',
                            type=str,
                            default="bandpass",
                            help="(Default: bandpass) Which band filter method "
                            + "should be applied: "
                            + "lowpass, highpass, bandstop, bandpass")

    if 'frequency_band' in query:
        parser.add_argument('--frequency_band',
                            dest='frequency_band',
                            type=str,
                            default="delta",
                            help="(Default: delta) "
                            + "Frequency band used for band ranges: "
                            + str([val for val in config.frequency_bands]))

    if 'gen_spectra' in query:
        parser.add_argument('--gen_spectra',
                            dest='gen_spectra',
                            type=bool,
                            default=True,
                            help="(Default: True) Whether spectra should "
                            + "automatically be generated and written to file "
                            + "after making contigs")


    # Machine Learning
    # ==========================
    if 'epochs' in query:
        parser.add_argument('--epochs',
                            dest='epochs',
                            type=int,
                            default=100,
                            help="(Default: 100) Number of training "
                            + " iterations to be run")

    if 'sample_weight' in query:
        parser.add_argument('--sample_weight',
                            dest='sample_weight',
                            type=bool,
                            default=True,
                            help="(Default: True) If True, uses auto sample "
                            + "weighting to try to resolve class imbalances")

    if 'tt_split' in query:
        parser.add_argument('--tt_split',
                            dest='tt_split',
                            type=float,
                            default=0.33,
                            help="(Default: 0.33) Ratio of test samples "
                            + "to train samples. Note: not applicable if using "
                            + "k_folds.")

    if 'learning_rate' in query:
        parser.add_argument('--learning_rate',
                            dest='learning_rate',
                            type=float,
                            default=0.0001,
                            help="(Default: 0.0001) CNN step size")

    if 'lr_decay' in query:
        parser.add_argument('--lr_decay',
                            dest='lr_decay',
                            type=bool,
                            default=False,
                            help="(Default: False) Whether learning rate should "
                            + "decay adhering to a 0.96 decay rate schedule")

    if 'k_folds' in query:
        parser.add_argument('--k_folds',
                            dest='k_folds',
                            type=int,
                            default=1,
                            help="(Default: 1) If you want to perform "
                            + "cross evaluation, set equal to number of "
                            + "k-folds.")

    if 'repetitions' in query:
        parser.add_argument('--repetitions',
                            dest='repetitions',
                            type=int,
                            default=1,
                            help="(Default: 1) Unlike k-fold, trains the "
                            + "model n times without mixing around subjects. "
                            + "Can still be used within each k-fold.")

    if 'depth' in query:
        parser.add_argument('--depth',
                            dest='depth',
                            type=int,
                            default=5,
                            help="(Default: 5) Number of sets of "
                            + "{convolutional, pooling, batch norm} to "
                            + "include in the model.")

    if 'regularizer' in query:
        parser.add_argument('--regularizer',
                            dest='regularizer',
                            type=str,
                            default=None,
                            help="(Default: l1_l2) Regularization method to "
                            + "be used. One of: ['l1', 'l2', 'l1_l2']")

    if 'regularizer_param' in query:
        parser.add_argument('--regularizer_param',
                            dest='regularizer_param',
                            type=float,
                            default=0.01,
                            help="(Default: 0.01) Regularization parameter")

    if 'focal_gamma' in query:
        parser.add_argument('--focal_gamma',
                            dest='focal_loss_gamma',
                            type=float,
                            default=0,
                            help="(Default: 0) At zero, focal loss is exactly "
                            + "cross-entropy. At higher values, easier data "
                            + "contributes more weakly to loss function, and "
                            + "difficult classifications are stronger.")

    if 'dropout' in query:
        parser.add_argument('--dropout',
                            dest='dropout',
                            type=float,
                            default=None,
                            help="(Default: None) Dropout rate used after "
                            + "convolutional layers")

    if 'hypertune' in query:
        parser.add_argument('--hypertune',
                            dest='hypertune',
                            type=bool,
                            default=False,
                            help="(Default: False) If True, all args will be "
                            + "tuned in keras tuner and saved to logs/fit.")

    if 'logistic_regression' in query:
        parser.add_argument('--logistic_regression',
                            dest='logistic_regression',
                            type=bool,
                            default=False,
                            help="(Default: False) if True, the model will "
                            + "consist only of logistic regression, and will "
                            + "not be convolutional. Significantly reduced "
                            + "complexity is implied.")

    # Machine Learning - Evaluation
    # ==========================
    if 'log_dirs' in query:
        parser.add_argument('--log_dirs',
                            dest='log_dirs',
                            type=str,
                            nargs='+',
                            default=["logs/fit/"],
                            help="(Default: logs/fit) Parent directory for "
                            + "checkpoints.")

    if 'checkpoint_dirs' in query:
        parser.add_argument('--checkpoint_dirs',
                            dest='checkpoint_dirs',
                            type=str,
                            nargs='+',
                            default=None,
                            help="(Default: None) Checkpoint directory (most "
                            + "likely found in logs/fit) containing saved model.")

    if 'combine' in query:
        parser.add_argument('--combine',
                            dest='combine',
                            type=bool,
                            default=False,
                            help="(Default: False) Combines all study names "
                            + "provided into single test dataset.")

    # if 'pred_level' in query:
    #     parser.add_argument('--pred_level',
    #                         dest='pred_level',
    #                         type=str,
    #                         default='all',
    #                         help="(Default: all) Whether to save predictions "
    #                         + "on saved model for each data array provided "
    #                         + "or summarize as % of subjects' data.")

    if 'fallback' in query:
        parser.add_argument('--fallback',
                            dest='fallback',
                            type=bool,
                            default=False,
                            help="(Default: False) In the event that a subject "
                            + "in the study group does not have contigs "
                            + "generated (i.e. data too noisy), will attempt "
                            + "to grab their data from the next-loosest "
                            + "schema in the given data folder.")

    # Machine Learning - Plots
    # ==========================
    if 'plot_ROC' in query:
        parser.add_argument('--plot_ROC',
                            dest='plot_ROC',
                            type=bool,
                            default=False,
                            help="(Default: False) Plot sensitivity-"
                            + "specificity curve on validation dataset")

    if 'plot_hist' in query:
        parser.add_argument('--plot_hist',
                            dest='plot_hist',
                            type=bool,
                            default=False,
                            help="(Default: False) Plot histogram "
                            + "of model evaluations.")

    if 'plot_conf' in query:
        parser.add_argument('--plot_conf',
                            dest='plot_conf',
                            type=bool,
                            default=False,
                            help="(Default: False) Plot confusion matrix "
                            + "on validation dataset")

    if 'plot_3d_preds' in query:
        parser.add_argument('--plot_3d_preds',
                            dest='plot_3d_preds',
                            type=bool,
                            default=False,
                            help="(Default: False) Plot 3-dimensional scatter "
                            + "plot of validation dataset predictions")

    if 'plot_spectra' in query:
        parser.add_argument('--plot_spectra',
                            dest='plot_spectra',
                            type=bool,
                            default=False,
                            help="(Default: False) Plot spectra by group for "
                            + "training data")

    # SYSTEM SETTINGS
    # ==========================
    if 'force' in query:
        parser.add_argument('--force',
                            dest='force',
                            type=bool,
                            default=False,
                            help='(Default: False) If True, will still run '
                            + 'given that some potential file-overwrite '
                            + 'was printed. Warning: use of this option may '
                            + 'result in the overwriting of files. Proceed '
                            + 'with extreme caution. ')

    if 'use_gpu' in query:
        parser.add_argument('--use_gpu',
                            dest='use_gpu',
                            type=bool,
                            default=False,
                            help="(Default: False) If True, will attempt to "
                            + "replace numpy and scipy functions with cupy "
                            + "- requires NVIDIA GPU attached, and an existing "
                            + "installation of CUDA, as well as cupy.")

    if 'verbosity' in query:
        parser.add_argument('--verbosity',
                            dest='verbosity',
                            type=bool,
                            default=True,
                            help="(Default: True) If True, will attempt to "
                            + "quiet most printed output.")

    # ==========================
    # save the variables in 'args'
    args = parser.parse_args()

    # ERROR HANDLING
    # ==========================
    # studies folder
    if hasattr(args, "studies_folder"):
        if not os.path.isdir(args.studies_folder):
            print(
                "Invalid entry for studies_folder, "
                + "path does not exist as directory.")
            raise FileNotFoundError
            sys.exit(3)

    # study_names
    if hasattr(args, "study_names"):
        for study_name in args.study_names:
            # check folder exists
            if not os.path.isdir(os.path.join(args.studies_folder, study_name)):
                print(
                    "Invalid entry for study_name:", study_name,
                    "does not exist as directory in", args.studies_folder)
                raise FileNotFoundError
                sys.exit(3)

            # check folder is a parent
            if len(os.listdir(
                os.path.join(
                    args.studies_folder,
                    study_name))) == 0:
                print(
                    "No files or folders were found in", study_name,
                    "Are you sure that this is the correct path?")
                raise FileNotFoundError
                sys.exit(3)

            # check folder has 'raw'
            # if os.path.isdir(os.path.join(studies_folder, study_name, 'raw')):
            #     if len(os.listdir(os.path.join(
            #         studies_folder,
            #         study_name, 'raw'))
            #         ) == 0:
            #         os.rmdir(os.path.join(studies_folder, study_name, 'raw'))

            if not os.path.isdir(
                os.path.join(
                    args.studies_folder,
                    study_name,
                    'raw')):
                print(
                    "Warning: the expected 'raw' folder was not found, and so "
                    "will be created automatically for you - including any "
                    "files that were in the directory supplied.")

                os.mkdir(os.path.join(args.studies_folder, study_name, 'raw'))
                for file_folder in [file for file in
                    os.listdir(os.path.join(args.studies_folder, study_name))]:
                    if not os.path.isdir(file_folder) and file_folder != 'raw':
                        shutil.move(
                            os.path.join(
                                args.studies_folder,
                                study_name,
                                file_folder),
                            os.path.join(
                                args.studies_folder,
                                study_name,
                                'raw',
                                file_folder))

    # reference studies
    if hasattr(args, "reference_studies"):
        for study_name in args.reference_studies:
            # check folder exists
            if not os.path.isdir(os.path.join(args.studies_folder, study_name)):
                print(
                    "Invalid entry for study_name:", study_name,
                    "does not exist as directory in", args.studies_folder)
                raise FileNotFoundError
                sys.exit(3)

            # check folder is a parent
            if len(os.listdir(os.path.join(args.studies_folder, study_name))) == 0:
                print(
                    "No files or folders were found in", study_name,
                    "Are you sure that this is the correct path?")
                raise FileNotFoundError
                sys.exit(3)

    # task
    if hasattr(args, "task"):
        if args.task not in config.tasks:
            print(
                "Invalid entry for task, "
                + "not accepted as regular task name in config.")
            raise ValueError
            sys.exit(3)

    # group nums
    if hasattr(args, "group_nums"):
        if len(args.group_nums) > 1:
            if len(args.group_nums) != len(args.study_names):
                print("Unequal numebr of group nums and study names provided.")
                raise ValueError
                sys.exit(3)
        else:
            if len(args.study_names) > 1:
                args.group_nums = [
                    args.group_nums[0] for study in args.study_names]
        for group_num in args.group_nums:
            if group_num not in range(0, 9):
                print("group_num must be an int, between 0 and 9.")
                raise ValueError
                sys.exit(3)

    # data type
    if hasattr(args, "data_type"):
        if args.data_type not in ["erps", "spectra", "contigs"]:
            print(
                "Invalid entry for data_type. "
                + "Must be one of ['erps', 'contigs', 'spectra']")
            raise ValueError
            sys.exit(3)

    # length
    if hasattr(args, "length"):
        if type(args.length) is int is False:
            print("Length must be an integer (in Hz).")
            raise ValueError
            sys.exit(3)

        try:
            if (args.length <= 0) or (args.length > 10000):
                print("Invalid entry for length, must be between 0 and 10000.")
                raise ValueError
                sys.exit(3)
        except TypeError:
            print(
                "Invalid entry for length, "
                + "must be integer value between 0 and 10000.")
            raise ValueError
            sys.exit(3)

    # channels
    if hasattr(args, "channels"):
        try:
            str(args.channels)
        except ValueError:
            print(
                "Invalid entry for channels. Must be 19-char long string of "
                + "1s and 0s")
            raise ValueError
            sys.exit(3)

        if len(args.channels) != 19:
            print(
                "Invalid entry for channels. Must be 19-char long string of "
                + "1s and 0s")
            raise ValueError
            sys.exit(3)

        for char in args.channels:
            if char != '0' and char != '1':
                print(
                    "Invalid entry for channels. Must be 19-char long string of "
                    + "1s and 0s")
                raise ValueError
                sys.exit(3)

    # artifact
    if hasattr(args, "artifact"):
        try:
            if len(str(args.artifact)) == 19:
                for char in args.artifact:
                    if int(char) < 0 or int(char) > 2:
                        raise ValueError

            elif args.artifact in ["0", "1", "2"]:
                args.artifact = int(args.artifact)

            else:
                raise ValueError

        except ValueError:
            print(
                "Invalid entry for artifact. Must be str with length 19, "
                + "or int between 0 and 2.")
            raise ValueError
            sys.exit(3)

    # erp degree
    if hasattr(args, "erp_degree"):
        if args.erp_degree not in [1, 2, None]:
            print("Invalid entry for erp_degree. Must be None, 1, or 2.")
            raise ValueError
            sys.exit(3)

    # filter band
    if hasattr(args, "filter_band"):
        if args.filter_band == "nofilter":
            pass
        elif any(band == args.filter_band for band in config.frequency_bands):
            pass
        elif any("no"+band == args.filter_band for band in config.frequency_bands):
            pass
        elif any("lo"+band == args.filter_band for band in config.frequency_bands):
            pass
        elif any("hi"+band == args.filter_band for band in config.frequency_bands):
            pass
        else:
            print("That is not a valid filterband option.")
            raise ValueError
            sys.exit(3)

    # limited subjects
    if hasattr(args, "limited_subjects"):
        if args.limited_subjects is not None:
            for subject in args.limited_subjects:
                if len(subject) != config.participantNumLen:
                    print("Subject number provided not correct length. "
                    + "Should be: " + str(config.participantNumLen))
                    raise ValueError
                    sys.exit(3)

    # balance

    # normalize
    if hasattr(args, "normalize"):
        if args.normalize not in ["standard", "minmax", None]:
            print(
            "Invalid entry for normalize. "
            + "Must be one of ['standard', 'minmax', 'None'].")
            raise ValueError
            sys.exit(3)

    # filter type
    if hasattr(args, "filter_type"):
        if args.filter_type not in ["lowpass", "highpass", "bandpass", "bandstop"]:
            print(
                "Invalid entry for type, "
                + "must be one of: lowpass, highpass, bandpass, bandstop")
            raise ValueError
            sys.exit(3)

    # frequency band
    if hasattr(args, "frequency_band"):
        if args.frequency_band not in config.frequency_bands:
            print(
                "Invalid entry for band, "
                + "must be one of: " + [val for val in config.frequency_bands])
            raise ValueError
            sys.exit(3)

    # gen spectra
    if hasattr(args, "gen_spectra"):
        if (type(args.gen_spectra) is bool) is True:
            args.gen_spectra = bool(args.gen_spectra)
        else:
            print("Type must be bool, gen_spectra is:", args.gen_spectra)

    # epochs
    if hasattr(args, "epochs"):
        try:
            if (args.epochs <= 0) or (args.epochs > 10000):
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

    # sample weight

    # tt split
    if hasattr(args, "tt_split"):
        if args.tt_split < 0 or args.tt_split > 0.999:
            print(
                "Invalid entry for tt_split. Must be float between "
                + "0 and 0.999.")
            raise ValueError
            sys.exit(3)

    # learning rate
    if hasattr(args, "learning_rate"):
        if args.learning_rate < 0.00001 or args.learning_rate > 0.99999:
            print(
                "Invalid entry for learning_rate. Must be float between "
                + "0.00001 and 0.99999.")
            raise ValueError
            sy.exit(3)

    # lr decay

    # k folds
    if hasattr(args, "k_folds"):
        if args.k_folds <= 0:
            print("Invalid entry for k_folds. Must be int 1 or greater.")
            raise ValueError
            sys.exit(3)

    # repetitions

    # depth

    # regularizer
    if hasattr(args, "regularizer"):
        if args.regularizer is not None:
            if args.regularizer not in ['l1', 'l2', 'l1_l2']:
                print("Invalid entry for regularizer. Must be in "
                + "{l1, l2, l1_l2}")
                raise ValueError
                sys.exit(3)

    # regularizer param
    if hasattr(args, "regularizer_param"):
        if (args.regularizer_param <= 0) or (args.regularizer_param >= 1):
            print(
                "Invalid entry for regularizer param. Must be float between "
                + "0 and 1.")
            raise ValueError
            sys.exit(3)

    # focal loss gamma
    if hasattr(args, "focal_loss_gamma"):
        if (args.focal_loss_gamma < 0):
            print("Invalid entry for focal gamma. Must be float >= 0.")
            raise ValueError
            sys.exit(3)

    # dropout
    if hasattr(args, "dropout"):
        if args.dropout is not None:
            if (args.dropout <= 0) or (args.dropout >= 1):
                print(
                    "Invalid entry for dropout. Must be float between 0 and 1 "
                    + "or None.")
                raise ValueError
                sys.exit(3)

    # hypertune

    # logistic regression

    # log dirs

    # checkpoint dirs

    # use gpu

    # verbosity

    return args

if __name__ == '__main__':
    main()
