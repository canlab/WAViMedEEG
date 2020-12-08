import ML, os
from tqdm import tqdm
import config
import argparse

def main():

    parser = argparse.ArgumentParser(description = 'Runs CNN (convoluional neural network) method of ML.Classifier')

    parser.add_argument('data_type',
                        dest = 'data_type',
                        type = str,
                        help = 'Input data type: contigs, erps, or spectra')

    parser.add_argument('--studies_folder',
                        dest = 'studies_folder',
                        type = str,
                        help = 'Path to parent folder containing study folders',
                        default = config.myStudies)

    parser.add_argument('--study_name',
                        dest = 'study_name',
                        type = str,
                        help = 'Study folder containing condition-positive dataset',
                        default = config.studyDirectory)

    parser.add_argument('--task',
                        dest = 'task',
                        type = str,
                        help = 'Four-character task name. Options: ' + [key for key, val in config.tasks],
                        default = 'P300')

    parser.add_argument('--duration',
                        dest = 'contig_length',
                        type = str,
                        help = 'Duration of input data, in number of samples @ ' + config.sampleRate + ' Hz',
                        default = '250')

    parser.add_argument('--channels',
                        dest = 'channel_config',
                        type = str,
                        help = 'Binary string specifying which of the following EEG channels will be included in analysis: ' + config.channel_names,
                        default = '1111111111111111111')

    args = parser.parse_args()


    # patient_path points to our 'condition-positive' dataset
    # ex. patient_path = "/wavi/EEGstudies/CANlab/spectra/P300_250_1111111111111111111_0_1"
    patient_path = args.studies_folder\
        + '/'\
        + args.study_name\
        + '/'\
        + args.data_type\
        + '/'\
        + args.task\
        + '_'\
        + args.contig_length\
        + '_'\
        + args.channel_config

    # Instantiate a 'Classifier' Object
    myclf = ML.Classifier(args.data_type)

    # ============== Load Patient (Condition-Positive) Data ==============

    for fname in tqdm(os.listdir(patient_path)):
        if fname[:config.participantNumLen] not in config.excludeSubs:
            myclf.LoadData(patient_path+"/"+fname)

    # ============== Load Control (Condition-Negative) Data ==============
    # the dataset will automatically add healthy control data found in the reference folders

    myclf.Balance(args.studies_folder)

    # ============== Run 'CNN' method of 'Classifier' ==============
    # This method will structure the input classes (in this case, 'Spectra' objects)

    # ?should all these parameters be also in the argparse? or something like a config file?
    # epochs: (int) default 100, number of training iterations to be run
    # normalize: (None, 'standard', 'minmax') default None, z-score normalize input data (features)
    # plot_ROC: (bool) default 'False', plot sensitivity-specificity curve on validation dataset
    # tt_split: (float) default 0.33, ratio of test samples to train samples
    # learning_rate: (float) default 0.01
    # lr_decay: (bool) default False, whether or not the learning rate should decay adhering to a 0.96 decay rate schedule

    myclf.CNN(learning_rate=0.01, plot_ROC=True)

if __name__ == '__main__':
    main()
