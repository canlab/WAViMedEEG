import ML, os
from tqdm import tqdm
import config
import argparse

def main():


    parser = argparse.ArgumentParser(description = 'Options for CNN (convoluional neural network) method of ML.Classifier')

    parser.add_argument('data_type',
                        dest = 'data_type',
                        type = str,
                        help = 'Input data type: contigs, erps, or spectra')

    parser.add_argument('--studies_folder',
                        dest = 'studies_folder',
                        type = str,
                        default = config.myStudies,
                        help = 'Path to parent folder containing study folders')

    parser.add_argument('--study_name',
                        dest = 'study_name',
                        type = str,
                        default = config.studyDirectory,
                        help = 'Study folder containing condition-positive dataset')

    parser.add_argument('--task',
                        dest = 'task',
                        type = str,
                        default = 'P300',
                        help = 'Four-character task name. Options: ' + str([key for key, val in config.tasks]))

    parser.add_argument('--length',
                        dest = 'length',
                        type = str,
                        default = '250',
                        help = 'Duration of input data, in number of samples @ ' + str(config.sampleRate) + ' Hz')

    parser.add_argument('--channels',
                        dest = 'channels',
                        type = str,
                        default = '1111111111111111111',
                        help = 'Binary string specifying which of the following EEG channels will be included in analysis: ' + str(config.channel_names))

    # save the variables in 'args'
    args = parser.parse_args()

    data_type = args.data_type
    studies_folder = args.studies_folder
    study_name = args.study_name
    task = args.task
    length = args.length
    channels = args.channels


    # patient_path points to our 'condition-positive' dataset
    # ex. patient_path = "/wavi/EEGstudies/CANlab/spectra/P300_250_1111111111111111111_0_1"
    patient_path = studies_folder\
        + '/'\
        + study_name\
        + '/'\
        + data_type\
        + '/'\
        + task\
        + '_'\
        + length\
        + '_'\
        + channels

    # Instantiate a 'Classifier' Object
    myclf = ML.Classifier(args.data_type)

    # ============== Load Patient (Condition-Positive) Data ==============

    for fname in tqdm(os.listdir(patient_path)):
        if fname[:config.participantNumLen] not in config.excludeSubs:
            myclf.LoadData(patient_path+"/"+fname)

    # ============== Load Control (Condition-Negative) Data ==============
    # the dataset will automatically add healthy control data found in the reference folders

    myclf.Balance(studies_folder)

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
