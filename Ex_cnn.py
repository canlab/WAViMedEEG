import ML, os
from tqdm import tqdm
import config
import argparse
import sys

def main():
    
    parser = argparse.ArgumentParser(description = 'user input to initialize data conditions for use in CNN')
    
    parser.add_argument('--format',
                        dest = 'data_type',
                        type = str,
                        help = 'specifies data type. Options: spectra, contig, or raw')
    
    parser.add_argument('--what',  # needs a better name
                        dest = 'no_idea',  # def needs a better name
                        type = str,
                        help = 'wtf is p300, it says its erp??? idk')  # and a better explanation
    
    parser.add_argument('--duration',
                        dest = 'contig_length',
                        type = str,
                        help = 'specifies duration of data collection. Options: 250, 500, ????')  # add to help
    
    parser.add_argument('--channels',
                        dest = 'channel_config',
                        type = str,
                        help = 'specifies accepted EEG channels. Options: ??????')  # for all the actual options
    
    parser.add_argument('--data',
                        dest = 'ML_class',
                        type = str,
                        help = 'specifies data type given to the neural net. Options: erps, contigs, or spectra')
    
    args = parser.parse_args()
    
    # set defaults when calling function   
    
    # patient_path points to our 'condition-positive' dataset
    # patient_path = "/wavi/EEGstudies/CANlab/spectra/P300_250_1111111111111111111_0_1"
    patient_path = '/wavi/EEGstudies/CANlab/' + args.data_type + '/' + args.no_idea + '_' + args.contig_length + '_' + args.channel_config

    # reference_path points to a folder containing healthy control data study folders
    # ?should this be added to arparse?
    reference_path = "/wavi/EEGstudies" 
    
    # Instantiate a 'Classifier' Object
    myclf = ML.Classifier(args.ML_class)
    
    
    # ============== Load Patient (Condition-Positive) Data ==============
    
    for fname in tqdm(os.listdir(patient_path)):
    if fname[:config.participantNumLen] not in ['1004', '1020']:
        myclf.LoadData(patient_path+"/"+fname)
    else:
        print("Skipped successfully.")
    
    # ============== Load Control (Condition-Negative) Data ==============
    # the dataset will automatically add healthy control data found in the reference folders
    
    myclf.Balance(reference_path)
    
    
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
    