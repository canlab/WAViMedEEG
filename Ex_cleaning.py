import Clean
import os
import sys
from tqdm import tqdm
import config
import argparse


def main():

    parser = argparse.ArgumentParser(
        description='Options for Clean.StudyFolder.autoclean() method')

    parser.add_argument('--studies_folder',
                        dest='studies_folder',
                        type=str,
                        default=config.myStudies,
                        help='Path to parent folder containing study folders')

    parser.add_argument('--study_name',
                        dest='study_name',
                        type=str,
                        default=config.studyDirectory,
                        help='Study folder containing dataset')

    parser.add_argument('--group_num',
                        dest='group_num',
                        type=int,
                        default=1,
                        help='Group number to be assigned to dataset')

    # save the variables in 'args'
    args = parser.parse_args()

    studies_folder = args.studies_folder
    study_name = args.study_name
    group_num = args.group_num

    if not os.path.isdir(studies_folder):
        print(
            "Invalid entry for studies_folder, "
            + "path does not exist as directory.")
        raise FileNotFoundError
        sys.exit(1)

    if not os.path.isdir(os.path.join(studies_folder, study_name)):
        print(
            "Invalid entry for study_name, "
            + "path does not exist as directory.")
        raise FileNotFoundError
        sys.exit(1)
        
    if group_num > 10 or group_num < 0:
        print("group_num value is out of bounds")
        raise ValueError
        sys.exit(1)
        
    if group_num is not int(group_num):
        print("group_num must be an integer")
        raise ValueError
        sys.exit(1)
        
    # my_study points to our dataset
    # ex. my_study = "/wavi/EEGstudies/CANlab/"
    my_study = studies_folder\
        + '/'\
        + study_name\

    # Instantiate a 'StudyFolder' Object
    my_study_folder = Clean.StudyFolder(my_study)

    # attempt to autoclean
    my_study_folder.autoclean(group_num=group_num)


if __name__ == '__main__':
    main()
