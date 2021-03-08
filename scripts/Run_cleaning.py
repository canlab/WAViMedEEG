import sys
sys.path.append('..')
from src import Clean
from src import config
import os
from tqdm import tqdm
import argparse


def main():

    parser = argparse.ArgumentParser(
        description='Options for Clean.StudyFolder.autoclean() method')

    parser.add_argument('--studies_folder',
                        dest='studies_folder',
                        type=str,
                        default=config.my_studies,
                        help="(Default: " + config.my_studies + ") Path to "
                        + "parent folder containing study folders")

    parser.add_argument('--study_name',
                        dest='study_name',
                        type=str,
                        default=config.study_directory,
                        help="(Default: " + config.study_directory + ") "
                        + "Study folder containing dataset")

    parser.add_argument('--group_num',
                        dest='group_num',
                        type=int,
                        default=1,
                        help='(Default: 1) Group number to be '
                        + 'assigned to dataset')

    # save the variables in 'args'
    args = parser.parse_args()

    studies_folder = args.studies_folder
    study_name = args.study_name
    group_num = args.group_num

    # ERROR HANDLING
    if not os.path.isdir(studies_folder):
        print(
            "Invalid entry for studies_folder, "
            + "path does not exist as directory.")
        raise FileNotFoundError
        sys.exit(3)

    if not os.path.isdir(os.path.join(studies_folder, study_name)):
        print(
            "Invalid entry for study_name, "
            + "path does not exist as directory.")
        raise FileNotFoundError
        sys.exit(3)

    if not os.path.isdir(os.path.join(studies_folder, study_name, 'raw')):
        print(
            "No 'raw' folder found in the supplied directory. See the README. "
            + "Data should be stored in studies_folder/study_name/raw before "
            + "data cleaning can be run automatically."
        )
        raise FileNotFoundError
        sys.exit(3)

    if len(os.listdir(os.path.join(studies_folder, study_name))) > 2:
        print(
            "Looks like this folder has already been cleaned. "
            "Should probably move other study folders to avoid overwrite."
        )
        raise FileExistsError
        sys.exit(3)
        # TODO avoid overwrites here, / add force option

    if group_num not in range(0, 9):
        print("group_num must be an int, between 0 and 9.")
        raise ValueError
        sys.exit(3)

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
