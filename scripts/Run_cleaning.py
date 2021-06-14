import sys
sys.path.append('..')
from src import Clean
from src import config
import os
import shutil
from tqdm import tqdm
import argParser


def main():

    args = argParser.main([
        'studies_folder',
        'study_names',
        'group_nums',
        'force',
        ])

    studies_folder = args.studies_folder
    study_names = args.study_names
    group_nums = args.group_nums
    force = args.force

    # my_study points to our dataset
    # ex. my_study = "/wavi/EEGstudies/CANlab/"
    my_studies = [
        os.path.join(
            studies_folder,
            study_name)\
        for study_name in study_names]

    for i, my_study in enumerate(my_studies):
        # Instantiate a 'StudyFolder' Object
        my_study_folder = Clean.StudyFolder(my_study)

        # attempt to autoclean
        my_study_folder.autoclean(group_num=group_nums[i])


if __name__ == '__main__':
    main()
