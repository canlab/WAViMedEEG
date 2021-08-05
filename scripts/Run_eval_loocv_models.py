import os
from tqdm import tqdm
import argParser
from datetime import datetime
import numpy as np


def main():

    args = argParser.main([
        'studies_folder',
        # 'study_names',
        'task',
        'data_type',
        'log_dirs',
        'checkpoint_dirs',
        'normalize',
    ])

    studies_folder = args.studies_folder
    # study_names = args.study_names
    task = args.task
    data_type = args.data_type
    log_dirs = args.log_dirs
    checkpoint_dirs = args.checkpoint_dirs
    normalize = args.normalize

    if checkpoint_dirs is None:
        checkpoint_dirs = [
            log_dirs[0] + folder\
            for folder in os.listdir(log_dirs[0])
            if "_"+data_type in folder]
        checkpoint_dirs.sort()
    else:
        checkpoint_dirs = [log_dirs[0] + dir for dir in checkpoint_dirs]
        # each checkpoint_dir here is a fold of xvalidation.

    # for cross-fold in LOO-cross-validation
    for fold in checkpoint_dirs:
        # get num, source of left-out subject
        f = open(fold+'/subjects.txt', 'r')
        line = ""
        while "Test" not in line:
            line = f.readline()
        line = str(f.readline())
        subject, source = line.split('\t')

        # remove header (may have come from different PC, or server)
        source = '/' + source.split('//')[1]
        source = studies_folder + source

        if not os.path.isdir(source):
            print("Configuration supplied was not found in study folder data.")
            print("Failed:", source)
            raise FileNotFoundError
            sys.exit(3)

        # Instantiate a 'Classifier' Object
        myclf = ML.Classifier(data_type)

        fnames = os.listdir(source)
        fnames = [fname for fname in fnames
            if fname[:config.participantNumLen] == str(subject)]

        for fname in fnames:
            myclf.LoadData(source+'/'+fname)

        y_preds = myclf.eval_saved_CNN(
            fold,
            save_results=True)

if __name__ == '__main__':
    main()
