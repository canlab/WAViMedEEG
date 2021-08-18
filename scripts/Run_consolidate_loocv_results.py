import os
from tqdm import tqdm
import argParser
from datetime import datetime
import numpy as np
from src import ML
from src import config


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

    # start results file
    with open(log_dirs[0]+'/loocv_results.txt', 'w') as f:

        sub_scores = {}
        artifacts = {}
        translator = {}
        # for cross-fold in LOO-cross-validation
        for fold in checkpoint_dirs:
            fold_preds = open(fold+'/loocv_predictions.txt', 'r')
            positive_logit = 1
            negative_logit = 1 if positive_logit == 0 else 0

            for line in fold_preds:
                subject = str(line.split('\t')[0])
                data_path = str(line.split('\t')[1])
                scores = str(line.split('\t')[2])
                if subject not in sub_scores:
                    sub_scores[subject] = []
                if subject not in artifacts:
                    artifacts[subject] = line.split('\t')[3]
                scores = [float(score) for score in scores.split(' ')[:2]]
                if len(scores) != 2:
                    print("More than two scores!")
                sub_scores[subject].append(scores)

                with open(os.path.dirname(os.path.dirname(data_path))+'/translator_P300.txt', 'r') as trans_f:
                    for trans_line in trans_f:
                        translator[trans_line.split('\t')[1].strip()] = ' '.join(trans_line.split('\t')[0].split(' ')[:2])

            fold_preds.close()

        sorted_values = sorted(translator.values())
        sorted_translator = {}

        for i in sorted_values:
            for k in translator.keys():
                if translator[k] == i:
                    sorted_translator[k] = translator[k]
                    break

        # iterate through keys of translator alphabetically by values
        for sub_k in sorted_translator.keys():

            # count total number of condition-positive predictions for this subject
            # i.e. of the groups used to train the model,
            # how many classified as the one with the higher group-number?
            # how many had output-layer node 1 higher than node 0 on prediction?
            total_positive = 0

            # of all contig scores for this subject,
            if sub_k in sub_scores:
                for contig_score in sub_scores[sub_k]:
                    # check if w @ higher output node > w @ lesser output node
                    if contig_score[positive_logit] > contig_score[negative_logit]:
                        total_positive += 1

                    percent_positive = total_positive / len(sub_scores[sub_k])

                f.write(sorted_translator[sub_k])
                f.write('\t')
                f.write(sub_k)
                f.write('\t')
                f.write(str(percent_positive))
                f.write('\t')
                f.write(artifacts[sub_k])

    f.close()


if __name__ == '__main__':
    main()
