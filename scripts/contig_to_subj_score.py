import os
import sys
import numpy as np

path = str(sys.argv[1])
correct_logit = int(sys.argv[2])
incorrect_logit = 1 if correct_logit == 0 else 0

sub_scores = {}

with open(path) as f:
    for line in f:
        subject = str(line.split('\t')[0])
        scores = line.split('\t')[1]
        if subject not in sub_scores:
            sub_scores[subject] = []
        # scores = scores.replace('[', '')
        # scores = scores.replace(']', '')
        # scores = scores.replace('  ', ' ')
        # scores = scores.strip(' ')
        # print(scores)
        scores = [float(score) for score in scores.split(' ')[:2]]
        if len(scores) != 2:
            print("More than two scores!")
        sub_scores[subject].append(scores)

f.close()

with open(path[:-4]+"_subjects"+path[-4:], 'w') as f:
    for subject, scores in sub_scores.items():
        total_correct = 0
        for contig_score in scores:
            if contig_score[correct_logit] > contig_score[incorrect_logit]:
                total_correct += 1

        percent_correct = total_correct / len(scores)

        f.write(subject)
        f.write('\t')
        f.write(str(percent_correct))
        f.write('\n')

f.close()
