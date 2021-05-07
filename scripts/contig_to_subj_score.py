import os
import sys
import numpy as np

scores_path = str(sys.argv[1])

translator_path = str(sys.argv[2])

try:
    positive_logit = int(sys.argv[3])
except:
    positive_logit = 1
negative_logit = 1 if positive_logit == 0 else 0

sub_scores = {}

with open(scores_path) as f:
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

translator = {}

with open(translator_path, 'r') as f:
    for line in f:
        translator[line.split('\t')[1].strip()] = ' '.join(line.split('\t')[0].split('_')[:2])

with open(scores_path[:-4]+"_translation"+scores_path[-4:], 'w') as f:
    for subject, scores in sub_scores.items():
        total_positive = 0
        for contig_score in scores:
            if contig_score[positive_logit] > contig_score[negative_logit]:
                total_positive += 1

        percent_positive = total_positive / len(scores)

        f.write(translator[subject])
        f.write('\t')
        f.write(subject)

        f.write('\t')
        f.write(str(percent_positive))
        f.write('\n')

f.close()
