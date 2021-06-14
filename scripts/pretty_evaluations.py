import os
import sys
import numpy as np

# path to contig-level output predictions
scores_path = str(sys.argv[1])

# path to study's translator file
translator_path = str(sys.argv[2])

# # path to study's subject list (optional)
# if len(sys.argv) == 4:
#     model_subjects_file = open(sys.argv[3], 'r')
#
#     # decide if a subject is in train
#     while f.read_line() != 'Train':
#         pass
#     train_subjects = []
#     while 'Test' not in f.read_line():
#         train_subjects.append(line)
#
#

try:
    positive_logit = int(sys.argv[3])
except:
    positive_logit = 1
negative_logit = 1 if positive_logit == 0 else 0

sub_scores = {}
artifacts = {}

with open(scores_path) as f:
    for line in f:
        subject = str(line.split('\t')[0])
        scores = line.split('\t')[1]
        if subject not in sub_scores:
            sub_scores[subject] = []
        if subject not in artifacts:
            artifacts[subject] = line.split('\t')[2]
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

sorted_values = sorted(translator.values())
sorted_translator = {}

for i in sorted_values:
    for k in translator.keys():
        if translator[k] == i:
            sorted_translator[k] = translator[k]
            break

with open(scores_path[:-4]+"_translation"+scores_path[-4:], 'w') as f:

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
            # f.write('\n')

f.close()
