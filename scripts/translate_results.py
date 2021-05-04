import os, sys

translator_path = sys.argv[1]
results_path = sys.argv[2]

translator = {}

with open(translator_path, 'r') as f:
    for line in f:
        translator[line.split('\t')[1].strip()] = ' '.join(line.split('\t')[0].split('_')[:2])

with open(results_path, 'r') as r:
    with open(results_path[:-4] +"_decoded" + results_path[-4:], 'w') as f:
        for line in r:
            f.write(translator[line.split('\t')[0]])
            f.write('\t')
            f.write(line.split('\t')[1])
            f.write('\n')
