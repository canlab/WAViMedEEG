import os
import config
from tqdm import tqdm
import numpy as np

scoreFiles = [fname for fname in os.listdir(config.resultsDir) if "txt" in fname]

accuracies = []
pain_acc = []
ctrl_acc = []

print("Reading NN output files")
print("==========\n")
for fname in tqdm(scoreFiles):
    f = open(config.resultsDir+"/"+fname, 'r')
    f.readline()
    if fname[0]=="1":
        accuracy = float(f.readline()[:-2])
        pain_acc.append(accuracy)
    elif fname[0]=="2":
        accuracy = 1 - float(f.readline()[:-2])
        ctrl_acc.append(accuracy)
    accuracies.append(accuracy)

import matplotlib.pyplot as plt

n, bins, patches = plt.hist(x=[ctrl_acc], bins='auto', color='blue', alpha=0.7, rwidth=0.85)
plt.grid(axis='y', alpha=0.75)
plt.xlabel('Accuracy')
plt.xticks(rotation=315)
plt.ylabel('Count')
plt.title('Accuracy Distribution')
plt.text(23,45,r'$\mu=15, b=3$')
maxfreq = n.max()

plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)

plt.show()
