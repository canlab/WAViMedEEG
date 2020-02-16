import os
import config
from tqdm import tqdm
import numpy as np

scoreFiles = [fname for fname in os.listdir(config.resultsDir) if "txt" in fname]
accuracies = []
pain_acc = []
ctrl_acc = []

losses = []
pain_loss = []
ctrl_loss = []

print("Reading NN output files")
print("==========\n")
for fname in tqdm(scoreFiles):
    f = open(config.resultsDir+"/"+fname, 'r')
    group = int(f.readline().split()[1])
    print(group)
    loss = float(f.readline()[:-2].split()[1])
    losses.append(loss)
    if group==1:
        accuracy = float(f.readline()[:-2].split()[1])
        pain_acc.append(accuracy)
        pain_loss.append(loss)
    elif group==2:
        accuracy = 1 - float(f.readline()[:-2].split()[1])
        ctrl_acc.append(accuracy)
        ctrl_loss.append(loss)
    accuracies.append(accuracy)

import matplotlib.pyplot as plt

n, bins, patches = plt.hist(x=[accuracies], bins='auto', color='blue', alpha=0.7, rwidth=0.85)
plt.grid(axis='y', alpha=0.75)
plt.xlabel('Accuracy')
plt.xticks(rotation=315)
plt.ylabel('Count')
plt.title('Accuracy Distribution from Jacknife Learning')
plt.text(23,45,r'$\mu=15, b=3$')
maxfreq = n.max()

plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)

plt.show()
