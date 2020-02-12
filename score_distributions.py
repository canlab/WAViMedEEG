import os
import config
from tqdm import tqdm
import numpy as np

scoreFiles = [fname for fname in os.listdir(config.resultsDir) if "txt" in fname]
scores = []
pain_scores = []
ctrl_scores = []

print("Reading NN output files")
print("==========\n")
for fname in tqdm(scoreFiles):
    f = open(config.resultsDir+"/"+fname, 'r')
    loss = float(f.readline()[:-2].split()[1])
    accuracy = float(f.readline()[:-2].split()[1])
    if fname[0]=="1":
        accuracy = 1 - accuracy
        pain_scores.append((loss, accuracy))
    elif fname[0]=="2":
        ctrl_scores.append((loss, accuracy))
    scores.append((loss, accuracy))


import matplotlib.pyplot as plt

n, bins, patches = plt.hist(x=[score[0] for score in scores], bins='auto', color='blue', alpha=0.7, rwidth=0.85)
plt.grid(axis='y', alpha=0.75)
plt.xlabel('Accuracy')
plt.xticks(rotation=315)
plt.ylabel('Count')
plt.title('Accuracy Distribution from Jacknife Learning')
plt.text(23,45,r'$\mu=15, b=3$')
maxfreq = n.max()

plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)

plt.show()
