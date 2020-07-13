import numpy as np
import matplotlib.pyplot as plt
import config
import os

tasks = ["Flanker", "P300"]

resultsFolders = [
    config.studyDirectory+"/results/flnk_1250_eyes_contigs",
    config.studyDirectory+"/results/p300_1250_eyes_contigs"
]

fig, ax = plt.subplots()
ax.set_title("Eye State: Open")
plt.xlabel("Tasks")
plt.ylabel("Model Confidence in State")

for i, folder in enumerate(resultsFolders):
    data = []
    fnames = os.listdir(folder)
    for fname in fnames:
        f = open(folder+"/"+fname, 'r')
        f.readline()
        f.readline()
        prediction = float(f.readline().split(' ')[1])
        data.append(prediction)
    ax.violinplot(data, positions=[i])
    ax.boxplot(data, positions=[i])
plt.xticks(range(i+1), labels=tasks)
fig.show()
