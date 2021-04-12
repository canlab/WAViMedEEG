import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

path = '/wavi/WAViMedEEG/logs/fit'

folders = [path+'/'+folder for folder in os.listdir(path)]
folders.sort()

i = 0

train_accs = []
train_loss = []
test_accs = []
test_loss = []

while i < len(folders):
    t_accs = []
    t_loss = []
    te_accs = []
    te_loss = []

    for folder in folders[i:i+4]:
        df = pd.read_csv(folder+'/training.log')
        t_accs.append(df['accuracy'][19])
        t_loss.append(df['loss'][19])
        te_accs.append(df['val_accuracy'][19])
        te_loss.append(df['val_loss'][19])

    train_accs.append(np.std(t_accs))
    train_loss.append(np.std(t_loss))
    test_accs.append(np.std(te_accs))
    test_loss.append(np.std(te_accs))

    i += 5

gammas = [
    0.001,
    0.00119526,
    0.00125893,
    0.00158489,
    0.00251189,
    0.00316228,
    0.00398107,
    0.00501187,
    0.00630957,
    0.00794328,
    # 0.01,
    0.01195262,
    0.01258925,
    0.01584893,
    0.02511886,
    0.03162278,
    0.03981072,
    0.05011872,
    0.06309573,
    0.07943282,
    0.1,
    0.11952623149688797,
    0.12589254117941673,
    0.15848931924611134,
    0.251188643150958,
    0.31622776601683794,
    0.3981071705534973,
    0.5011872336272724,
    0.6309573444801934,
    0.7943282347242815]

plt.xlabel('gamma')

plt.plot(gammas, train_accs, label='train accuracy')
plt.plot(gammas, train_loss, label='train loss')
plt.plot(gammas, test_accs, label='test accuracy')
plt.plot(gammas, test_loss, label='test loss')

plt.xscale('log')

plt.legend()

plt.show()
