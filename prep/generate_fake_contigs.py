import config
import random
import numpy as np

from scipy import signal
import matplotlib.pyplot as plt

# 6 peaks
t = np.linspace(0, 5, 1250, endpoint=False)
# i = signal.gausspulse(t, fc=1/100)

# square pulse
sq = signal.square(2 * np.pi * 1 * t)

# sinoid signal
sig = np.sin(2 * np.pi * t)

# convolved
newsig = signal.convolve(sig, sq, mode='same') / 600

# frequency-swept signal
sweep = signal.sweep_poly(t, np.poly1d([1, 2]))

# 3 channels, small difference
i = np.array(sweep)
j = i - 0.1
k = i - 0.2

plt.plot(t, i, t, j, t, k)
# plt.plot(t, newsig)
plt.show()

# combine into array
arr = np.stack([i, j, k]).T
print(arr.shape)

# # twenty subjects
# a = 0
# while a < 20:
#     # 20 random indeces / contigs
#     for r in range(20):
#         index = random.randint(0, 75000)
#         fname = "2"+"0"*(config.participantNumLen-len(str(a))-1)+str(a)+"_alpha_"+str(index)
#         np.savetxt(config.studyDirectory+"/contigs/p300_1250/"+fname+".csv", arr, delimiter=",", fmt="%2.1f")
#     a+=1
