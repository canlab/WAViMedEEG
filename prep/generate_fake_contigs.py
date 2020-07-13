import config
import random
import numpy as np

from scipy import signal
import matplotlib.pyplot as plt

print("Enter subject code you'd like to use:")
subject_code = input()

print("Enter number of peaks you'd like to use:")
peaks1 = int(input())
print("Second wave peaks:")
peaks2 = int(input())
print("Third wave peaks:")
peaks3 = int(input())

print("Enter the scale of the noise you'd like to use (can be float or int):")
noise_scale = float(input())

# 2nd parameter is number of peaks
t1 = np.linspace(0, peaks1, 1250, endpoint=False)
t2 = np.linspace(0, peaks2, 1250, endpoint=False)
t3 = np.linspace(0, peaks3, 1250, endpoint=False)

# i = signal.gausspulse(t, fc=1/100)

# square pulse
# sq = signal.square(2 * np.pi * 1 * t)

# sinoid signal
sig1 = 0.3*np.sin(2 * np.pi * t1)
sig2 = 0.1*np.sin(2 * np.pi * t2)
sig3 = 0.6*np.sin(2 * np.pi * t3)

# convolved
# newsig = signal.convolve(sig, sq, mode='same') / 600

# frequency-swept signal
# sweep = signal.sweep_poly(t, np.poly1d([1, 2]))

# plt.plot(t, newsig)
plt.show()

# twenty subjects
a = 0
while a < 20:
    # 25 random indeces / contigs
    r = 0
    while r < 25:
        # random starting index
        index = random.randint(0, 75000)

        # 3 channels, each with random noise
        i = np.array(sig1 + sig2 + sig3 + np.random.normal(0, noise_scale, 1250))
        j = np.array(sig1 + sig2 + sig3 + np.random.normal(0, noise_scale, 1250))
        k = np.array(sig1 + sig2 + sig3 + np.random.normal(0, noise_scale, 1250))

        # plot first contig
        if a == 0:
            if r == 0:
                plt.plot(t1, i, t2, j, t3, k)
                plt.show()

        # combine into array
        arr = np.stack([i, j, k]).T

        fname = subject_code+"0"*(config.participantNumLen-len(str(a))-1)+str(a)+"_alpha_"+str(index)
        np.savetxt(config.studyDirectory+"/contigs/p300_1250/"+fname+".csv", arr, delimiter=",", fmt="%2.1f")
        r+=1
    a+=1
