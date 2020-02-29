import config
import scipy
from scipy import signal
from scipy import stats
import os
import numpy as np
import matplotlib.pyplot as plt

sourceDir = config.studyDirectory+"/contigs_"+config.selectedTask
resultsDir = config.resultsBaseDir+"/spectral"

subject_list = [fname[:3] for fname in os.listdir(sourceDir) if fname[:0]!="0"]
subject_list = set(subject_list)

print("Analyzing spectral density of subjects:", subject_list)

ctrl_low_Pz = []
ctrl_low_P3 = []
ctrl_low_P4 = []
ctrl_high_Pz = []
ctrl_high_P3 = []
ctrl_high_P4 = []

pain_low_Pz = []
pain_low_P3 = []
pain_low_P4 = []
pain_high_Pz = []
pain_high_P3 = []
pain_high_P4 = []

for sub in subject_list:
    sub_low_Pz = []
    sub_low_P3 = []
    sub_low_P4 = []
    sub_high_Pz = []
    sub_high_P3 = []
    sub_high_P4 = []

    subject_files = [fname for fname in os.listdir(sourceDir) if fname[:3]==sub]
    for contig in subject_files:
        array = np.genfromtxt(sourceDir+"/"+contig, delimiter=',')
        channel_number = 0
        for sensor_waveform in array.T:
            electrode = config.network_channels[channel_number]
            f, Pxx_den = scipy.signal.periodogram(
                sensor_waveform,
                fs=float(config.sampleRate),
                window='hann'
                )
            if electrode=="Pz":
                sub_low_Pz.append(Pxx_den[9])
                sub_high_Pz.append(Pxx_den[10])
            elif electrode=="P3":
                sub_low_P3.append(Pxx_den[9])
                sub_high_P3.append(Pxx_den[10])
            elif electrode=="P4":
                sub_low_P4.append(Pxx_den[9])
                sub_high_P4.append(Pxx_den[10])
            # plt.clf()
            # plt.semilogy(f, Pxx_den)
            # plt.ylim([1e-7, 1e2])
            # plt.xlabel('frequency [Hz]')
            # plt.ylabel('PSD [V**2/Hz]')
            # plt.show()
            channel_number+=1
    if sub[0]=="1":
        pain_low_Pz.append(sum(sub_low_Pz) / len(sub_low_Pz))
        pain_low_P3.append(sum(sub_low_P3) / len(sub_low_P3))
        pain_low_P4.append(sum(sub_low_P4) / len(sub_low_P4))
        pain_high_Pz.append(sum(sub_high_Pz) / len(sub_high_Pz))
        pain_high_P3.append(sum(sub_high_P3) / len(sub_high_P3))
        pain_high_P4.append(sum(sub_high_P4) / len(sub_high_P4))
    elif sub[0]=="2":
        ctrl_low_Pz.append(sum(sub_low_Pz) / len(sub_low_Pz))
        ctrl_low_P3.append(sum(sub_low_P3) / len(sub_low_P3))
        ctrl_low_P4.append(sum(sub_low_P4) / len(sub_low_P4))
        ctrl_high_Pz.append(sum(sub_high_Pz) / len(sub_high_Pz))
        ctrl_high_P3.append(sum(sub_high_P3) / len(sub_high_P3))
        ctrl_high_P4.append(sum(sub_high_P4) / len(sub_high_P4))

pain_diff_Pz = np.subtract(pain_high_Pz, pain_low_Pz)
pain_diff_P3 = np.subtract(pain_high_P3, pain_low_P3)
pain_diff_P4 = np.subtract(pain_high_P4, pain_low_P4)
ctrl_diff_Pz = np.subtract(ctrl_high_Pz, ctrl_low_Pz)
ctrl_diff_P3 = np.subtract(ctrl_high_P3, ctrl_low_P3)
ctrl_diff_P4 = np.subtract(ctrl_high_P4, ctrl_low_P4)

print("Pz")
print("Pain Average:", sum(pain_diff_Pz) / len(pain_diff_Pz))
print("Ctrl Average:", sum(ctrl_diff_Pz) / len(ctrl_diff_Pz))
t_test_statistic, t_test_p_value = scipy.stats.ttest_ind(pain_diff_Pz, ctrl_diff_Pz)
print("T-score:", t_test_statistic)
print("P-value:", t_test_p_value )

print("P3")
print("Pain Average:", sum(pain_diff_P3) / len(pain_diff_P3))
print("Ctrl Average:", sum(ctrl_diff_P3) / len(ctrl_diff_P3))
t_test_statistic, t_test_p_value = scipy.stats.ttest_ind(pain_diff_P3, ctrl_diff_P3)
print("T-score:", t_test_statistic)
print("P-value:", t_test_p_value )

print("P4")
print("Pain Average:", sum(pain_diff_P4) / len(pain_diff_P4))
print("Ctrl Average:", sum(ctrl_diff_P4) / len(ctrl_diff_P4))
t_test_statistic, t_test_p_value = scipy.stats.ttest_ind(pain_diff_P4, ctrl_diff_P4)
print("T-score:", t_test_statistic)
print("P-value:", t_test_p_value )
