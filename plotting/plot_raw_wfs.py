import config
import matplotlib.pyplot as plt
import numpy as np
import os

# Plotting raw waveforms
wav_fname = config.plot_waves
art_fname = config.plot_art

# fig, axs = plt.subplots(nrows=len(config.channel_names), figsize=(16, 8), sharex=True, sharey=True)
fig, axs = plt.subplots(nrows=1, figsize=(16, 8))

trials = []
plots = []
for channel in config.channel_names:
    i = config.channel_names.index(channel)
    trial = np.genfromtxt(wav_fname, delimiter=" ").T[i]
    trials.append(trial)
    wavs = trial * 0.1
    print(wavs)
    t = np.arange(0, len(wavs)) / config.sampleRate
    wavs = np.subtract(wavs, i*25)
    plots.append(axs.plot(t, wavs)[0])

# def
axs.axis([t[0], t[-1], -25*len(config.channel_names), 5*len(config.channel_names)])
leg = axs.legend(config.channel_names, loc="right", fancybox=True)
leg.get_frame().set_alpha(0.4)
axs.grid(b=True)
axs.set_title("Raw Waveforms: Subject " + config.plot_subject + " " + config.selectedTask)
axs.set_ylabel("Voltage [mV]")
axs.set_xlabel('Time [seconds]')

# we will set up a dict mapping legend line to orig line, and enable
# picking on the legend line
lined = dict()
for legline, origline in zip(leg.get_lines(), plots):
    legline.set_picker(5)  # 5 pts tolerance
    lined[legline] = origline

def onpick(event):
    # on the pick event, find the orig line corresponding to the
    # legend proxy line, and toggle the visibility
    legline = event.artist
    origline = lined[legline]
    vis = not origline.get_visible()
    origline.set_visible(vis)
    # Change the alpha on the line in the legend so we can see what lines
    # have been toggled
    if vis:
        legline.set_alpha(1.0)
    else:
        legline.set_alpha(0.2)
    fig.canvas.draw()

fig.canvas.mpl_connect('pick_event', onpick)

# cross-spectroscopy
# for wav in trials:
#     axs[1].psd(wav, NFFT=len(t), pad_to=len(t), Fs=config.sampleRate)
#     axs[1].psd(wav, NFFT=len(t), pad_to=len(t)*2, Fs=config.sampleRate)
#     axs[1].psd(wav, NFFT=len(t), pad_to=len(t)*4, Fs=config.sampleRate)


# FMRIPREP Motion Regressors
#
# regfname = config.studyDirectory+"/preproc/regressors/"+config.subject_of_interest+"-rest_regressors.tsv"
#
# try:
#     regressors = np.genfromtxt(regfname, delimiter="\t")
#     mri_t = range(len(regressors)-1)
#     mri_t = [t*config.scanner_TR for t in mri_t]
#
#     motion = regressors[1:].T[26:32]
#
#     mov_plots = []
#     for motiontype in motion:
#         mov_plots.append(axs[1].plot(mri_t, motiontype)[0])
#
#     trans_x = regressors.T[26][1:]
#     trans_y = regressors.T[27][1:]
#     trans_z = regressors.T[28][1:]
#     rot_x = regressors.T[29][1:]
#     rot_y = regressors.T[30][1:]
#     rot_z = regressors.T[31][1:]
#
#     axs[1].axis([mri_t[0], mri_t[-1], np.amin(motion), np.amax(motion)])
#     mov_leg = axs[1].legend([
#         "trans_x",
#         "trans_y",
#         "trans_z",
#         "rot_x",
#         "rot_y",
#         "rot_z"
#     ], loc="right")
#     mov_leg.get_frame().set_alpha(0.4)
#     axs[1].grid(b=True)
#     axs[1].set_title("Motion Estimates Extracted from FMRIPREP Output")
#     axs[1].set_ylabel("Motion")
#     axs[1].set_xlabel("Time [seconds]")
#
# except:
#     print("Your motion regressors didn't read in properly. Check that the naming convention matches the above code block.")

# we will set up a dict mapping legend line to orig line, and enable
# picking on the legend line
# movlined = dict()
# for movlegline, movorigline in zip(mov_leg.get_lines(), mov_plots):
#     movlegline.set_picker(5)  # 5 pts tolerance
#     movlined[movlegline] = movorigline

def movonpick(event):
    # on the pick event, find the orig line corresponding to the
    # legend proxy line, and toggle the visibility
    movlegline = event.artist
    movorigline = movlined[movlegline]
    vis = not movorigline.get_visible()
    movorigline.set_visible(vis)
    # Change the alpha on the line in the legend so we can see what lines
    # have been toggled
    if vis:
        movlegline.set_alpha(1.0)
    else:
        movlegline.set_alpha(0.2)
    fig.canvas.draw()

fig.canvas.mpl_connect('pick_event', movonpick)

plt.tight_layout()
plt.show()
