import config
import matplotlib.pyplot as plt
import numpy as np
import os

# Physio Waveforms

fname = config.studyDirectory+"/csv/"+config.selectedTask+"/"+config.subject_of_interest+".csv"

fig, axs = plt.subplots(nrows=2, figsize=(16, 8))

plots = []
for i in config.channel_names[1:]:
    channel = config.channel_names.index(i)
    trial = np.genfromtxt(fname, delimiter=",").T[channel]
    t = np.arange(0, len(trial)) / config.sampleRate
    plots.append(axs[0].plot(t, trial)[0])

axs[0].axis([t[0], t[-1], -50, 50])
leg = axs[0].legend(config.channel_names[1:], loc="right", fancybox=True)
leg.get_frame().set_alpha(0.4)
axs[0].grid(b=True)
axs[0].set_title("Filtered Waveforms: Subject " + config.subject_of_interest + " " + config.selectedTask)
axs[0].set_ylabel("Voltage")
axs[0].set_xlabel('Time [seconds]')

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

# FMRIPREP Motion Regressors

regfname = config.studyDirectory+"/preproc/regressors/"+config.subject_of_interest+"-rest_regressors.tsv"

try:
    regressors = np.genfromtxt(regfname, delimiter="\t")
    mri_t = range(len(regressors)-1)
    mri_t = [t*config.scanner_TR for t in mri_t]

    motion = regressors[1:].T[26:32]

    mov_plots = []
    for motiontype in motion:
        mov_plots.append(axs[1].plot(mri_t, motiontype)[0])

    trans_x = regressors.T[26][1:]
    trans_y = regressors.T[27][1:]
    trans_z = regressors.T[28][1:]
    rot_x = regressors.T[29][1:]
    rot_y = regressors.T[30][1:]
    rot_z = regressors.T[31][1:]

    axs[1].axis([mri_t[0], mri_t[-1], np.amin(motion), np.amax(motion)])
    mov_leg = axs[1].legend([
        "trans_x",
        "trans_y",
        "trans_z",
        "rot_x",
        "rot_y",
        "rot_z"
    ], loc="right")
    mov_leg.get_frame().set_alpha(0.4)
    axs[1].grid(b=True)
    axs[1].set_title("Motion Estimates Extracted from FMRIPREP Output")
    axs[1].set_ylabel("Motion")
    axs[1].set_xlabel("Time [seconds]")

except:
    print("Your motion regressors didn't read in properly. Check that the naming convention matches the above code block.")

# we will set up a dict mapping legend line to orig line, and enable
# picking on the legend line
movlined = dict()
for movlegline, movorigline in zip(mov_leg.get_lines(), mov_plots):
    movlegline.set_picker(5)  # 5 pts tolerance
    movlined[movlegline] = movorigline

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
