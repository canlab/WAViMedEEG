from scipy import signal
import numpy as np

def sin_wave(time, sampleRate=250, plot=False):

    t = np.linspace(0, time, sampleRate, endpoint=False, noise=True)

    sig = np.sin(2 * np.pi * t)

    if noise == True:
        sig = sig + np.random.normal(0, time, sampleRate)

    if plot==True:
        from Plots import plot_signal
        plot_signal(t, sig)

    return np.array(sig)

def square_wave(time, sampleRate=250, plot=False):

    t = np.linspace(0, time, sampleRate, endpoint=False, noise=True)

    sig = np.sin(2 * np.pi * t)

    if noise == True:
        sig = sig + np.random.normal(0, time, sampleRate)

    pwm = signal.square(2 * np.pi * 30 * t, duty=(sig + 1)/2)

    if plot==True:
        from Plots import plot_signal
        plot_signal(t, pwm)

    return np.array(pwm)
