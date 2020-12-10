from scipy import signal
import numpy as np
import config


def sin_wave(
    time,
    sample_rate=config.sample_rate,
    plot=False,
        noise=True):

    t = np.linspace(
        0,
        time // sample_rate,
        time * sample_rate,
        endpoint=False)

    sig = np.sin(
        12 * np.pi * t)

    if noise is True:
        sig = sig + np.random.normal(
            loc=0,
            size=time*sample_rate,
            scale=0.5)

    if plot is True:
        from Plots import plot_signal
        plot_signal(t, sig)

    return np.array(sig)


def square_wave(
    time,
    sample_rate=config.sample_rate,
    plot=False,
        noise=True):

    t = np.linspace(
        0,
        time // sample_rate,
        time * sample_rate,
        endpoint=False)

    sig = np.sin(2 * np.pi * t)

    if noise is True:
        sig = sig + np.random.normal(
            loc=0,
            size=time*sample_rate,
            scale=0.5)

    pwm = signal.square(
        12 * np.pi * 30 * t,
        duty=(sig + 1)/2)

    if plot is True:
        from Plots import plot_signal
        plot_signal(t, pwm)

    return np.array(pwm)
