from src import config
from scipy import signal
import numpy as np
import random


def sin_wave(
    time,
    sample_rate=config.sample_rate,
    plot=False,
        noise=True):

    t = np.linspace(
        0,
        time,
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
        time,
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


def rand_bin_string(
    time,
    sample_rate=config.sample_rate,
    weighted=False,
        hi_bound=1):

    t = np.linspace(
        0,
        time,
        time * sample_rate,
        endpoint=False)

    if weighted is False:
        sig = [random.randint(0, hi_bound) for rand in t]
    elif weighted is True:
        weights = [0.9]
        pop = np.linspace(0, hi_bound, 1, endpoint=True)
        for samp in pop[1:]:
            weights.append(0.1)
        sig = random.choices(pop, cum_weights=weights, k=len(t))

    return np.array(sig)
