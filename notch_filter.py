# -*- coding: utf-8 -*-
"""
Created on Sun Dec 22 12:15:20 2019

Notch filter test for high-frequency SSVEP

@author: Brynhildr
"""

#%% Import 3rd part modules
from scipy import signal
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

#%% Initialization
fs = 10000           # sample frequency
f0 = 50.0            # notch frequency
Q = 10.0             # quality factor
w0 = f0 / (fs/2)     # normalized frequency

#%% Design notch filter
b, a = signal.iirnotch(w0, Q)

#%% frequency response
w, h = signal.freqz(b, a)

# generate frequency axis
freq = w*fs / (2*np.pi)

# plot
sns.set(style='whitegrid')

fig, ax = plt.subplots(2, 1, figsize=(8, 6))

ax[0].plot(freq, 20*np.log10(abs(h)), color='tab:blue')
ax[0].set_title('Frequency Response', fontsize=16)
ax[0].set_ylabel('Amplitude/dB', fontsize=16)
ax[0].set_xlabel('Frequency/Hz', fontsize=16)
ax[0].set_xlim([0, 100])
ax[0].set_ylim([-40, 10])

ax[1].plot(freq, np.unwrap(np.angle(h))*180/np.pi, color='tab:orange')
ax[1].set_ylabel('Anlge(degrees)', fontsize=16)
ax[1].set_xlabel('Frequency/Hz', fontsize=16)
ax[1].set_xlim([0, 100])
ax[1].set_yticks([-90, -60, -30, 0, 30, 60, 90])
ax[1].set_ylim([-90, 90])

plt.show()

#%% triangle waveform test
def triangle_wave(x, T, bx, a, by, d):
    '''
    make a triangle wave sequence
    param T: time points in one cycle
    param bx: bias in x-axis (decrease for right, increase for left)
    param a: amplitude of wave (peak-to-peak value)
    param by: bias in y-axis (decrease for down, increase for up)
    param d: the deviation of Guass noise
    '''
    # downside of wave
    y = np.where(np.mod(x-bx, T) < T/2, -4/T * (np.mod(x-bx, T))+1+by/a, 0)
    # upside of wave
    y = np.where(np.mod(x-bx, T) >= T/2, 4/T * (np.mod(x-bx, T))-3+by/a, y)
    # plus amplitude, add noise
    return a*y + d*np.random.randn(len(x))

x = np.arange(0,1000,1)
y = triangle_wave(x=x, T=50, bx=3, a=4, by=4, d=1)

plt.plot(y)
plt.show()
