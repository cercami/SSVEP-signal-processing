# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 14:45:31 2019
Data preprocessing:
    (1)load data from .cnt file to .mat file
    (2)fitering data
    (3)SRCA optimization
    
Continuously updating...
@author: Brynhildr
"""

# %%
import os
import scipy.io as io

import numpy as np
from numpy import newaxis as NA

import mne
from mne import Epochs
from mne.filter import filter_data

import copy
import string

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
%matplotlib auto
# %% D:\SSVEP\photocell_no_interval
tmin, tmax = -0.1, 2
sfreq = 1000

n_events = 32
n_trials = 32

n_times = int((tmax-tmin)*sfreq + 1)

data = np.zeros((1, n_trials, n_times))
symbols = ''.join([string.ascii_uppercase, '_12345'])
for text in symbols:
    filename = text + '.cnt'
    filepath = 'D:\SSVEP\photocell_no_interval\\' + filename
    montage = mne.channels.make_standard_montage('standard_1020')
    raw_cnt = mne.io.read_raw_cnt(filepath, eog=['HEO', 'VEO'], emg=['EMG'], ecg=['EKG'],
            preload=True, verbose=False)

    events, events_id = mne.events_from_annotations(raw_cnt)

    drop_chans = ['M1', 'M2']
    picks = mne.pick_types(raw_cnt.info, emg=False, eeg=True, stim=False, eog=False, exclude=drop_chans)
    picks_ch_names = [raw_cnt.ch_names[i] for i in picks]

    seg_data = Epochs(raw_cnt, events=events, event_id=1, tmin=tmin, picks=picks, tmax=tmax,
                baseline=None, preload=True).get_data()
    data = np.concatenate((data, seg_data[NA,:,26,:]), axis=0)

del seg_data, raw_cnt, drop_chans, events, events_id, filename, filepath
del montage, picks, text

data = np.delete(data, 0, axis=0)

mean_data = data.mean(axis=1)

# %%
def trim_axs(axs, N):
    axs = axs.flat
    for ax in axs[N:]:
        ax.remove()

    return axs[:N]

symbols = list(''.join([string.ascii_uppercase, '_12345']))
figsize = (80, 30)
cols, rows = 8, 4

sns.set(style='whitegrid')

axs = plt.figure(figsize=figsize, constrained_layout=True).subplots(rows, cols)
axs = trim_axs(axs, len(symbols))

i = 0
for ax, symbol in zip(axs, symbols):
    symbol_index = symbols.index(symbol)
    ax.set_title(symbol, fontsize=18)
    ax.tick_params(axis='both', labelsize=14)
    ax.plot(mean_data[i,:],label='total data')
    ax.vlines(100, 0.025, 0.225, colors='black', linestyle='dashed', label='1st code (start)')
    ax.vlines(400, 0.025, 0.225, colors='black', linestyle='dashed', label='2nd code')
    ax.vlines(700, 0.025, 0.225, colors='black', linestyle='dashed', label='3rd code')
    ax.vlines(1000, 0.025, 0.225, colors='black', linestyle='dashed', label='4th code')
    ax.vlines(1300, 0.025, 0.225, colors='black', linestyle='dashed', label='5th code')
    ax.vlines(1600, 0.025, 0.225, colors='black', linestyle='dashed', label='end')
    ax.set_xlabel('Time/ms', fontsize=16)
    ax.set_ylabel('Amplitude/V', fontsize=16)
    ax.legend(loc='lower left', fontsize=14)
    i += 1

plt.show()
plt.savefig(r'C:\Users\Administrator\Desktop\photocell_32codes.png', dpi=600)

# %%
data_path = r'D:\SSVEP\program\code_1bits.mat'
code_data = io.loadmat(data_path)
code_series = code_data['VEPSeries_1bits']  # (n_codes, n_elements)
# code_A = code_series[]

# %%
for i in range(32):
    fig = plt.figure(figsize=(8,3))
    gs = GridSpec(1,1, figure=fig)
    sns.set(style='whitegrid')

    title = 'code-' + symbols[i] + ': ' + str(code_series[:,i])
    ax = fig.add_subplot(gs[:,:])

    ax.set_title(title, fontsize=26)
    ax.tick_params(axis='both', labelsize=22)
    ax.plot(mean_data[i,:])
    ax.vlines(112, 0.025, 0.225, colors='black', linewidth=2, linestyle='dashed')
    ax.vlines(412, 0.025, 0.225, colors='black', linewidth=2, linestyle='dashed')

    ax.vlines(512, 0.025, 0.225, colors='black', linewidth=2, linestyle='dashed')
    ax.vlines(812, 0.025, 0.225, colors='black', linewidth=2, linestyle='dashed')

    ax.vlines(912, 0.025, 0.225, colors='black', linewidth=2, linestyle='dashed')
    ax.vlines(1212, 0.025, 0.225, colors='black', linewidth=2, linestyle='dashed')

    ax.vlines(1312, 0.025, 0.225, colors='black', linewidth=2, linestyle='dashed')
    ax.vlines(1612, 0.025, 0.225, colors='black', linewidth=2, linestyle='dashed')

    ax.vlines(1712, 0.025, 0.225, colors='black', linewidth=2, linestyle='dashed')
    ax.vlines(2012, 0.025, 0.225, colors='black', linewidth=2, linestyle='dashed')
    ax.set_xlabel('Time/ms', fontsize=24)
    ax.set_ylabel('Amplitude/μV', fontsize=24)

    plt.tight_layout()
    filename = r'C:\Users\Administrator\Desktop\photocell_' + symbols[i] + '.png'
    plt.savefig(filename, dpi=600)
    print("code_" + symbols[i] + "'s figure done!")

# %%
a,b = 0,2
code_a, code_b = str(code_series[:,a]), str(code_series[:,b])
fig = plt.figure(figsize=(21,3))
plt.plot(mean_data[a,:], label=code_a)
plt.plot(mean_data[b,:], label=code_b)

plt.vlines(112, 0.025, 0.225, colors='black', linewidth=2, linestyle='dashed')
plt.vlines(412, 0.025, 0.225, colors='black', linewidth=2, linestyle='dashed')

plt.vlines(512, 0.025, 0.225, colors='black', linewidth=2, linestyle='dashed')
plt.vlines(812, 0.025, 0.225, colors='black', linewidth=2, linestyle='dashed')

plt.vlines(912, 0.025, 0.225, colors='black', linewidth=2, linestyle='dashed')
plt.vlines(1212, 0.025, 0.225, colors='black', linewidth=2, linestyle='dashed')

plt.vlines(1312, 0.025, 0.225, colors='black', linewidth=2, linestyle='dashed')
plt.vlines(1612, 0.025, 0.225, colors='black', linewidth=2, linestyle='dashed')

plt.vlines(1712, 0.025, 0.225, colors='black', linewidth=2, linestyle='dashed')
plt.vlines(2012, 0.025, 0.225, colors='black', linewidth=2, linestyle='dashed')

plt.legend(loc='best')

# %%
fig = plt.figure(figsize=(24,18))
gs = GridSpec(4,1, figure=fig)
sns.set(style='whitegrid')

title = []
for i in range(32):
    title.append("code-'" + symbols[i] + "': " + str(code_series[:,i]))

m = 7
k = m*4

ax1 = fig.add_subplot(gs[:1,:])
ax1.set_title(title[k], fontsize=24)
ax1.tick_params(axis='both', labelsize=20)
for i in range(5):
    if code_series[i,k] == 0:
        ax1.plot(np.arange(301)+i*300, mean_data[k, 112+i*300:413+i*300],
                 color='tab:blue')
    elif code_series[i,k] == 1:
        ax1.plot(np.arange(301)+i*300, mean_data[k, 112+i*300:413+i*300],
                 color='tab:orange')
# ax1.plot(np.arange(112), mean_data[k, :112], color='black')
# ax1.plot(np.arange(89)+2012, mean_data[k, -89:], color='black')
# for i in range(4):
#     ax1.plot(np.arange(100)+412+i*400, mean_data[k, 412+i*400:512+i*400], color='black')
ax1.vlines(0, 0.025, 0.225, colors='black', linewidth=2, linestyle='dashed')
ax1.vlines(300, 0.025, 0.225, colors='black', linewidth=2, linestyle='dashed')
ax1.vlines(600, 0.025, 0.225, colors='black', linewidth=2, linestyle='dashed')
ax1.vlines(900, 0.025, 0.225, colors='black', linewidth=2, linestyle='dashed')
ax1.vlines(1200, 0.025, 0.225, colors='black', linewidth=2, linestyle='dashed')
# ax1.vlines(512, 0.025, 0.225, colors='black', linewidth=2, linestyle='dashed')
# ax1.vlines(812, 0.025, 0.225, colors='black', linewidth=2, linestyle='dashed')
# ax1.vlines(912, 0.025, 0.225, colors='black', linewidth=2, linestyle='dashed')
# ax1.vlines(1212, 0.025, 0.225, colors='black', linewidth=2, linestyle='dashed')
# ax1.vlines(1312, 0.025, 0.225, colors='black', linewidth=2, linestyle='dashed')
# ax1.vlines(1612, 0.025, 0.225, colors='black', linewidth=2, linestyle='dashed')
# ax1.vlines(1712, 0.025, 0.225, colors='black', linewidth=2, linestyle='dashed')
# ax1.vlines(2012, 0.025, 0.225, colors='black', linewidth=2, linestyle='dashed')
ax1.set_xlabel('Time/ms', fontsize=24)
ax1.set_ylabel('Amplitude/V', fontsize=24)

ax2 = fig.add_subplot(gs[1:2,:])
ax2.set_title(title[k+1], fontsize=24)
ax2.tick_params(axis='both', labelsize=20)
for i in range(5):
    if code_series[i,k+1] == 0:
        ax2.plot(np.arange(301)+i*300, mean_data[k+1, 112+i*300:413+i*300], color='tab:blue')
    elif code_series[i,k+1] == 1:
        ax2.plot(np.arange(301)+i*300, mean_data[k+1, 112+i*300:413+i*300], color='tab:orange')
# ax2.plot(np.arange(112), mean_data[k+1, :112], color='black')
# ax2.plot(np.arange(89)+2012, mean_data[k+1, -89:], color='black')
# for i in range(4):
#     ax2.plot(np.arange(100)+412+i*400, mean_data[k+1, 412+i*400:512+i*400], color='black')
ax2.vlines(0, 0.025, 0.225, colors='black', linewidth=2, linestyle='dashed')
ax2.vlines(300, 0.025, 0.225, colors='black', linewidth=2, linestyle='dashed')
ax2.vlines(600, 0.025, 0.225, colors='black', linewidth=2, linestyle='dashed')
ax2.vlines(900, 0.025, 0.225, colors='black', linewidth=2, linestyle='dashed')
ax2.vlines(1200, 0.025, 0.225, colors='black', linewidth=2, linestyle='dashed')
# ax2.vlines(1212, 0.025, 0.225, colors='black', linewidth=2, linestyle='dashed')
# ax2.vlines(1312, 0.025, 0.225, colors='black', linewidth=2, linestyle='dashed')
# ax2.vlines(1612, 0.025, 0.225, colors='black', linewidth=2, linestyle='dashed')
# ax2.vlines(1712, 0.025, 0.225, colors='black', linewidth=2, linestyle='dashed')
# ax2.vlines(2012, 0.025, 0.225, colors='black', linewidth=2, linestyle='dashed')
ax2.set_xlabel('Time/ms', fontsize=24)
ax2.set_ylabel('Amplitude/V', fontsize=24)

ax3 = fig.add_subplot(gs[2:3,:])
ax3.set_title(title[k+2], fontsize=24)
ax3.tick_params(axis='both', labelsize=20)
for i in range(5):
    if code_series[i,k+2] == 0:
        ax3.plot(np.arange(301)+i*300, mean_data[k+2, 112+i*300:413+i*300], color='tab:blue')
    elif code_series[i,k+2] == 1:
        ax3.plot(np.arange(301)+i*300, mean_data[k+2, 112+i*300:413+i*300], color='tab:orange')
# ax3.plot(np.arange(112), mean_data[k+2, :112], color='black')
# ax3.plot(np.arange(89)+2012, mean_data[k+2, -89:], color='black')
# for i in range(4):
#     ax3.plot(np.arange(100)+412+i*400, mean_data[k+2, 412+i*400:512+i*400], color='black')
ax3.vlines(0, 0.025, 0.225, colors='black', linewidth=2, linestyle='dashed')
ax3.vlines(300, 0.025, 0.225, colors='black', linewidth=2, linestyle='dashed')
ax3.vlines(600, 0.025, 0.225, colors='black', linewidth=2, linestyle='dashed')
ax3.vlines(900, 0.025, 0.225, colors='black', linewidth=2, linestyle='dashed')
ax3.vlines(1200, 0.025, 0.225, colors='black', linewidth=2, linestyle='dashed')
# ax3.vlines(1212, 0.025, 0.225, colors='black', linewidth=2, linestyle='dashed')
# ax3.vlines(1312, 0.025, 0.225, colors='black', linewidth=2, linestyle='dashed')
# ax3.vlines(1612, 0.025, 0.225, colors='black', linewidth=2, linestyle='dashed')
# ax3.vlines(1712, 0.025, 0.225, colors='black', linewidth=2, linestyle='dashed')
# ax3.vlines(2012, 0.025, 0.225, colors='black', linewidth=2, linestyle='dashed')
ax3.set_xlabel('Time/ms', fontsize=24)
ax3.set_ylabel('Amplitude/V', fontsize=24)

ax4 = fig.add_subplot(gs[3:,:])
ax4.set_title(title[k+3], fontsize=24)
ax4.tick_params(axis='both', labelsize=20)
for i in range(5):
    if code_series[i,k+3] == 0:
        ax4.plot(np.arange(301)+i*300, mean_data[k+3, 112+i*300:413+i*300], color='tab:blue')
    elif code_series[i,k+3] == 1:
        ax4.plot(np.arange(301)+i*300, mean_data[k+3, 112+i*300:413+i*300], color='tab:orange')
# ax4.plot(np.arange(112), mean_data[k+3, :112], color='black', linewidth=3)
# ax4.plot(np.arange(89)+2012, mean_data[k+3, -89:], color='black', linewidth=3)
# for i in range(4):
#     ax4.plot(np.arange(100)+412+i*400, mean_data[k+3, 412+i*400:512+i*400], color='black')
ax4.vlines(0, 0.025, 0.225, colors='black', linewidth=2, linestyle='dashed')
ax4.vlines(300, 0.025, 0.225, colors='black', linewidth=2, linestyle='dashed')
ax4.vlines(600, 0.025, 0.225, colors='black', linewidth=2, linestyle='dashed')
ax4.vlines(900, 0.025, 0.225, colors='black', linewidth=2, linestyle='dashed')
ax4.vlines(1200, 0.025, 0.225, colors='black', linewidth=2, linestyle='dashed')
# ax4.vlines(1212, 0.025, 0.225, colors='black', linewidth=2, linestyle='dashed')
# ax4.vlines(1312, 0.025, 0.225, colors='black', linewidth=2, linestyle='dashed')
# ax4.vlines(1612, 0.025, 0.225, colors='black', linewidth=2, linestyle='dashed')
# ax4.vlines(1712, 0.025, 0.225, colors='black', linewidth=2, linestyle='dashed')
# ax4.vlines(2012, 0.025, 0.225, colors='black', linewidth=2, linestyle='dashed')
ax4.set_xlabel('Time/ms', fontsize=24)
ax4.set_ylabel('Amplitude/V', fontsize=24)

plt.tight_layout()
plt.show()
plt.savefig(r'C:\Users\Administrator\Desktop\photocell_noInterval_7', dpi=600)

# %%
