# -*- coding: utf-8 -*-
"""
For little Ding
Created on Wed Dec 18 14:45:31 2019
Data preprocessing:
    (1)load data from .cnt file to .mat file
    (2)fitering data
    
Continuously updating...
@author: Brynhildr
"""

# %% load 3rd modules
import os
import scipy.io as io

import numpy as np
from numpy import newaxis as NA

import mne
from mne import Epochs
from mne.io import concatenate_raws
from mne.filter import filter_data

# %% if there are multiple files in the same folder
# load data 
filepath = r'D:\Documents\医学工程与转化医学研究院\研究生课题\同行工作'
subjectlist = ['三个电极-镍钛-距离较近']  # can be multiple

# splicing files' names
filefolders = []
for subindex in subjectlist:
    filefolder = os.path.join(filepath, subindex)
    filefolders.append(filefolder)
filelist = []
for filefolder in filefolders:
    for file in os.listdir(filefolder):
        filefullpath = os.path.join(filefolder, file)
        filelist.append(filefullpath)

# splicing raw objects
raw_cnts = []
for file in filelist:
    # montage = mne.channels.make_standard_montage('standard_1020')
    raw_cnt = mne.io.read_raw_cnt(file, eog=['HEO', 'VEO'], emg=['EMG'], ecg=['EKG'], preload=True, verbose=False)
    raw_cnts.append(raw_cnt)
raw = concatenate_raws(raw_cnts)

# release RAM
del raw_cnts, file, filefolder, filefolders, filefullpath, filelist
del filepath, subindex, subjectlist, raw_cnt

# extract events information
events, evnets_id = mne.events_from_annotations(raw)

# drop channels (unnecessary)
# drop_chans = ['M1', 'M2']
# pick useful EEG channels, exclude EMG, Stim, EOG and drop_chans
picks = mne.pick_types(raw.info, emg=False, eeg=True, stim=False, eog=False)
picks_ch_names = [raw.ch_names[i] for i in picks]  # layout picked channels' names

# preparation for extracting data
# baseline = (-0.2, 0)  # time period for baseline processing
tmin, tmax = -1, 1.5    # set the time range
sfreq = 1000          # sampling frequency

# transform raw object into array
n_events = len(evnets_id)
n_trials = int(events.shape[0]/n_events)
n_chans = len(picks)
n_times = int((tmax-tmin)*sfreq + 1)

data = np.zeros((n_events, n_trials, n_chans, n_times))
f_data = np.zeros_like(data)
for i in range(n_events):
    data[i,...] = Epochs(raw, events=events, event_id=i+1, tmin=tmin, tmax=tmax,
                        picks=picks, baseline=None, preload=True).get_data() * 1e6
    # filtering
    f_data[i,...] = filter_data(data[i,...], sfreq=sfreq, l_freq=5, h_freq=20, n_jobs=8, method='fir')

# save data into .mat files
raw_data_path = r'D:\Documents\医学工程与转化医学研究院\研究生课题\同行工作\raw.mat'
io.savemat(raw_data_path, {'data':data, 'chan_info':picks_ch_names})

f_data_path = r'D:\Documents\医学工程与转化医学研究院\研究生课题\同行工作\fir.mat'
io.savemat(f_data_path, {'f_data':f_data, 'chan_info':picks_ch_names})
print('Preprocessing Done.')


# %% if you wang to process a single file
# load data
filename = r'D:\SSVEP\dataset\xwt_bishe\wuqiaoyi\80hz_2.cnt'

# make raw object
raw = mne.io.read_raw_cnt(filename, eog=['HEO', 'VEO'], emg=['EMG'], ecg=['EKG'], preload=True, verbose=False)

del filename  # release RAM

# the following parts are basically the same as above
# extract events information
events, evnets_id = mne.events_from_annotations(raw)

# drop channels (unnecessary)
drop_chans = ['M1', 'M2']
# pick useful EEG channels, exclude EMG, Stim, EOG and drop_chans
picks = mne.pick_types(raw.info, emg=False, eeg=True, stim=False, eog=False, exclude=drop_chans)
picks_ch_names = [raw.ch_names[i] for i in picks]  # layout picked channels' names

# preparation for extracting data
# baseline = (-0.2, 0)  # time period for baseline processing
tmin, tmax = -1, 1    # set the time range
sfreq = 1000          # sampling frequency

# transform raw object into array
n_events = len(evnets_id)
n_trials = int(events.shape[0]/n_events)
n_chans = len(picks)
n_times = int((tmax-tmin)*sfreq + 1)

data = np.zeros((n_events, n_trials, n_chans, n_times))
f_data = np.zeros_like(data)
for i in range(n_events):
    data[i,...] = Epochs(raw, events=events, event_id=i+1, tmin=tmin, tmax=tmax,
                        picks=picks, baseline=None, preload=True).get_data() * 1e6
    # filtering
    f_data[i,...] = filter_data(data[i,...], sfreq=sfreq, l_freq=60, h_freq=90, n_jobs=8, method='fir')

# save data into .mat files
raw_data_path = r'D:\SSVEP\dataset\preprocessed_data\xwt_bishe\wuqiaoyi\raw_80.mat'
io.savemat(raw_data_path, {'data':data, 'chan_info':picks_ch_names})

f_data_path = r'D:\SSVEP\dataset\preprocessed_data\xwt_bishe\wuqiaoyi\f_80.mat'
io.savemat(f_data_path, {'f_data':f_data, 'chan_info':picks_ch_names})
print('Preprocessing Done.')
