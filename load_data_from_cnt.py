# -*- coding: utf-8 -*-
"""
Created on Sat Nov 30 14:50:22 2019

@author: Brynhildr
"""

#%% load 3rd-part module
import os
import numpy as np
import mne
import scipy.io as io
from mne.io import concatenate_raws
from mne import Epochs, pick_types, find_events
from mne.baseline import rescale
import time

#%% timing
start = time.clock()


# %% load data
filepath = r'F:\SSVEP\dataset'

subjectlist = ['weisiwen']

filefolders = []
for subindex in subjectlist:
    filefolder = os.path.join(filepath, subindex)
    filefolders.append(filefolder)

filelist = []
for filefolder in filefolders:
    for file in os.listdir(filefolder):
        filefullpath = os.path.join(filefolder, file)
        filelist.append(filefullpath)

raw_cnts = []
for file in filelist:
    montage = mne.channels.read_montage('standard_1020')
    raw_cnt = mne.io.read_raw_cnt(file, montage=montage,
            eog=['HEO', 'VEO'], emg=['EMG'], ecg=['EKG'],
            preload=True, verbose=False, stim_channel='True')
    # misc=['CB1', 'CB2', 'M1', 'M2'],
    raw_cnts.append(raw_cnt)

raw = concatenate_raws(raw_cnts)

del raw_cnts, file, filefolder, filefolders, filefullpath, filelist
del filepath, subindex, subjectlist


#%% preprocessing
events = mne.find_events(raw, output='onset')

# drop channels
drop_chans = ['FC6', 'FT8', 'C6', 'T8', 'TP7', 'CP6', 'M1', 'M2']

picks = mne.pick_types(raw.info, emg=False, eeg=True, stim=False, eog=False,
                       exclude=drop_chans)
picks_ch_names = [raw.ch_names[i] for i in picks]  # layout picked chans' name

# define labels
event_id = dict(f8=1, f10=2, f15=3)

#baseline = (-0.2, 0)    # define baseline
tmin, tmax = -3., 3.5    # set the time range
sfreq = 1000


#%% store data into array
n_stims = int(len(event_id))
n_trials = int(events.shape[0] / n_stims)
n_chans = int(64 - len(drop_chans))
n_times = int((tmax - tmin) * sfreq + 1)

data = np.zeros((n_stims, n_trials, n_chans, n_times))
for i in range(len(event_id)):

    epochs = Epochs(raw, events=events, event_id=i+1, tmin=tmin, picks=picks,
                    tmax=tmax, baseline=None, preload=True)
    data[i,:,:,:] = epochs.get_data()  # get the 3D array of data
    # (n_trials, n_chans, n_times)
    del epochs
    
del raw, picks, i, n_stims, n_trials, n_chans, n_times


#%% store data into .mat file
data_path = r'F:\SSVEP\dataset\preprocessed_data\weisiwen\raw_data.mat'
io.savemat(data_path, {'raw_data':data})
del data

data_path = r'F:\SSVEP\dataset\preprocessed_data\weisiwen\chan_info.mat'
io.savemat(data_path, {'chan_info':picks_ch_names})


#%% timing
end = time.clock()
print('Running time: ' + str(end - start) + 's')
