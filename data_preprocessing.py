# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 14:45:31 2019
Data preprocessing:
    (1)load data from .cnt file to .mat file
    (2)fitering data
    (3)
    
Continuously updating...
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
from mne.filter import filter_data

#%% load data
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

#%% transform raw object into array
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
del drop_chans, event_id, events, tmax, tmin

# store raw data
data_path = r'F:\SSVEP\dataset\preprocessed_data\weisiwen\raw_data.mat'
io.savemat(data_path, {'raw_data':data,
                       'chan_info':picks_ch_names})

#%% filtering
n_events = data.shape[0]
n_trials = data.shape[1]
n_chans = data.shape[2]
n_times = data.shape[3]

f_data = np.zeros((n_events, n_trials, n_chans, n_times))
for i in range(n_events):
    f_data[i,:,:,:] = filter_data(data[i,:,:,:], sfreq=sfreq, l_freq=5,
                      h_freq=90, n_jobs=6)

# release RAM
del i, data

# store fitered data
data_path = r'F:\SSVEP\dataset\preprocessed_data\weisiwen\f_data.mat'
io.savemat(data_path, {'f_data':f_data,
                       'chan_info':picks_ch_names})

# release RAM
del data_path, f_data, n_chans, n_events, n_times, n_trials, picks_ch_names, sfreq

#%% prevent from pressing 'F5'
stop here!

#%% mcee optimization
# pick channels from parietal and occipital areas
tar_chans = ['PZ ','PO5','PO3','POZ','PO4','PO6','O1 ','OZ ','O2 ']

n_events = 3
n_trials = 100
n_chans = len(tar_chans)
n_times = 1640

# (n_events, n_trials, n_chans, n_times)
mcee_sig = np.zeros((n_events, n_trials, n_chans, n_times))

# mcee optimization
for nt in range(11):
    # stepwise
    for ntc in range(len(tar_chans)):
        for nf in range(3):
            freq = nf  # 0 for 8Hz, 1 for 10Hz, 2 for 15Hz
            target_channel = tar_chans[ntc]
            # load local data (extract from .cnt file)
            eeg = io.loadmat(r'F:\SSVEP\dataset\preprocessed_data\weisiwen\f_data.mat')
            f_data = eeg['f_data'][freq,:,:,2000:3640] * 1e6
            w = f_data[:,:,0:1000]
            if nt == 0:  # 500ms
                signal_data = f_data[:,:,1140:1640]
                ep = 1640
            elif nt == 1:  # 400ms
                signal_data = f_data[:,:,1140:1540]
                ep = 1540
            elif nt == 2:  # 300ms
                signal_data = f_data[:,:,1140:1440]
                ep = 1440
            elif nt == 3:  # 200ms
                signal_data = f_data[:,:,1140:1340]
                ep = 1340
            elif nt == 4:  # 180ms
                signal_data = f_data[:,:,1140:1320]
                ep = 1320
            elif nt == 5:  # 160ms
                signal_data = f_data[:,:,1140:1300]
                ep = 1300
            elif nt == 6:  # 140ms
                signal_data = f_data[:,:,1140:1280]
                ep = 1280
            elif nt == 7:  # 120ms
                signal_data = f_data[:,:,1140:1260]
                ep = 1260
            elif nt == 8:  # 100ms
                signal_data = f_data[:,:,1140:1240]
                ep = 1240
            elif nt == 9:  # 80ms
                signal_data = f_data[:,:,1140:1220]
                ep = 1220
            elif nt == 10:  # 60ms
                signal_data = f_data[:,:,1140:1200]
                ep = 1200
                
            chans = eeg['chan_info'].tolist() 
            del eeg

            # basic information
            sfreq = 1000

            # variables initialization
            w_o = w[:,chans.index(target_channel),:]
            w_temp = copy.deepcopy(w)
            w_i = np.delete(w_temp, chans.index(target_channel), axis=1)
            del w_temp

            sig_o = signal_data[:,chans.index(target_channel),:]
            f_sig_o = f_data[:,chans.index(target_channel),:]
            sig_temp = copy.deepcopy(signal_data)
            sig_i = np.delete(sig_temp, chans.index(target_channel), axis=1)
            del sig_temp

            mcee_chans = copy.deepcopy(chans)
            del mcee_chans[chans.index(target_channel)]

            snr = mcee.snr_time(sig_o)
            msnr = np.mean(snr)
  
            # use stepwise method to find channels
            model_chans, snr_change = mcee.stepwise_MCEE(chans=mcee_chans,
                    msnr=msnr, w=w, w_target=w_o, signal_data=sig_i, data_target=sig_o)
            del snr_change
            del w_i, sig_i, mcee_chans, snr, msnr

            # pick channels chosen from stepwise
            w_i = np.zeros((w.shape[0], len(model_chans), w.shape[2]))
            f_sig_i = np.zeros((f_data.shape[0], len(model_chans), f_data.shape[2]))

            for nc in range(len(model_chans)):
                w_i[:,nc,:] = w[:,chans.index(model_chans[nc]),:]
                f_sig_i[:,nc,:] = f_data[:,chans.index(model_chans[nc]),:]
            del nc

            # mcee main process
            rc, ri, r2 = SPF.mlr_analysis(w_i, w_o)
            w_es_s, w_ex_s = SPF.sig_extract_mlr(rc, f_sig_i, f_sig_o, ri)
            del rc, ri, r2, w_es_s

            # save optimized data
            mcee_sig[nf,:,ntc,:] = w_ex_s
            del w_ex_s, model_chans, f_sig_i, sig_o, w_i, w_o, w, signal_data, f_sig_o
            
    data_path = r'F:\SSVEP\dataset\preprocessed_data\weisiwen\b_140ms\mcee_5chan_%d.mat'%(nt)
    io.savemat(data_path, {'mcee_sig': mcee_sig,
                           'chan_info': tar_chans})