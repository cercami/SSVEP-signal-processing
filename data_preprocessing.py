# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 14:45:31 2019
Data preprocessing:
    (1)load data from .cnt file to .mat file
    (2)fitering data
    (3)mcee optimization
    
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
import copy
import mcee
import signal_processing_function as SPF
import matplotlib.pyplot as plt

#%% load data
filepath = r'I:\SSVEP\dataset'

subjectlist = ['wuqiaoyi']

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

# preprocessing
events = mne.find_events(raw, output='onset')

# drop channels
#drop_chans = ['FC6', 'FT8', 'C6', 'T8', 'TP7', 'CP6', 'M1', 'M2']
drop_chans = ['M1', 'M2']

picks = mne.pick_types(raw.info, emg=False, eeg=True, stim=False, eog=False,
                       exclude=drop_chans)
picks_ch_names = [raw.ch_names[i] for i in picks]  # layout picked chans' name

# define labels
#event_id = dict(f8=1, f10=2, f15=3)
event_id = dict(f60p0=1, f60p1=2, f40p0=3, f40p1=4)

#baseline = (-0.2, 0)    # define baseline
tmin, tmax = -2., 2.5    # set the time range
sfreq = 1000

# transform raw object into array
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
data_path = r'I:\SSVEP\dataset\preprocessed_data\wuqiaoyi\raw & filtered data\raw_data.mat'
io.savemat(data_path, {'raw_data':data,
                       'chan_info':picks_ch_names})

# filtering
data = data[:2,:,:,:]
#
n_events = data.shape[0]
n_trials = data.shape[1]
n_chans = data.shape[2]
n_times = data.shape[3]

f_data = np.zeros((n_events, n_trials, n_chans, n_times))
for i in range(n_events):
    f_data[i,:,:,:] = filter_data(data[i,:,:,:], sfreq=sfreq, l_freq=50,
                      h_freq=90, n_jobs=4)

# release RAM
del i, data

#%%
x1 = f_data[0,:,:,2000:3140]*1e6
x1 = x1[:,[45,51,52,53,54,55,58,59,60],:]
x2 = f_data[1,:,:,2000:3140]*1e6
x2 = x2[:,[45,51,52,53,54,55,58,59,60],:]
#%%
plt.plot(x2[:,7,:].T)
#plt.plot(np.mean(x2[:,7,:], axis=0))
#%%
error = []
for i in range(120):
    for j in range(9):
        if np.max(x2[i,j,:])>15:
            error.append(str(i)+'t'+str(j)+'c')
    del j
del i
#%%
x1 = np.delete(x1, [34,41,42], axis=0)
x2 = np.delete(x2, [34,41,42], axis=0)
#%%
x = np.zeros((2,117,9,1140))
x[0,:,:,:] = x1
x[1,:,:,:] = x2
#%%
w_p, fs = SPF.welch_p(x[:,:,7,:], sfreq=1000, fmin=40, fmax=90, n_fft=2048,
                      n_overlap=0, n_per_seg=2048)
#
plt.plot(fs[1,1,:], np.mean(w_p[1,:,:], axis=0), color='tab:blue')
data_path = r'I:\SSVEP\dataset\preprocessed_data\wuqiaoyi\mcee data\psd_ori.mat'
io.savemat(data_path, {'freqs':fs,
                       'psd':w_p})
#%%
# delete bad trials
f_data = np.delete(f_data, [34,41,42], axis=1)

# store fitered data
data_path = r'I:\SSVEP\dataset\preprocessed_data\wuqiaoyi\raw & filtered data\50_90_bp.mat'
io.savemat(data_path, {'f_data':f_data,
                       'chan_info':picks_ch_names})

# release RAM
del data_path, f_data, n_chans, n_events, n_times, n_trials, picks_ch_names, sfreq

#%% prevent from pressing 'F5'
stop here!

#%% mcee optimization
# pick channels from parietal and occipital areas
tar_chans = ['PZ ','PO5','PO3','POZ','PO4','PO6','O1 ','OZ ','O2 ']

n_events = 2
n_trials = 117
n_chans = len(tar_chans)
n_times = 2140

# (n_events, n_trials, n_chans, n_times)
mcee_sig = np.zeros((n_events, n_trials, n_chans, n_times))

# mcee optimization
for nfile in range(2):
    for nt in range(8):
        model_info = []
        # stepwise
        for ntc in range(len(tar_chans)):
            for ne in range(2):  # ne for n_events
                target_channel = tar_chans[ntc]
                # load local data (extract from .cnt file)
                #if nfile == 0:  # 30-90Hz
                    #eeg = io.loadmat(r'I:\SSVEP\dataset\preprocessed_data\wuqiaoyi\raw & filtered data\30_90_bp.mat')
                eeg = io.loadmat(r'I:\SSVEP\dataset\preprocessed_data\wuqiaoyi\raw & filtered data\50_90_bp.mat')
                if nfile == 0:  # begin from 140ms
                    bp = 1140
                elif nfile == 1:  # begin from 60ms
                    bp = 1060
                    
                f_data = eeg['f_data'][ne,:,:,2000:4140] * 1e6
                w = f_data[:,:,0:1000]
                if nt == 0:  # 60ms
                    signal_data = f_data[:,:,bp:1200]
                elif nt == 1:  # 80ms
                    signal_data = f_data[:,:,bp:1220]
                elif nt == 2:  # 100ms
                    signal_data = f_data[:,:,bp:1240]
                elif nt == 3:  # 200ms
                    signal_data = f_data[:,:,bp:1340]
                elif nt == 4:  # 300ms
                    signal_data = f_data[:,:,bp:1440]
                elif nt == 5:  # 400ms
                    signal_data = f_data[:,:,bp:1540]
                elif nt == 6:  # 500ms
                    signal_data = f_data[:,:,bp:1640]
                elif nt == 7:  # 1000ms
                    signal_data = f_data[:,:,bp:2140]
                
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
                mcee_sig[ne,:,ntc,:] = w_ex_s
                model_info.append(model_chans)
                del w_ex_s, model_chans, f_sig_i, sig_o, w_i, w_o, w, signal_data, f_sig_o
        
        if nfile == 0:  # begin from 60ms
            data_path = r'I:\SSVEP\dataset\preprocessed_data\wuqiaoyi\mcee data\begin from 140ms\50_90_bp\mcee_%d.mat'%(nt)
        elif nfile == 1:  # begin from 140ms
            data_path = r'I:\SSVEP\dataset\preprocessed_data\wuqiaoyi\mcee data\begin from 60ms\50_90_bp\mcee_%d.mat'%(nt)           
   
        # save model channels' information
        model_info_0 = model_info[::2]
        model_info_1 = model_info[1::2]
        #model_info_2 = model_info[2::4]
        #model_info_3 = model_info[3::4]
        
        io.savemat(data_path, {'mcee_sig': mcee_sig,
                               'chan_info': tar_chans,
                               'model_info_60p0': model_info_0,
                               'model_info_60p1': model_info_1})
            #'model_info_40p0': model_info_2, 'model_info_40p1': model_info_3})
        del model_info, model_info_0, model_info_1 
        #del model_info_2, model_info_3
 
#%% test part       
# check data
#eeg = io.loadmat(r'I:\SSVEP\dataset\preprocessed_data\wuqiaoyi\mcee data\begin from 140ms\50_90_bp\+1s.mat')
eeg = io.loadmat(r'I:\SSVEP\dataset\preprocessed_data\wuqiaoyi\raw & filtered data\50_90_bp.mat')
#data = eeg['mcee_sig']
data = eeg['f_data'][:,:,[45,51,52,53,54,55,58,59,60],:]*1e6
#data = np.delete(data, [41,42], axis=1)
#chan_info = eeg['chan_info']
#model_info_0 = [x.T.tolist() for x in eeg['model_info_60p0'].flatten().tolist()]
#model_info_1 = [x.T.tolist() for x in eeg['model_info_60p1'].flatten().tolist()]
#model_info_2 = eeg['model_info_2']
#model_info_3 = eeg['model_info_3']
#del eeg
#%%
w_p, fs = SPF.welch_p(data[:,:,7,1140:], sfreq=1000, fmin=40, fmax=90,
                      n_fft=1024, n_overlap=0, n_per_seg=1024)
plt.title('origin data', fontsize=20)
plt.tick_params(axis='both', labelsize=16)
plt.plot(fs[1,1,:], np.mean(w_p[0,:,:], axis=0), label='0 phase',
         color='tab:orange', linewidth=1)
plt.plot(fs[1,1,:], np.mean(w_p[1,:,:], axis=0), label='pi phase',
         color='tab:blue', linewidth=1)
plt.ylim(0,0.14)
plt.legend(loc='best', fontsize=16)
plt.show()

#%%
plt.plot(np.mean(data[0,:,7,1140:], axis=0))
#%% check waveform
ne = 1
bp = 1140
ep = 1640
plt.title('60Hz & pi phase (origin)', fontsize=20)
plt.plot(np.mean(data[ne,:,0,bp:ep], axis=0).T, label='PZ')
plt.plot(np.mean(data[ne,:,1,bp:ep], axis=0).T, label='PO5')
plt.plot(np.mean(data[ne,:,2,bp:ep], axis=0).T, label='PO3')
plt.plot(np.mean(data[ne,:,3,bp:ep], axis=0).T, label='POZ')
plt.plot(np.mean(data[ne,:,4,bp:ep], axis=0).T, label='PO4')
plt.plot(np.mean(data[ne,:,5,bp:ep], axis=0).T, label='PO6')
plt.plot(np.mean(data[ne,:,6,bp:ep], axis=0).T, label='O1')
plt.plot(np.mean(data[ne,:,7,bp:ep], axis=0).T, label='OZ')
plt.plot(np.mean(data[ne,:,8,bp:ep], axis=0).T, label='O2')
plt.vlines(0, np.min(np.mean(data[ne,:,:,bp:ep], axis=0)),
           np.max(np.mean(data[ne,:,:,bp:ep], axis=0)),
           color='black', linestyle='dashed', label='140ms')
plt.xlabel('Time/ms', fontsize=16)
plt.ylabel('Amplitude/uV', fontsize=16)
plt.tick_params(axis='both', labelsize=16)
plt.legend(loc='best', fontsize=14)

#%% cut data into pieces
eeg = io.loadmat(r'I:\SSVEP\dataset\preprocessed_data\wuqiaoyi\mcee data\begin from 140ms\50_90_bp\+1s.mat')
data = eeg['mcee_sig'][:2,:,:,1140:]
chan_info = eeg['chan_info']
model_info_0 = [x.T.tolist() for x in eeg['model_info_60p0'].flatten().tolist()]
model_info_1 = [x.T.tolist() for x in eeg['model_info_60p1'].flatten().tolist()]
del eeg