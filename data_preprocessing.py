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

#%% load 3rd-part module
import os
import numpy as np
#import mne
import scipy.io as io
#from mne.io import concatenate_raws
#from mne import Epochs, pick_types, find_events
#from mne.baseline import rescale
#from mne.filter import filter_data
import copy
import srca
import matplotlib.pyplot as plt

#%% load data
filepath = r'F:\SSVEP\dataset'

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
drop_chans = ['M1', 'M2']

picks = mne.pick_types(raw.info, emg=False, eeg=True, stim=False, eog=False,
                       exclude=drop_chans)
picks_ch_names = [raw.ch_names[i] for i in picks]  # layout picked chans' name

# define labels
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
    data[i,:,:,:] = epochs.get_data()  # (n_trials, n_chans, n_times)
    del epochs
    
del raw, picks, i, n_stims, n_trials, n_chans, n_times
del drop_chans, event_id, events, tmax, tmin

# store raw data
data_path = r'F:\SSVEP\dataset\preprocessed_data\wuqiaoyi\raw & filtered data\raw_data.mat'
io.savemat(data_path, {'raw_data':data, 'chan_info':picks_ch_names})

# filtering
data = data[:2,:,:,:]
n_events = data.shape[0]
n_trials = data.shape[1]
n_chans = data.shape[2]
n_times = data.shape[3]
f_data = np.zeros((n_events, n_trials, n_chans, n_times))
for i in range(n_events):
    f_data[i,:,:,:] = filter_data(data[i,:,:,:], sfreq=sfreq, l_freq=50,
                      h_freq=70, n_jobs=4)
del i, data

#%% find & delete bad trials
f_data = np.delete(f_data, [10,33,34,41,42], axis=1)

# store fitered data
data_path = r'F:\SSVEP\dataset\preprocessed_data\brynhildr\50_70_bp.mat'
io.savemat(data_path, {'f_data':f_data, 'chan_info':picks_ch_names})

# release RAM
del data_path, f_data, n_chans, n_events, n_times, n_trials, picks_ch_names, sfreq

#%% real te tr
# pick channels from parietal and occipital areas
tar_chans = ['PZ ','PO5','PO3','POZ','PO4','PO6','O1 ','OZ ','O2 ']

n_events = 2
n_trials = 115
n_chans = len(tar_chans)
n_test = 60

# mcee optimization
for nfile in range(4):
    if nfile == 0:  
        ns = 55    
    elif nfile == 1:  
        ns = 40       
    elif nfile == 2:  
        ns = 30       
    elif nfile == 3:
        ns = 25   
    for nt in range(5):
        model_info = []
        snr_alteration = []
        srca_sig = np.zeros((n_events, n_test, n_chans, int(100+nt*100)))
        for ntc in range(len(tar_chans)):
            for ne in range(2):  # ne for n_events
                target_channel = tar_chans[ntc]
                sfreq = 1000
                # load local data (extract from .cnt file)
                eeg = io.loadmat(r'F:\SSVEP\dataset\preprocessed_data\wuqiaoyi\50_70_bp.mat')
                f_data = eeg['f_data'][ne,:,:,2000:4140] * 1e6
                w = f_data[:ns,:,0:1000]
                signal_data = f_data[:ns,:,1140:int(1240+nt*100)]
                chans = eeg['chan_info'].tolist() 
                del eeg
                # variables initialization-model
                w_o = w[:,chans.index(target_channel),:]
                w_temp = copy.deepcopy(w)
                w_i = np.delete(w_temp, chans.index(target_channel), axis=1)
                del w_temp, w
                # variables initialization-signal
                sig_o = signal_data[:,chans.index(target_channel),:]
                sig_temp = copy.deepcopy(signal_data)
                sig_i = np.delete(sig_temp, chans.index(target_channel), axis=1)
                del sig_temp, signal_data
                # config chans & parameter info
                srca_chans = copy.deepcopy(chans)
                del srca_chans[chans.index(target_channel)]
                para = srca.snr_time(sig_o)
                mpara = np.mean(para)
                del para
                # use stepwise method to find channels
                model_chans, para_change = srca.stepwise_SRCA(chans=srca_chans,
                        msnr=mpara, w=w_i, w_target=w_o, signal_data=sig_i,
                        data_target=sig_o, method='Ridge', alpha=0.5,
                        l1_ratio=0.5)
                snr_alteration.append(para_change)
                del para_change, w_i, w_o, sig_i, sig_o, srca_chans, mpara 
                # preparation for test dataset
                w_te = f_data[-n_test:,:,0:1000]
                w_o_te = w_te[:,chans.index(target_channel),:]
                signal_data_te = f_data[-n_test:,:,1140:int(1240+nt*100)]
                sig_o_te = signal_data_te[:,chans.index(target_channel),:]
                # pick channels chosen from stepwise
                w_i_te = np.zeros((w_te.shape[0], len(model_chans), w_te.shape[2]))
                sig_i_te = np.zeros((signal_data_te.shape[0], len(model_chans), signal_data_te.shape[2]))
                for nc in range(len(model_chans)):
                    w_i_te[:,nc,:] = w_te[:,chans.index(model_chans[nc]),:]
                    sig_i_te[:,nc,:] = signal_data_te[:,chans.index(model_chans[nc]),:]
                del nc, w_te, signal_data_te
                # srca main process
                w_es_s, w_ex_s = srca.SRCA_lm_extract(model_input=w_i_te,
                    model_target=w_o_te, data_input=sig_i_te, data_target=sig_o_te,
                    method='OLS', alpha=0.5, l1_ratio=0.5, mode='b')
                del w_es_s
                # save optimized data
                srca_sig[ne,:,ntc,:] = w_ex_s
                model_info.append(model_chans)
                del w_ex_s, model_chans, sig_i_te, sig_o_te, w_i_te, w_o_te
        data_path = r'F:\SSVEP\lasso\lasso+snr\real1_%d\mcee_%d.mat' %(int(nfile+1), nt)             
        io.savemat(data_path, {'srca_sig': srca_sig,
                           'chan_info': tar_chans,
                           'model_info': model_info,
                           'parameter': snr_alteration})


#%% common trca
acc_srca_tr = np.zeros((4,5,10))
for nfile in range(4):
    for nt in range(5):
        if nfile == 0:
            eeg = io.loadmat(r'F:\SSVEP\lasso\lasso+snr\real1_1\mcee_%d.mat' %(nt))
        elif nfile == 1:
            eeg = io.loadmat(r'F:\SSVEP\lasso\lasso+snr\real1_2\mcee_%d.mat' %(nt))
        elif nfile == 2:
            eeg = io.loadmat(r'F:\SSVEP\lasso\lasso+snr\real1_3\mcee_%d.mat' %(nt))
        elif nfile == 3:
            eeg = io.loadmat(r'F:\SSVEP\lasso\lasso+snr\real1_4\mcee_%d.mat' %(nt))
        data = eeg['srca_sig']
        print('Data length:' + str((nt+1)*100) + 'ms')
        del eeg
        # cross validation
        acc = []
        N = 5
        print('running TRCA program...')
        for cv in range(N):
            a = int(cv * (data.shape[1]/N))
            tr_data = data[:,a:a+int(data.shape[1]/N),:,:]
            te_data = copy.deepcopy(data)
            te_data = np.delete(te_data, [a+i for i in range(tr_data.shape[1])], axis=1)
            acc_temp = srca.pure_trca(train_data=tr_data, test_data=te_data)
            acc.append(np.sum(acc_temp))
            del acc_temp
            print(str(cv+1) + 'th cv complete')
        acc = np.array(acc)/(te_data.shape[1]*2)
        acc_srca_tr[nfile,:,nt] = acc
        del acc

acc_srca_te = np.zeros((4,5,10))
for nfile in range(4):
    for nt in range(5):
        if nfile == 0:
            eeg = io.loadmat(r'F:\SSVEP\lasso\lasso+snr\real1_1\mcee_%d.mat' %(nt))
        elif nfile == 1:
            eeg = io.loadmat(r'F:\SSVEP\lasso\lasso+snr\real1_2\mcee_%d.mat' %(nt))
        elif nfile == 2:
            eeg = io.loadmat(r'F:\SSVEP\lasso\lasso+snr\real1_3\mcee_%d.mat' %(nt))
        elif nfile == 3:
            eeg = io.loadmat(r'F:\SSVEP\lasso\lasso+snr\real1_4\mcee_%d.mat' %(nt))
        data = eeg['srca_sig']
        print('Data length:' + str((nt+1)*100) + 'ms')
        del eeg
        # cross validation
        acc = []
        N = 5
        print('running TRCA program...')
        for cv in range(N):
            a = int(cv * (data.shape[1]/N))
            tr_data = data[:,a:a+int(data.shape[1]/N),:,:]
            te_data = copy.deepcopy(data)
            te_data = np.delete(te_data, [a+i for i in range(tr_data.shape[1])], axis=1)
            acc_temp = srca.pure_trca(train_data=te_data, test_data=tr_data)
            acc.append(np.sum(acc_temp))
            del acc_temp
            print(str(cv+1) + 'th cv complete')
        acc = np.array(acc)/(tr_data.shape[1]*2)
        acc_srca_te[nfile,:,nt] = acc
        del acc