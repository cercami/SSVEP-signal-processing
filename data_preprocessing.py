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
import mne
import scipy.io as io
from mne.io import concatenate_raws
from mne import Epochs
from mne.filter import filter_data
import copy
#import srca
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
#import fp_growth as fpg
import mcee
import seaborn as sns

#%% load data
filepath = r'F:\SSVEP\dataset\60&48'

subjectlist = ['mengqiangfan']

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
    montage = mne.channels.make_standard_montage('standard_1020')
    raw_cnt = mne.io.read_raw_cnt(file, eog=['HEO', 'VEO'], emg=['EMG'], ecg=['EKG'],
            preload=True, verbose=False)
    raw_cnts.append(raw_cnt)

raw = concatenate_raws(raw_cnts)

del raw_cnts, file, filefolder, filefolders, filefullpath, filelist
del filepath, subindex, subjectlist, raw_cnt

# preprocessing
events, events_id = mne.events_from_annotations(raw)

# drop channels
drop_chans = ['M1', 'M2']

picks = mne.pick_types(raw.info, emg=False, eeg=True, stim=False, eog=False,
                       exclude=drop_chans)
picks_ch_names = [raw.ch_names[i] for i in picks]  # layout picked chans' name

# define labels
event_id = dict(f60p0=1, f60p1=2, f48p0=3, f48p1=4)

#baseline = (-0.2, 0)    # define baseline
tmin, tmax = -1., 1.5     # set the time range
sfreq = 1000

# transform raw object into array
n_events = len(event_id)
n_trials = int(events.shape[0] / n_events)
n_chans = len(picks)
n_times = int((tmax - tmin) * sfreq + 1)

data = np.zeros((n_events, n_trials, n_chans, n_times))

f60p0 = Epochs(raw, events=events, event_id=1, tmin=tmin, picks=picks,
               tmax=tmax, baseline=None, preload=True).get_data() * 1e6
f60p1 = Epochs(raw, events=events, event_id=2, tmin=tmin, picks=picks,
               tmax=tmax, baseline=None, preload=True).get_data() * 1e6
f48p0 = Epochs(raw, events=events, event_id=3, tmin=tmin, picks=picks,
               tmax=tmax, baseline=None, preload=True).get_data() * 1e6
f48p1 = Epochs(raw, events=events, event_id=4, tmin=tmin, picks=picks,
               tmax=tmax, baseline=None, preload=True).get_data() * 1e6

#%% filter data
ff48p1 = filter_data(f48p1, sfreq=sfreq, l_freq=40, h_freq=90, n_jobs=4)
ff48p0 = filter_data(f48p0, sfreq=sfreq, l_freq=40, h_freq=90, n_jobs=4)
#%%
ff60p0 = filter_data(f60p0, sfreq=sfreq, l_freq=40, h_freq=90, n_jobs=4)
ff60p1 = filter_data(f60p1, sfreq=sfreq, l_freq=40, h_freq=90, n_jobs=4)

#%% find bad trials
data = ff60p1[:,59,1140:1640]
#plt.plot(data.T)

#%% delete bad trials
bad = np.where(data > 12)

#%% check abnormal waveform
plt.plot(data[42,0:])

#%% check abnormal FFT
plt.psd(data[0,:], 512, 1000)

#%% delete bad trials
#f48p0 = np.delete(f48p0, [3,6,24,127])
f48p1 = np.delete(f48p1, [133, 132], axis=0)
f60p1 = np.delete(f60p1, [132], axis=0)

#%%
ff60p1 = np.delete(ff60p1, [132], axis=0)
ff48p1 = np.delete(ff48p1, [132,133], axis=0)

#%%
fig = plt.figure(figsize=(16,9))
gs = GridSpec(2,2,figure=fig)
sns.set(style='whitegrid')

ax1 = fig.add_subplot(gs[:1,:1])
ax1.set_title('Photocell Measurement: 48Hz 0 phase', fontsize=24)
ax1.tick_params(axis='both', labelsize=20)
ax1.plot(f48p0[:, 34, 1000:2000].T)
ax1.set_xlabel('Time/ms', fontsize=22)
ax1.set_ylabel('Amplitude/uV', fontsize=22)
#ax1.legend(loc='upper right', fontsize=18)

ax2 = fig.add_subplot(gs[:1,1:])
ax2.set_title('Photocell Measurement: 48Hz pi phase', fontsize=24)
ax2.tick_params(axis='both', labelsize=20)
ax2.plot(f48p1[:, 34, 1000:2000].T)
ax2.set_xlabel('Time/ms', fontsize=22)
ax2.set_ylabel('Amplitude/uV', fontsize=22)
#ax2.legend(loc='upper right', fontsize=18)

ax3 = fig.add_subplot(gs[1:,:1])
ax3.set_title('Photocell Measurement: 60Hz 0 phase', fontsize=24)
ax3.tick_params(axis='both', labelsize=20)
ax3.plot(f60p0[:, 34, 1000:2000].T)
ax3.set_xlabel('Time/ms', fontsize=22)
ax3.set_ylabel('Amplitude/uV', fontsize=22)

ax4 = fig.add_subplot(gs[1:,1:])
ax4.set_title('Photocell Measurement: 60Hz pi phase', fontsize=24)
ax4.tick_params(axis='both', labelsize=20)
ax4.plot(f60p1[:, 34, 1000:2000].T)
ax4.set_xlabel('Time/ms', fontsize=22)
ax4.set_ylabel('Amplitude/uV', fontsize=22)

fig.tight_layout()
plt.show()
plt.savefig(r'C:\Users\brynh\Desktop\photocell-split.png', dpi=600)

#%%
data = np.zeros((4, 144, 62, 2501))
data[0,:,:,:] = ff60p0
data[1,:,:,:] = ff60p1

#%%
data[2,:,:,:] = ff48p0
data[3,:,:,:] = ff48p1

#%% store raw data
data_path = r'F:\SSVEP\dataset\preprocessed_data\mengqiangfan\40_90bp.mat'
io.savemat(data_path, {'f_data':data, 'chan_info':picks_ch_names})

#%% other test
pz = []
po5 = []
po3 = []
poz = []
po4 = []
po6 = []
o1 = []
oz = []
o2 = []

for i in range(10):
    eeg = io.loadmat(r'F:\SSVEP\SRCA data\begin：140ms\OLS\SNR\55_60\srca_%d.mat' %(i))
    exec("model_%d = eeg['model_info'].flatten().tolist()" %(i))
del i, eeg

j = 0
for i in range(10):
    exec("pz.append(model_%d[0+j].tolist())" %(i))
    exec("po5.append(model_%d[1+j].tolist())" %(i))
    exec("po3.append(model_%d[2+j].tolist())" %(i))
    exec("poz.append(model_%d[3+j].tolist())" %(i))
    exec("po4.append(model_%d[4+j].tolist())" %(i))
    exec("po6.append(model_%d[5+j].tolist())" %(i))
    exec("o1.append(model_%d[6+j].tolist())" %(i))
    exec("oz.append(model_%d[7+j].tolist())" %(i))
    exec("o2.append(model_%d[8+j].tolist())" %(i))
del model_0, model_1, model_2, model_3, model_4, model_5, model_6, model_7, model_8, model_9

#%%
eeg = io.loadmat(r'F:\SSVEP\dataset\preprocessed_data\wuqiaoyi\50_70_bp.mat')
ori_data = eeg['f_data'][:,-60:,[45,51,52,53,54,55,58,59,60],2140:2340]*1e6
chan_info = eeg['chan_info'].tolist()

#%%
fig = plt.figure(figsize=(16,12))
gs = GridSpec(4,4, figure=fig)

ax1 = fig.add_subplot(gs[:2,:])
ax1.tick_params(axis='both', labelsize=20)
ax1.set_title('Original signal waveform (OZ)', fontsize=24)
ax1.set_xlabel('Time/ms', fontsize=20)
ax1.set_ylabel('Amplitude/uV', fontsize=20)
ax1.plot(np.mean(ori_data[0,:,7,:], axis=0), label='0 phase')
ax1.plot(np.mean(ori_data[1,:,7,:], axis=0), label='pi phase')
ax1.legend(loc='upper left', fontsize=16)

ax2 = fig.add_subplot(gs[2:,:])
ax2.tick_params(axis='both', labelsize=20)
ax2.set_title('Fisher score optimized signal waveform (OZ)', fontsize=24)
ax2.set_xlabel('Time/ms', fontsize=20)
ax2.set_ylabel('Amplitude/uV', fontsize=20)
ax2.plot(np.mean(srca_data[0,:,7,:], axis=0), label='0 phase')
ax2.plot(np.mean(srca_data[1,:,7,:], axis=0), label='pi phase')
ax2.legend(loc='upper left', fontsize=16)

plt.show()

#%% FP-Growth
if __name__ == '__main__':
    '''
    Call function 'find_frequent_itemsets()' to form frequent items
    '''
    frequent_itemsets = fpg.find_frequent_itemsets(oz, minimum_support=5,
                                                   include_support=True)
    #print(type(frequent_itemsets))
    result = []
    # save results from generator into list
    for itemset, support in frequent_itemsets:  
        result.append((itemset, support))
    # ranking
    result = sorted(result, key=lambda i: i[0])
    print('FP-Growth complete!')

#%% compressed fg-growth result
compressed_result = []
number = []
for i in range(len(result)):
    if len(result[i][0]) > 3:
        compressed_result.append(result[i][0])
        number.append(result[i][1])
del i

#%% Real Cross Validation: Fisher Score
tar_chans = ['PZ ','PO5','PO3','POZ','PO4','PO6','O1 ','OZ ','O2 ']
regressionList = ['OLS']
methodList = ['FS']
trainNum = [40, 30, 20, 10]
n_events = 2
n_chans = len(tar_chans)

# load in data
eeg = io.loadmat(r'F:\SSVEP\dataset\preprocessed_data\guojiaming\60hz_50_70.mat')
f_data = eeg['f_data']
chans = eeg['chan_info'].tolist()
del eeg

# SRCA training
for loop in range(5):                                  # loop in cross validation
    for reg in range(len(regressionList)):             # loop in regression method
        regression = regressionList[reg]
        for met in range(len(methodList)):             # loop in SRCA parameters
            method = methodList[met]
            for nfile in range(len(trainNum)):         # loop in training trials
                ns = trainNum[nfile]
                for nt in range(5):                    # loop in training times
                    model_info = []
                    para_alteration = []
                    # randomly pick channels for training
                    randPick = np.arange(f_data.shape[1])
                    np.random.shuffle(randPick)
                    w = f_data[:, randPick[:ns], :, :1000]
                    signal_data = f_data[:, randPick[:ns], :, 1140:int(100*nt+1240)]
                    for ntc in range(len(tar_chans)):  # loop in target channels
                        # prepare model data
                        target_channel = tar_chans[ntc]
                        # w for rest-state data
                        w_o = w[:,:,chans.index(target_channel),:]
                        w_temp = copy.deepcopy(w)
                        w_i = np.delete(w_temp, chans.index(target_channel), axis=2)
                        del w_temp
                        # sig for mission-state data
                        sig_o = signal_data[:,:,chans.index(target_channel),:]
                        sig_temp = copy.deepcopy(signal_data)
                        sig_i = np.delete(sig_temp, chans.index(target_channel), axis=2)
                        del sig_temp
                        # prepare for infomation record
                        srca_chans = copy.deepcopy(chans)
                        del srca_chans[chans.index(target_channel)]
                        mpara = np.mean(mcee.fisher_score(sig_o))
                        # main SRCA process
                        model_chans, para_change = mcee.stepwise_SRCA_fs(srca_chans,
                            mpara, w_i, w_o, sig_i, sig_o, regression)
                        # refresh data
                        para_alteration.append(para_change)
                        model_info.append(model_chans)
                    # save data as .mat file
                    data_path = r'F:\SSVEP\realCV\guojiaming\%s\train_%d\loop_%d\srca_%d.mat' %(
                                        method, ns, loop+1, nt)             
                    io.savemat(data_path, {'modelInfo': model_info,
                                           'parameter': para_alteration,
                                           'trialInfo': randPick})

#%% Real Cross Validation: SNR
tar_chans = ['PZ ','PO5','PO3','POZ','PO4','PO6','O1 ','OZ ','O2 ']
regressionList = ['OLS']
methodList = ['SNR']
trainNum = [40, 30, 20, 10]

n_events = 2

# load in data (n_events, n_trials, n_chans, n_times)
eeg = io.loadmat(r'F:\SSVEP\dataset\preprocessed_data\chengqian\60hz_50_70.mat')
f_data = eeg['f_data']
chans = eeg['chan_info'].tolist()
del eeg

# SRCA training
for loop in range(5):                                  # loop in cross validation
    for reg in range(len(regressionList)):             # loop in regression method
        regression = regressionList[reg]
        for met in range(len(methodList)):             # loop in SRCA parameters
            method = methodList[met]
            for nfile in range(len(trainNum)):         # loop in training trials
                ns = trainNum[nfile]
                for nt in range(5):                    # loop in data length: 100-500ms
                    model_info = []
                    para_alteration = []
                    # randomly pick channels for training
                    randPick = np.arange(f_data.shape[1])
                    np.random.shuffle(randPick) 
                    for ntc in range(len(tar_chans)):  # loop in target channels
                        for ne in range(n_events):
                            # prepare model data
                            target_channel = tar_chans[ntc]
                            # w for rest-state data
                            w = f_data[ne, randPick[:ns], :, :1000]
                            w_o = w[:,chans.index(target_channel),:]
                            w_temp = copy.deepcopy(w)
                            w_i = np.delete(w_temp, chans.index(target_channel), axis=1)
                            del w_temp
                            # sig for mission-state data
                            signal_data = f_data[ne, randPick[:ns], :, 1140:int(100*nt+1240)]
                            sig_o = signal_data[:,chans.index(target_channel),:]
                            sig_temp = copy.deepcopy(signal_data)
                            sig_i = np.delete(sig_temp, chans.index(target_channel), axis=1)
                            del sig_temp
                            # prepare for infomation record
                            srca_chans = copy.deepcopy(chans)
                            del srca_chans[chans.index(target_channel)]
                            mpara = np.mean(mcee.snr_time(sig_o))
                            # main SRCA process
                            model_chans, para_change = mcee.stepwise_SRCA(srca_chans,
                                mpara, w_i, w_o, sig_i, sig_o, method, regression)
                            # refresh data
                            para_alteration.append(para_change)
                            model_info.append(model_chans)
                    # save data as .mat file
                    data_path = r'F:\SSVEP\realCV\%s\%s\train_%d\loop_%d\srca_%d.mat' %(regression,
                                        method, ns, loop, nt)             
                    io.savemat(data_path, {'modelInfo': model_info,
                                           'parameter': para_alteration,
                                           'trialInfo': randPick})

#%% make fs alteration chart
tar_chans = ['PZ ','PO5','PO3','POZ','PO4','PO6','O1 ','OZ ','O2 ']
eeg = io.loadmat(r'F:\SSVEP\dataset\preprocessed_data\pangjun\60hz_50_70.mat')
f_data = eeg['f_data']
chans = eeg['chan_info'].tolist()
trainNum = [10, 20, 30, 40]
tarChanIndex = [45,51,52,53,54,55,58,59,60]
del eeg
fs_alter_tr = np.zeros((4,5,9,5))  # (n_trians, n_loops, n_chans, n_times)
fs_alter_te = np.zeros_like(fs_alter_tr)
fs_ori_tr = np.zeros_like(fs_alter_tr)
fs_ori_te = np.zeros_like(fs_alter_tr)
for ntr in range(4):
    ns = trainNum[ntr]
    print('training trials: ' + str(ns))
    for nl in range(5):
        for nti in range(5):
            # load srca model & extract information
            srca = io.loadmat(r'F:\SSVEP\realCV\pangjun\FS\train_%d\loop_%d\srca_%d.mat' %(
                ns, nl, nti))
            tempChans = srca['modelInfo'].flatten().tolist()
            trainTrials = np.mean(srca['trialInfo'], axis=0).astype(int)
            modelChans = []
            for i in range(9):
                modelChans.append(tempChans[i].tolist())
            del tempChans, srca, i
            # extract two dataset
            trainData = f_data[:, trainTrials[:ns], :, :1240+100*nti]
            testData = f_data[:, trainTrials[-80:], :, :1240+100*nti]
            del trainTrials
            # apply SRCA model
            srca_trData = np.zeros((2, ns, 9, 100+nti*100))
            srca_trData[0, :, :, :] = mcee.apply_SRCA(trainData[0, :, :, :],
                tar_chans, modelChans, chans)
            srca_trData[1, :, :, :] = mcee.apply_SRCA(trainData[1, :, :, :],
                tar_chans, modelChans, chans)
            srca_teData = np.zeros((2, 80, 9, 100+nti*100))
            srca_teData[0, :, :, :] = mcee.apply_SRCA(testData[0, :, :, :],
                tar_chans, modelChans, chans)
            srca_teData[1, :, :, :] = mcee.apply_SRCA(testData[1, :, :, :],
                tar_chans, modelChans, chans)
            # compute each channels' fs alteration
            for nc in range(9):
                xtr = mcee.fisher_score(trainData[:, :, tarChanIndex[nc], 1140:])
                ytr = mcee.fisher_score(srca_trData[:, :, nc, :])
                ztr = (np.mean(ytr) - np.mean(xtr))/np.mean(xtr) * 100
                
                xte = mcee.fisher_score(testData[:, :, tarChanIndex[nc], 1140:])
                yte = mcee.fisher_score(srca_teData[:, :, nc, :])
                zte = (np.mean(yte) - np.mean(xte))/np.mean(xte) * 100
                
                fs_alter_tr[ntr, nl, nc, nti] = ztr
                fs_alter_te[ntr, nl, nc, nti] = zte
                fs_ori_tr[ntr, nl, nc, nti] = np.mean(xtr)
                fs_ori_te[ntr, nl, nc, nti] = np.mean(xte)
        print('loop ' + str(nl) + ' complete')
      
tar_chans = ['PZ ','PO5','PO3','POZ','PO4','PO6','O1 ','OZ ','O2 ']
eeg = io.loadmat(r'F:\SSVEP\dataset\preprocessed_data\chengqian\60hz_50_70.mat')
f_data = eeg['f_data']
chans = eeg['chan_info'].tolist()
trainNum = [10, 20, 30, 40]
tarChanIndex = [45,51,52,53,54,55,58,59,60]
del eeg
fs_alter_tr2 = np.zeros((4,5,9,5))  # (n_trians, n_loops, n_chans, n_times)
fs_alter_te2 = np.zeros_like(fs_alter_tr2)
fs_ori_tr2 = np.zeros_like(fs_alter_tr2)
fs_ori_te2 = np.zeros_like(fs_alter_tr2)
for ntr in range(4):
    ns = trainNum[ntr]
    print('training trials: ' + str(ns))
    for nl in range(5):
        for nti in range(5):
            # load srca model & extract information
            srca = io.loadmat(r'F:\SSVEP\realCV\chengqian\FS\train_%d\loop_%d\srca_%d.mat' %(
                ns, nl, nti))
            tempChans = srca['modelInfo'].flatten().tolist()
            trainTrials = np.mean(srca['trialInfo'], axis=0).astype(int)
            modelChans = []
            for i in range(9):
                modelChans.append(tempChans[i].tolist())
            del tempChans, srca, i
            # extract two dataset
            trainData = f_data[:, trainTrials[:ns], :, :1240+100*nti]
            testData = f_data[:, trainTrials[-80:], :, :1240+100*nti]
            del trainTrials
            # apply SRCA model
            srca_trData = np.zeros((2, ns, 9, 100+nti*100))
            srca_trData[0, :, :, :] = mcee.apply_SRCA(trainData[0, :, :, :],
                tar_chans, modelChans, chans)
            srca_trData[1, :, :, :] = mcee.apply_SRCA(trainData[1, :, :, :],
                tar_chans, modelChans, chans)
            srca_teData = np.zeros((2, 80, 9, 100+nti*100))
            srca_teData[0, :, :, :] = mcee.apply_SRCA(testData[0, :, :, :],
                tar_chans, modelChans, chans)
            srca_teData[1, :, :, :] = mcee.apply_SRCA(testData[1, :, :, :],
                tar_chans, modelChans, chans)
            # compute each channels' fs alteration
            for nc in range(9):
                xtr = mcee.fisher_score(trainData[:, :, tarChanIndex[nc], 1140:])
                ytr = mcee.fisher_score(srca_trData[:, :, nc, :])
                ztr = (np.mean(ytr) - np.mean(xtr))/np.mean(xtr) * 100
                
                xte = mcee.fisher_score(testData[:, :, tarChanIndex[nc], 1140:])
                yte = mcee.fisher_score(srca_teData[:, :, nc, :])
                zte = (np.mean(yte) - np.mean(xte))/np.mean(xte) * 100
                
                fs_alter_tr2[ntr, nl, nc, nti] = ztr
                fs_alter_te2[ntr, nl, nc, nti] = zte
                fs_ori_tr2[ntr, nl, nc, nti] = np.mean(xtr)
                fs_ori_te2[ntr, nl, nc, nti] = np.mean(xte)
        print('loop ' + str(nl) + ' complete')

tar_chans = ['PZ ','PO5','PO3','POZ','PO4','PO6','O1 ','OZ ','O2 ']
eeg = io.loadmat(r'F:\SSVEP\dataset\preprocessed_data\liutuo\60hz_50_70.mat')
f_data = eeg['f_data']
chans = eeg['chan_info'].tolist()
trainNum = [10, 20, 30, 40]
tarChanIndex = [45,51,52,53,54,55,58,59,60]
del eeg
fs_alter_tr3 = np.zeros((4,5,9,5))  # (n_trians, n_loops, n_chans, n_times)
fs_alter_te3 = np.zeros_like(fs_alter_tr3)
fs_ori_tr3 = np.zeros_like(fs_alter_tr3)
fs_ori_te3 = np.zeros_like(fs_alter_tr3)
for ntr in range(4):
    ns = trainNum[ntr]
    print('training trials: ' + str(ns))
    for nl in range(5):
        for nti in range(5):
            # load srca model & extract information
            srca = io.loadmat(r'F:\SSVEP\realCV\liutuo\FS\train_%d\loop_%d\srca_%d.mat' %(
                ns, nl, nti))
            tempChans = srca['modelInfo'].flatten().tolist()
            trainTrials = np.mean(srca['trialInfo'], axis=0).astype(int)
            modelChans = []
            for i in range(9):
                modelChans.append(tempChans[i].tolist())
            del tempChans, srca, i
            # extract two dataset
            trainData = f_data[:, trainTrials[:ns], :, :1240+100*nti]
            testData = f_data[:, trainTrials[-80:], :, :1240+100*nti]
            del trainTrials
            # apply SRCA model
            srca_trData = np.zeros((2, ns, 9, 100+nti*100))
            srca_trData[0, :, :, :] = mcee.apply_SRCA(trainData[0, :, :, :],
                tar_chans, modelChans, chans)
            srca_trData[1, :, :, :] = mcee.apply_SRCA(trainData[1, :, :, :],
                tar_chans, modelChans, chans)
            srca_teData = np.zeros((2, 80, 9, 100+nti*100))
            srca_teData[0, :, :, :] = mcee.apply_SRCA(testData[0, :, :, :],
                tar_chans, modelChans, chans)
            srca_teData[1, :, :, :] = mcee.apply_SRCA(testData[1, :, :, :],
                tar_chans, modelChans, chans)
            # compute each channels' fs alteration
            for nc in range(9):
                xtr = mcee.fisher_score(trainData[:, :, tarChanIndex[nc], 1140:])
                ytr = mcee.fisher_score(srca_trData[:, :, nc, :])
                ztr = (np.mean(ytr) - np.mean(xtr))/np.mean(xtr) * 100
                
                xte = mcee.fisher_score(testData[:, :, tarChanIndex[nc], 1140:])
                yte = mcee.fisher_score(srca_teData[:, :, nc, :])
                zte = (np.mean(yte) - np.mean(xte))/np.mean(xte) * 100
                
                fs_alter_tr3[ntr, nl, nc, nti] = ztr
                fs_alter_te3[ntr, nl, nc, nti] = zte
                fs_ori_tr3[ntr, nl, nc, nti] = np.mean(xtr)
                fs_ori_te3[ntr, nl, nc, nti] = np.mean(xte)
        print('loop ' + str(nl) + ' complete')