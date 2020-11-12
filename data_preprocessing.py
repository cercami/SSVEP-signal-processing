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
filepath = r'G:\岳金明\eeg\岳金明'

subjectlist = ['yjm']

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

#%% define labels
event_id = dict(f60p0=2, f60p1=3, f80p0=4, f80p1=5)
event_id = dict(f32=1)
#baseline = (-0.2, 0)    # define baseline
tmin, tmax = -1., 10.5     # set the time range
sfreq = 1000

# transform raw object into array
n_events = len(event_id)
n_trials = int(events.shape[0] / n_events)
n_chans = len(picks)
n_times = int((tmax - tmin) * sfreq + 1)

#%% pick up data
data = np.zeros((n_events, n_trials, n_chans, n_times))

f32 = Epochs(raw, events=events, event_id=1, tmin=tmin, picks=picks,
               tmax=tmax, baseline=None, preload=True).get_data() * 1e6
#%%
f60p1 = Epochs(raw, events=events, event_id=2, tmin=tmin, picks=picks,
               tmax=tmax, baseline=None, preload=True).get_data() * 1e6
f80p0 = Epochs(raw, events=events, event_id=3, tmin=tmin, picks=picks,
               tmax=tmax, baseline=None, preload=True).get_data() * 1e6
f80p1 = Epochs(raw, events=events, event_id=4, tmin=tmin, picks=picks,
               tmax=tmax, baseline=None, preload=True).get_data() * 1e6

#%% filter data
ff80p0 = filter_data(f80p0, sfreq=sfreq, l_freq=50, h_freq=90, n_jobs=4, method='iir')
ff80p1 = filter_data(f80p1, sfreq=sfreq, l_freq=50, h_freq=90, n_jobs=4, method='iir')
#
ff60p0 = filter_data(f60p0, sfreq=sfreq, l_freq=50, h_freq=90, n_jobs=4, method='iir')
ff60p1 = filter_data(f60p1, sfreq=sfreq, l_freq=50, h_freq=90, n_jobs=4, method='iir')
#%%
ff32 = filter_data(f32, sfreq=sfreq, l_freq=40, h_freq=80, n_jobs=4, method='iir')
#%% find bad trials
data = ff80p0[:,54,1140:2140]
plt.plot(np.mean(data, axis=0))
#
data = ff80p1[:,54,1140:2140]
plt.plot(np.mean(data, axis=0))
#%% delete bad trials
bad = np.where(data > 12)

#%% check abnormal waveform
plt.plot(data[42,0:])

#%% check abnormal FFT
data = np.mean(ff58, axis=0)
plt.psd(data, 1024, 1000)

#%% new task
eeg = io.loadmat(r'F:\SSVEP\dataset\preprocessed_data\weisiwen\raw_data.mat')
data = eeg['raw_data']
chans = eeg['chan_info'].tolist()
del eeg

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

data[2,:,:,:] = ff80p0
data[3,:,:,:] = ff80p1

#%% store raw data
data_path = r'F:\SSVEP\dataset\preprocessed_data\60&80\xiongwentian\iir_50_90.mat'
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
regression_list = ['OLS']
method_list = ['FS']
train_num = [40, 30, 20, 10]

n_events = 2
frequencies = [60, 60, 48, 48]
items = []
for i in combinations('0123', r=n_events):
    items.append(i)

name_list = ['chengqian', 'pangjun', 'mengqiangfan', 'guojiaming']

# SRCA training
for name in range(len(name_list)):
    people = name_list[name]
    eeg = io.loadmat(r'F:\SSVEP\dataset\preprocessed_data\%s\60hz_50_70.mat' %(people))
    for item in items:
        file_num = item[0] + item[1]
        f_data = eeg['f_data'][[eval(item[0]), eval(item[1])], :, :, :]
        chans = eeg['chan_info'].tolist()
        for loop in range(5):                      # loop in cross validation
            regression = regression_list[0]        # loop in regression method
            method = method_list[met]              # loop in SRCA parameters
            ns = train_num[nfile]                  # loop in training trials
            # randomly pick channels for training
            randPick = np.arange(f_data.shape[1])
            np.random.shuffle(randPick)
            for nt in range(5):                    # loop in training times
                model_info = []
                para_alteration = []
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
                    model_chans, para_change = mcee.stepwise_SRCA_fs(srca_chans, mpara,
                    w_i, w_o, sig_i, sig_o, method, regression, freq=freq, phase=phase, sfreq=1000)
                    # refresh data
                    para_alteration.append(para_change)
                    model_info.append(model_chans)
                # save data as .mat file
                data_path = r'F:\SSVEP\realCV\%s\%s\train_%d\loop_%d\srca_%d.mat' %(
                                        people, method, ns, loop, nt)             
                io.savemat(data_path, {'modelInfo': model_info,
                                       'parameter': para_alteration,
                                       'trialInfo': randPick})

#%% Real Cross Validation: CCA
tar_chans = ['PZ ','PO5','PO3','POZ','PO4','PO6','O1 ','OZ ','O2 ']
regressionList = ['OLS']
methodList = ['CCA']
trainNum = [80]

n_events = 2
frequencies = [60, 60, 48, 48]
items = []
for i in combinations('0123', r=n_events):
    items.append(i)

nameList = ['chengqian', 'pangjun', 'mengqiangfan', 'guojiaming']
#nameList = ['mengqiangfan']

# SRCA training
for name in range(len(nameList)):                   # loop in testees
    people = nameList[name]
    eeg = io.loadmat(r'F:\SSVEP\dataset\preprocessed_data\%s\40_70bp.mat' %(people))
    for item in items:
        file_num = item[0] + item[1]
        f_data = eeg['f_data'][[eval(item[0]), eval(item[1])], :, :, :]
        chans = eeg['chan_info'].tolist()
        for loop in range(5):                      # loop in cross validation
            regression = regressionList[0]
            method = methodList[0]
            ns = trainNum[0]
            # randomly pick channels for training
            randPick = np.arange(f_data.shape[1])
            np.random.shuffle(randPick)
            for nt in range(5):                    # loop in data length
                model_info = []
                para_alteration = []
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
                        # find best initial phase
                        if ne == 0:
                            freq = frequencies[eval(item[0])]
                        elif ne == 1:
                            freq = frequencies[eval(item[1])]
                        phase = mcee.template_phase(data=sig_o, freq=freq, step=100, sfreq=1000)
                        mpara = np.mean(mcee.template_corr(data=sig_o, freq=freq, phase=phase, sfreq=1000))
                        # main SRCA process
                        model_chans, para_change = mcee.stepwise_SRCA(srca_chans, mpara,
                        w_i, w_o, sig_i, sig_o, method, regression, freq=freq, phase=phase, sfreq=1000)
                        # refresh data
                        para_alteration.append(para_change)
                        model_info.append(model_chans)
                # save data as .mat file
                data_path = r'F:\SSVEP\realCV\%s\%s\%s\bp_40_70\Cross %s\train_%d\loop_%d\srca_%d.mat' %(
                            people, regression, method, file_num, ns, loop, nt)
                io.savemat(data_path, {'modelInfo': model_info,
                                       'parameter': para_alteration,
                                       'trialInfo': randPick})

#%% Real Cross Validation: SNR
tar_chans = ['PZ ','PO5','PO3','POZ','PO4','PO6','O1 ','OZ ','O2 ']
#regressionList = ['OLS', 'Ridge']
regressionList = ['OLS']
methodList = ['SNR']
trainNum = [80]

n_events = 2
frequencies = [60,60,48,48]
items = []
for i in combinations('0123', r=n_events):
    items.append(i)

nameList = ['chengqian', 'pangjun', 'mengqiangfan', 'guojiaming']

# SRCA training
for name in range(len(nameList)):                          # loop in testees
    people = nameList[name]
    eeg = io.loadmat(r'F:\SSVEP\dataset\preprocessed_data\%s\40_70bp.mat' %(people))
    for item in items:
        file_num = item[0] + item[1]
        f_data = eeg['f_data'][[eval(item[0]), eval(item[1])], :, :, :]
        chans = eeg['chan_info'].tolist()
        for loop in range(5):                                  # loop in cross validation
            regression = regressionList[0]
            method = methodList[0]
            ns = trainNum[0]
            for nt in range(5):                    # loop in data length
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
                data_path = r'F:\SSVEP\realCV\%s\%s\%s\bp_40_70\Cross %s\train_%d\loop_%d\srca_%d.mat' %(
                            people, regression, method, file_num, ns, loop, nt)
                io.savemat(data_path, {'modelInfo': model_info,
                                       'parameter': para_alteration,
                                       'trialInfo': randPick})
