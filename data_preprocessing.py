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
from mne import Epochs, pick_types, find_events
from mne.baseline import rescale
from mne.filter import filter_data
import copy
#import srca
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import fp_growth as fpg
import mcee
#import signal_processing_function as SPF

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

# store fitered dataw
data_path = r'F:\SSVEP\dataset\preprocessed_data\brynhildr\50_70_bp.mat'
io.savemat(data_path, {'f_data':f_data, 'chan_info':picks_ch_names})

# release RAM
del data_path, f_data, n_chans, n_events, n_times, n_trials, picks_ch_names, sfreq
    
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

#%%
compressed_result = []
number = []
for i in range(len(result)):
    if len(result[i][0]) > 3:
        compressed_result.append(result[i][0])
        number.append(result[i][1])
del i

#%%
delta = np.zeros((4,10,18))
summ = np.zeros((4,10,18))
for i in range(4):
    for j in range(10):
        eeg = io.loadmat(r'F:\SSVEP\ols\ols+snr\real1_%d\mcee_%d.mat' %(i+1,j))
        parameter_ols = eeg['parameter'].flatten().tolist()
        para_ols = []
        for k in range(len(parameter_ols)):
            para_ols.append(np.max(parameter_ols[k]))   
        eeg = io.loadmat(r'F:\SSVEP\ridge\ridge+snr\real1_%d\mcee_%d.mat' %(i+1,j))
        parameter_ri = eeg['parameter'].flatten().tolist()
        para_ri = []
        for l in range(len(parameter_ri)):
            para_ri.append(np.max(parameter_ri[l]))
        para_ols = np.array(para_ols)
        para_ri = np.array(para_ri)
        delta[i,j,:] = para_ri - para_ols
        summ[i,j,:] = para_ri + para_ols
ratio = (np.mean((delta+summ)/(summ-delta))-1)*100

#%% real te tr (obsolete)
# pick channels from parietal and occipital areas
tar_chans = ['PZ ','PO5','PO3','POZ','PO4','PO6','O1 ','OZ ','O2 ']
regressionList = ['OLS', 'Ridge']  # choose multi-linear regression method
methodList = ['FS']  # use fisher score as the output of target function
trainNum = [55, 40, 30, 25]  # use different training trials

n_events = 2
n_trials = 115
n_chans = len(tar_chans)
n_test = 60  # number of test trails
n_times = 2140 # 1000ms(rest state) + 140ms(latency) + 1000ms(mission state)

for reg in range(len(regressionList)):
    regression = regressionList[reg]
    for met in range(len(methodList)):
        method = methodList[met]
        for nfile in range(len(trainNum)):
            ns = trainNum[nfile]
            for nt in range(5):  # 100ms-500ms
                model_info = []  # SRCA channels
                para_alteration = []  # parameters' alteration
                for ntc in range(len(tar_chans)):
                    target_channel = tar_chans[ntc]
                    # load .mat data: (n_events, n_trials, n_chans, n_times)
                    eeg = io.loadmat(r'F:\SSVEP\dataset\preprocessed_data\wuqiaoyi\50_70_bp.mat')
                    f_data = eeg['f_data'][:,:,:,1000:3140] * 1e6  # filtered data
                    w = f_data[:,:ns,:,:1000]  # 1s rest state data
                    signal_data = f_data[:,:ns,:,1140:int(1240+nt*100)]
                    # all channels' names, list | e.g: [..., 'O1 ', 'OZ', 'O2', ...]
                    chans = eeg['chan_info'].tolist()  
                    del eeg  # release RAM
                    
                    # w_o for model output (single-channel data)
                    w_o = w[:,:,chans.index(target_channel),:]
                    # w_i for model input (multi-channel data)
                    w_temp = copy.deepcopy(w)
                    w_i = np.delete(w_temp, chans.index(target_channel), axis=2)
                    del w_temp
                    
                    # the same as before
                    sig_o = signal_data[:,:,chans.index(target_channel),:]
                    sig_temp = copy.deepcopy(signal_data)
                    sig_i = np.delete(sig_temp, chans.index(target_channel), axis=2)
                    del sig_temp, signal_data
                    
                    srca_chans = copy.deepcopy(chans)
                    del srca_chans[chans.index(target_channel)]
                    mpara = np.mean(mcee.fisher_score(sig_o))
                    # main function
                    model_chans, para_change = mcee.stepwise_SRCA_fs(srca_chans,
                            mpara, w, w_o, sig_i, sig_o, regression)
                    # refresh data
                    para_alteration.append(para_change)
                    model_info.append(model_chans)
                # save data as .mat
                data_path = r'F:\SSVEP\SRCA data\real_cv\%s\%s\%d_60\srca_%d.mat' %(regression, method, ns, nt)             
                io.savemat(data_path, {'model_info': model_info, 'parameter': para_alteration})


#%% TRCA/eTRCA for SRCA/origin data (Online Mode)
tarChans = ['PZ ','PO5','PO3','POZ','PO4','PO6','O1 ','OZ ','O2 ']
trainNum = [10, 20, 30, 40]
acc_srca_trca = np.zeros((len(trainNum), 5, 5))  # (trainNum, cv, n_length)
acc_srca_etrca = np.zeros_like(acc_srca_trca)
acc_ori_trca = np.zeros_like(acc_srca_trca)
acc_ori_etrca = np.zeros_like(acc_srca_trca)

eeg = io.loadmat(r'F:\SSVEP\dataset\preprocessed_data\wuqiaoyi\50_70_bp.mat')
f_data = eeg['f_data'][:, :, [45,51,52,53,54,55,58,59,60], 1000:2640] * 1e6
chans = eeg['chan_info'].tolist()
nTargetChan = f_data.shape[2]
del eeg

for tn in range(len(trainNum)):
    ns = trainNum[tn]
    for cv in range(5):  # cross-validation in training
        print('CV: %d turn...' %(cv+1))
        for nt in range(5):
            print('Data length: %d00ms' %(nt+1))
            srcaModel = io.loadmat(r'F:\SSVEP\realCV\OLS\SNR\train_%d\loop_%d\srca_%d.mat' 
                                   %(tn, cv, nt))
            # extract model info used in SRCA
            modelInfo = srcaModel['model_info'].flatten().tolist()
            modelChans = []
            for i in range(len(modelInfo)):
                modelChans.append(modelInfo[i].tolist())
            del modelInfo
            # error shouldn't exist hhh
            modelChans = modelChans[nt*nTargetChan*2:(nt+1)*nTargetChan*2]
            # extract trials info used in Cross Validation
            # another name error (trail) hhh
            trainTrial = np.mean(srcaModel['trail_info'], axis=0).astype(int)
            del srcaModel
            # extract origin data with correct trials & correct length
            trainData = f_data[:, trainTrial[:ns], :, :1240+nt*100]
            testData = f_data[:, trainTrial[-60:], :, :1240+nt*100]
            del trainTrial
            # target identification main process
            accTRCA = mcee.srca_trca(trainData, testData, tarChans, modelChans, chans,
                                 'OLS', 1140)
            acceTRCA = mcee.e_srca_trca(trainData, testData, tarChans, modelChans, chans,
                                 'OLS', 1140)
            accOriTRCA = mcee.pure_trca(trainData, testData)
            accOrieTRCA = mcee.e_trca(trainData, testData)
            # re-arrange accuracy data
            accTRCA = np.sum(accTRCA)/(testData.shape[1]*2)
            acceTRCA = np.sum(acceTRCA)/(testData.shape[1]*2)
            accOriTRCA = np.sum(accOriTRCA)/(testData.shape[1]*2)
            accOrieTRCA = np.sum(accOrieTRCA)/(testData.shape[1]*2)
            # save accuracy data
            acc_srca_trca[tn, cv, nt] = accTRCA
            acc_srca_etrca[tn, cv, nt] = acceTRCA
            acc_ori_trca[tn, cv, nt] = accOriTRCA
            acc_ori_etrca[tn, cv, nt] = accOrieTRCA
            del accTRCA, acceTRCA, accOriTRCA, accOrieTRCA
        print(str(cv+1) + 'th cross-validation complete!\n')
    print(str(ns) + ' training trials complete!\n')

    
#%% ensemble trca (obsolete)
tar_chans = ['PZ ','PO5','PO3','POZ','PO4','PO6','O1 ','OZ ','O2 ']
regressionList = ['OLS']
#regressionList = ['OLS', 'Ridge']
#methodList = ['SNR', 'Corr']
methodList = ['SNR']
trainNum = [55]
#trainNum = [55, 40, 30, 25]

acc_srca_tr = np.zeros((len(regressionList), len(methodList), len(trainNum), 5))
for reg in range(len(regressionList)):
    regression = regressionList[reg]
    for met in range(len(methodList)):
        method = methodList[met]
        for nfile in range(len(trainNum)):
            ns = trainNum[nfile]
            for nt in range(5):
                # extract model info used in SRCA
                eeg = io.loadmat(r'F:\SSVEP\SRCA data\begin：140ms-0.4pi\%s\%s\%d_60\srca_%d.mat'
                                     %(regression, method, ns, nt))
                model = eeg['model_info'].flatten().tolist()
                model_chans = []
                for i in range(18):
                    model_chans.append(model[i].tolist())
                print('Data length:' + str((nt+1)*100) + 'ms')
                del model, eeg       
                # extract origin data
                eeg = io.loadmat(r'F:\SSVEP\dataset\preprocessed_data\wuqiaoyi\50_70_bp.mat')
                temp_tr = eeg['f_data'][:, :ns, :, 1000:int(2240+nt*100)]*1e6
                tr_data = np.zeros_like(temp_tr)
                tr_data[0,:,:,:] = temp_tr[0,:,:,:]
                tr_data[1,:,:,:] = eeg['f_data'][1, :ns, :, 1000:int(2240+nt*100)]*1e6
                temp_te = eeg['f_data'][:, -60:, :, 1000:int(2240+nt*100)]*1e6
                te_data = np.zeros_like(temp_te)
                te_data[0,:,:,:] = temp_te[0,:,:,:]
                te_data[1,:,:,:] = eeg['f_data'][1, -60:, :, 1000:int(2240+nt*100)]*1e6
                chans = eeg['chan_info'].tolist()
                del eeg, temp_tr, temp_te
                # cross validation
                acc = []
                acc_temp = mcee.srca_trca(train_data=tr_data, test_data=te_data,
                    tar_chans=tar_chans, model_chans=model_chans, chans=chans,
                    regression=regression, sp=1140)
                acc.append(np.sum(acc_temp))
                del acc_temp
                acc = np.array(acc)/(te_data.shape[1]*2)
                acc_srca_tr[reg, met, nfile, nt] = acc
                del acc
                            
#%%
regressionList = ['OLS']
trainNum = [55, 40, 30, 25]
for reg in range(len(regressionList)):
    for nfile in range(len(trainNum)):
        for nt in range(10):
            eeg = io.loadmat(r'F:\SSVEP\dataset\preprocessed_data\wuqiaoyi\50_70_bp.mat')
            train_data = eeg['f_data'][:, -60:, :, 1000:2300+nt*100]*1e6
            chans = eeg['chan_info'].tolist()
            del eeg
            tar_chans = ['PZ ','PO5','PO3','POZ','PO4','PO6','O1 ','OZ ','O2 ']
            eeg = io.loadmat(r'F:\SSVEP\SRCA data\begin：200ms\%s\FS\%d_60\srca_%d'
                             %(regressionList[reg], trainNum[nfile], nt))
            model = eeg['model_info'].flatten().tolist()
            model_chans = []
            for i in range(len(model)):
                model_chans.append(model[i].tolist())
            del eeg, i, model
            n_events = train_data.shape[0]
            n_trains = train_data.shape[1]
            n_chans = len(tar_chans)
            n_times = train_data.shape[-1] - 1200
            model_sig = np.zeros((n_events, n_trains, n_chans, n_times))
            for ntc in range(len(tar_chans)):
                target_channel = tar_chans[ntc]
                model_chan = model_chans[ntc]
                w_i = np.zeros((n_events, n_trains, len(model_chan), 1000))
                sig_i = np.zeros((n_events, n_trains, len(model_chan), n_times))
                for nc in range(len(model_chan)):
                    w_i[:, :, nc, :] = train_data[:, :, chans.index(model_chan[nc]), :1000]
                    sig_i[:, :, nc, :] = train_data[:, :, chans.index(model_chan[nc]), 1200:]
                del nc
                w_o = train_data[:, :, chans.index(target_channel), :1000]
                sig_o = train_data[:, :, chans.index(target_channel), 1200:]
                w_i = np.swapaxes(w_i, 1, 2)
                sig_i = np.swapaxes(sig_i, 1, 2)
                for ne in range(n_events):
                    w_ex_s = mcee.mlr(w_i[ne, :, :, :], w_o[ne, :, :],
                          sig_i[ne, :, :, :], sig_o[ne, :, :], regressionList[reg])
                    model_sig[ne, :, ntc, :] = w_ex_s
                del ne
            del ntc, model_chan, w_i, w_o, sig_i, sig_o, w_ex_s
            data_label = [0 for i in range(60)]
            data_label += [1 for i in range(60)]
            srca_data = np.zeros((120, 9, n_times))
            srca_data[:60, :, :] = model_sig[0,:,:,:]
            srca_data[60:, :, :] = model_sig[1,:,:,:]
            srca_data = np.swapaxes(srca_data, 0, -1)
            del model_sig, train_data
            data_path = r'F:\SSVEP\DCPM_Pre\begin：200ms\%s\%d_60\t%d.mat' %(regressionList[reg],
                    trainNum[nfile], nt)
            io.savemat(data_path, {'srca_data':srca_data, 'label':data_label})
            print('Method:{}--Data length:{}ms--{} samples complete!'.format(regressionList[reg],
                    100*(nt+1), trainNum[nfile]))


#%% Real Cross Validation: Fisher Score
tar_chans = ['PZ ','PO5','PO3','POZ','PO4','PO6','O1 ','OZ ','O2 ']
#regressionList = ['OLS', 'Ridge']
regressionList = ['OLS']
methodList = ['FS']
#trainNum = [50, 40, 30, 20]
trainNum = [30]

n_events = 2
n_trials = 115
n_chans = len(tar_chans)
n_test = 60
n_times = 2140

# load in data
eeg = io.loadmat(r'F:\SSVEP\dataset\preprocessed_data\wuqiaoyi\50_70_bp.mat')
f_data = eeg['f_data'][:,:,:,1000:3140] * 1e6
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
                            mpara, w, w_o, sig_i, sig_o, regression)
                        # refresh data
                        para_alteration.append(para_change)
                        model_info.append(model_chans)
                    # save data as .mat file
                    data_path = r'F:\SSVEP\realCV\%s\%s\train_%d\loop_%d\srca_%d.mat' %(regression,
                                        method, ns, loop+1, nt)             
                    io.savemat(data_path, {'modelInfo': model_info,
                                           'parameter': para_alteration,
                                           'trialInfo': randPick})

#%% Real Cross Validation: SNR
tar_chans = ['PZ ','PO5','PO3','POZ','PO4','PO6','O1 ','OZ ','O2 ']
regressionList = ['OLS', 'Ridge']
#regressionList = ['OLS']
methodList = ['SNR']
trainNum = [50, 40, 30, 20]

n_events = 2
n_trials = 115
n_chans = len(tar_chans)
n_test = 60
n_times = 2140

# load in data
eeg = io.loadmat(r'F:\SSVEP\dataset\preprocessed_data\wuqiaoyi\50_70_bp.mat')
f_data = eeg['f_data'][:,:,:,1000:3140] * 1e6
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
                                mpara, w, w_o, sig_i, sig_o, method, regression)
                            # refresh data
                            para_alteration.append(para_change)
                            model_info.append(model_chans)
                    # save data as .mat file
                    data_path = r'F:\SSVEP\realCV\%s\%s\train_%d\loop_%d\srca_%d.mat' %(regression,
                                        method, ns, loop+1, nt)             
                    io.savemat(data_path, {'modelInfo': model_info,
                                           'parameter': para_alteration,
                                           'trialInfo': randPick})


#%%
eeg = io.loadmat(r'F:\SSVEP\dataset\preprocessed_data\wuqiaoyi\50_70_bp.mat')
for nt in range(10):
    data = eeg['f_data'][:,-60:,[45,51,52,53,54,55,58,59,60],2140:2240+nt*100]*1e6
    temp = data[:,:,:,:(nt+1)*100]
    srca_data = np.zeros((120,9,(nt+1)*100))
    data_path = r'F:\SSVEP\DCPM_Pre\origin：0.4pi\t%d.mat' %(nt)
    srca_data[:60,:,:] = temp[0,:,:,:]
    srca_data[60:,:,:] = data[1,:,:,:(nt+1)*100+10]
    label = [0 for i in range(60)]
    label += [1 for i in range(60)]
    srca_data = np.swapaxes(srca_data, 0, -1)
    io.savemat(data_path, {'srca_data':srca_data, 'label':label})
    
#%%
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

fig = plt.figure(figsize=(15,10))
gs = GridSpec(2,2,figure=fig)

ax1 = fig.add_subplot(gs[:,:])
ax1.set_title('Origin Signal', fontsize=24)
ax1.tick_params(axis='both', labelsize=20)
ax1.plot(np.mean(data[0,:,7,:], axis=0), label='0 phase')
ax1.plot(np.mean(data[1,:,7,:], axis=0), label='pi phase')
ax1.set_xlabel('Time/ms', fontsize=22)
ax1.set_ylabel('Amplitude/μV', fontsize=22)
ax1.vlines(140, -1, 1, color='black', linestyle='dashed', label='140ms')
ax1.legend(loc='upper left', fontsize=20)

fig.tight_layout()
plt.show()
plt.savefig(r'C:\Users\brynh\Desktop\fuck.png', dpi=600)

#%%
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

tar_chans = ['PZ ','PO5','PO3','POZ','PO4','PO6','O1 ','OZ ','O2 ']

fig = plt.figure(figsize=(15,10))
gs = GridSpec(2,2,figure=fig)
sns.set(style='whitegrid')

ax1 = fig.add_subplot(gs[:,:])
ax1.set_title('Fisher Score Alteration', fontsize=24)
ax1.tick_params(axis='both', labelsize=20)
for i in range(len(tar_chans)):
    exec('ax1.plot(parameter[i].T, label=tar_chans[i], linewidth=1.5)')
ax1.set_xlabel('Number of channels', fontsize=22)
ax1.set_ylabel('Fisher Score', fontsize=22)
ax1.legend(loc='upper right', fontsize=20)

fig.tight_layout()
plt.show()

#%%
x = [int(100*(parameter[y].T[-2]-parameter[y].T[0])/parameter[y].T[0])
     for y in range(len(parameter))]