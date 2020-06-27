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
#import srca
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import fp_growth as fpg
import mcee
import signal_processing_function as SPF

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
    eeg = io.loadmat(r'F:\SSVEP\ridge\ridge+corr\real1_1\mcee_%d.mat' %(i))
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
#%% real te tr
# pick channels from parietal and occipital areas
tar_chans = ['PZ ','PO5','PO3','POZ','PO4','PO6','O1 ','OZ ','O2 ']
n_events = 2
n_trials = 115
n_chans = len(tar_chans)
n_test = 60
n_times = 2140
# mcee optimization
for nfile in range(1):
    if nfile == 0:  
        ns = 55    
    #elif nfile == 1:  
	 #   ns = 40       
    #elif nfile == 2:  
     #   ns = 30       
    #elif nfile == 3:
     #   ns = 25   
    for nt in range(1):
        model_info = []
        snr_alteration = []
        mcee_sig = np.zeros((n_events, n_test, n_chans, n_times))
        for ntc in range(len(tar_chans)):
            for ne in range(2):  # ne for n_events
                target_channel = tar_chans[ntc]
                sfreq = 1000
                # load local data (extract from .cnt file)
                eeg = io.loadmat(r'F:\SSVEP\dataset\preprocessed_data\wuqiaoyi\50_70_bp.mat')
                f_data = eeg['f_data'][ne,:,:,1000:3140] * 1e6
                w = f_data[:ns,:,:1000]
                signal_data = f_data[:ns,:,1140:int(1840+nt*100)]
                chans = eeg['chan_info'].tolist() 
                del eeg
                # variables initialization-model
                w_o = w[:,chans.index(target_channel),:]
                w_temp = copy.deepcopy(w)
                w_i = np.delete(w_temp, chans.index(target_channel), axis=1)
                del w_temp
                # variables initialization-signal
                sig_o = signal_data[:,chans.index(target_channel),:]
                sig_temp = copy.deepcopy(signal_data)
                sig_i = np.delete(sig_temp, chans.index(target_channel), axis=1)
                del sig_temp, signal_data
                # config chans & parameter info
                srca_chans = copy.deepcopy(chans)
                del srca_chans[chans.index(target_channel)]
                corr = mcee.pearson_corr(sig_o)
                mcorr = np.mean(corr)
                del corr
                # use stepwise method to find channels
                model_chans, para_change = mcee.stepwise_MCEE(chans=srca_chans,
                        msnr=mcorr, w=w, w_target=w_o, signal_data=sig_i,
                        data_target=sig_o)
                snr_alteration.append(para_change)				
				# pick channels chosen from stepwise
                w_i_te = np.zeros((60, len(model_chans), 1000))
                sig_i_te = np.zeros((60, len(model_chans), 2140))
                w_te = f_data[-60:,:,:1000]
                sig_te = f_data[-60:,:,:]
                for nc in range(len(model_chans)):
                    w_i_te[:,nc,:] = w_te[:,chans.index(model_chans[nc]),:]
                    sig_i_te[:,nc,:] = sig_te[:,chans.index(model_chans[nc]),:]
                del nc
                # mcee main process
                w_o_te = w_te[:,chans.index(target_channel),:]
                sig_o_te = sig_te[:,chans.index(target_channel),:]
                rc, ri, r2 = SPF.mlr_analysis(w_i_te, w_o_te)
                w_es_s, w_ex_s = SPF.sig_extract_mlr(rc, sig_i_te, sig_o_te, ri)
                del rc, ri, r2, w_es_s
                # save optimized data
                mcee_sig[ne,:,ntc,:] = w_ex_s
                model_info.append(model_chans)
                #del w_ex_s, model_chans, f_sig_i, sig_o, w_i, w_o, w, signal_data, f_sig_o
                #del para_change, w_i, w_o, sig_i, sig_o, srca_chans, mcorr 
        data_path = r'F:\SSVEP\ridge\ridge+corr\real1_%d\mcee_%d.mat' %(int(nfile+1), nt+6)             
        io.savemat(data_path, {'mcee_sig':mcee_sig,
								'model_info': model_info,
								'parameter': snr_alteration})

# common trca
acc_srca_tr = np.zeros((4,5,10))
for nfile in range(1):
    for nt in range(4):
        if nfile == 0:
            eeg = io.loadmat(r'F:\SSVEP\ridge\ridge+corr\real1_1\mcee_%d.mat' %(nt+6))
        #elif nfile == 1:
         #   eeg = io.loadmat(r'F:\SSVEP\ridge\ridge+corr\real1_2\mcee_%d.mat' %(nt))
        #elif nfile == 2:
         #   eeg = io.loadmat(r'F:\SSVEP\ridge\ridge+corr\real1_3\mcee_%d.mat' %(nt))
        #elif nfile == 3:
         #   eeg = io.loadmat(r'F:\SSVEP\ridge\ridge+corr\real1_4\mcee_%d.mat' %(nt))
        data = eeg['mcee_sig'][:,:,:,1140:int(1840+nt*100)]
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
            acc_temp = mcee.pure_trca(train_data=tr_data, test_data=te_data)
            acc.append(np.sum(acc_temp))
            del acc_temp
            print(str(cv+1) + 'th cv complete')
        acc = np.array(acc)/(te_data.shape[1]*2)
        acc_srca_tr[nfile,:,nt] = acc
        del acc

acc_srca_te = np.zeros((4,5,10))
for nfile in range(1):
    for nt in range(4):
        if nfile == 0:
            eeg = io.loadmat(r'F:\SSVEP\ridge\ridge+corr\real1_1\mcee_%d.mat' %(nt+6))
        #elif nfile == 1:
         #   eeg = io.loadmat(r'F:\SSVEP\ridge\ridge+corr\real1_2\mcee_%d.mat' %(nt))
        #elif nfile == 2:
         #   eeg = io.loadmat(r'F:\SSVEP\ridge\ridge+corr\real1_3\mcee_%d.mat' %(nt))
        #elif nfile == 3:
         #   eeg = io.loadmat(r'F:\SSVEP\ridge\ridge+corr\real1_4\mcee_%d.mat' %(nt))
        data = eeg['mcee_sig'][:,:,:,1140:int(1840+nt*100)]
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
            acc_temp = mcee.pure_trca(train_data=te_data, test_data=tr_data)
            acc.append(np.sum(acc_temp))
            del acc_temp
            print(str(cv+1) + 'th cv complete')
        acc = np.array(acc)/(tr_data.shape[1]*2)
        acc_srca_te[nfile,:,nt] = acc
        del acc