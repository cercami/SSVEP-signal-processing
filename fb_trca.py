# -*- coding: utf-8 -*-
"""
Created on Wed Jan  1 21:04:13 2020

TRCA + 10-fold cross-validation

@author: Brynhildr
"""

#%% Import third part module
import numpy as np
import scipy.io as io
from scipy import signal

import copy

import mcee

import matplotlib.pyplot as plt
import seaborn as sns

#%% load data (origin)
eeg = io.loadmat(r'I:\SSVEP\dataset\preprocessed_data\weisiwen\f_data.mat')
data = eeg['f_data'][:,:,:,2000:3640]
chans = eeg['chan_info'].tolist()
data *= 1e6  
del eeg

#%% initialization
n_events = data.shape[0]
n_trials = data.shape[1]

# pick channels from parietal and occipital areas
tar_chans = ['PZ ', 'POZ', 'O1 ', 'OZ ', 'O2 ']
#tar_chans = ['POZ', 'O1 ', 'OZ ', 'O2 ']
#tar_chans = ['O1 ', 'OZ ', 'O2 ']
#tar_chans = ['POZ', 'OZ ']

n_chans = len(tar_chans)
n_times = data.shape[3]

#%% cross validation
sfreq = 1000

N = 10
n_bands = 10
acc_cv = np.zeros((11, n_bands))

# pick specific channels from origin data
pk_sig = np.zeros((data.shape[0], data.shape[1], n_chans, data.shape[3]))
for i in range(n_chans):
    pk_sig[:,:,i,:] = data[:,:,chans.index(tar_chans[i]),:]
del i, data 

# main loop for origin data | fixed channels, nb bands, nt times 
for nb in range(n_bands):  # form filter banks before extracting data
    # filter bank: 8-88Hz | IIR method, each band has 2Hz's buffer
    # change number of filter banks
    l_freq = [(8*(x+1)-2) for x in range(nb+1)]
    h_freq = 90
    # 5-D tension
    fb_data = np.zeros((n_events, nb+1, n_trials, n_chans, pk_sig.shape[3]))
    # apply Chebyshev I bandpass filter
    for i in range(nb+1):
        b, a = signal.cheby1(N=5, rp=0.5, Wn=[l_freq[i], h_freq], btype='bandpass',
                                 analog=False, output='ba', fs=sfreq)
        fb_data[:,i,:,:,:] = signal.filtfilt(b=b, a=a, x=pk_sig, axis=-1)
        del b, a
    del i
    
    # extract different lengths' data
    for nt in range(11):
        if nt == 0: # 500ms
            trca_data = fb_data[:,:,:,:,1060:]
            print('Data length: 500ms')
        elif nt == 1:  # 400ms
            trca_data = fb_data[:,:,:,:,1060:1540]
            print('Data length: 400ms')
        elif nt == 2:  # 300ms
            trca_data = fb_data[:,:,:,:,1060:1440]
            print('Data length: 300ms')
        elif nt == 3:  # 200ms
            trca_data = fb_data[:,:,:,:,1060:1340]
            print('Data length: 200ms')
        elif nt == 4:  # 180ms
            trca_data = fb_data[:,:,:,:,1060:1320]
            print('Data length: 180ms')
        elif nt == 5:  # 160ms
            trca_data = fb_data[:,:,:,:,1060:1300]
            print('Data length: 160ms')
        elif nt == 6:  # 140ms
            trca_data = fb_data[:,:,:,:,1060:1280]
            print('Data length: 140ms')
        elif nt == 7:  # 120ms
            trca_data = fb_data[:,:,:,:,1060:1260]
            print('Data length: 120ms')
        elif nt == 8:  # 100ms
            trca_data = fb_data[:,:,:,:,1060:1240]
            print('Data length: 100ms')
        elif nt == 9:  # 80ms
            trca_data = fb_data[:,:,:,:,1060:1220]
            print('Data length: 80ms')
        elif nt == 10:  # 60ms
            trca_data = fb_data[:,:,:,:,1060:1200]
            print('Data length: 60ms')
            
        # cross-validation + TRCA
        print('Running TRCA algorithm...')
        acc_temp = []
        for i in range(N):   
            # divide dataset
            a = i * 10
            # training dataset: (n_events, n_bands, n_trials, n_chans, n_times)
            tr_fb_data = trca_data[:,:,a:a+int(n_trials/N),:,:]
            # test dataset: (n_events, n_bands, n_trials, n_chans, n_times)
            te_fb_data = copy.deepcopy(trca_data)
            te_fb_data = np.delete(te_fb_data, [a,a+1,a+2,a+3,a+4,a+5,a+6,a+7,a+8,a+9], axis=2)
            # template data: (n_events, n_bands, n_chans, n_times)| basic unit: (n_chans, n_times)
            template = np.mean(tr_fb_data, axis=2)

            # main loop of TRCA
            acc = mcee.trca(tr_fb_data, te_fb_data)
            acc_temp.append(np.sum(acc))
            del acc
            # end                                
            print(str(i+1) + 'th cross-validation complete!')
    
        acc_cv[nt, nb] = np.sum(acc_temp)
        del acc_temp
        print(str(nb+1) + ' bands complete!\n')
        
#%% reshape results
acc_cv = acc_cv.T
acc_cv/=2700
acc_cv*=100

#%%
k=0
np.where(acc_cv[:,k]==np.min(acc_cv[:,k]))

#%%
f = 0
plt.plot(x[f,:,:].T, linewidth=0.5)
plt.plot(np.mean(x[f,:,:], axis=0), linewidth=4, color='black')
plt.vlines(1008.3, np.min(x[f,:,:]), np.max(x[f,:,:]), color='black', linestyle='dashed')
plt.vlines(1148.3, np.min(x[f,:,:]), np.max(x[f,:,:]), color='black', linestyle='dashed')

#%%
x = np.zeros((16,11))
x[0,:] = [99.4074,99.2593,98.7407,98.7778,97.4444,94.4444,93.037,92.3333,91.7037,89.5185,83.7407]
x[1,:] = [98.963,99.037,97.5926,97.6296,94.7778,90.1481,88.8889,89.2222,86.6296,85,77.4815]
x[2,:] = [99.4074,99.2963,99.1481,98.8889,98.1852,96,94.6667,93.7778,91.7407,88.8519,81.8148]
x[3,:] = [99.2963,99.1852,98.2222,98.2222,95.5185,90.2593,90.0741,89.2963,86.5556,83.8148,72.1111]
x[4,:] = [99.3333,99.2963,98.6667,98.3704,97,93.7778,91.8519,91.7778,90.9259,88.1111,80.3704]
x[5,:] = [98.9259,98.9259,97.7778,97.2963,94.7778,89.5556,88.0741,88.8519,86.1852,85.3704,77.963]
x[6,:] = [99.4074,99.1852,98.8519,98.8148,98.1111,96.2222,94.8889,93.963,91.3333,88,80.6296]
x[7,:] = [99.2222,99.0741,98.2593,98.4074,96.0741,92.0741,90.5926,89.7407,86.8148,83.9259,73.5185]
x[8,:] = [88.2963,88.2593,83.4074,81.7037,80.3704,76.2593,74.1111,72.2222,67.5185,63,60.8889]
x[9,:] = [86.1852,86.037,80.7778,77.3704,76.8519,71.037,68.2963,66.1852,59.7037,56.1111,54.5556]
x[10,:] = [88.4815,88,82.5185,79.8519,78.1481,73.5926,71.3704,68.8889,65.9259,60.3333,55.3333]
x[11,:] = [86.037,85.037,80.0741,74.7407,72.963,65.9259,64.2963,62.2593,59.8519,53.4074,49.3333]
x[12,:] = [99.3704,99.2963,98.7407,98.5926,97.037,93.2963,91.6667,90.7407,90.2222,89.5185,77.8889]
x[13,:] = [99.037,99.1111,98.2222,97.5926,94.1481,89.2593,88.7407,88.2963,87.2222,85.8889,73.6296]
x[14,:] = [99.4074,99.2222,99.1481,99.037,97.9259,95.5556,94.7778,93.8889,92.1111,90.4444,79]
x[15,:] = [99.1481,99.1111,98.5926,98.7037,95.8519,90.6296,91.4444,91.0741,89.2222,87.037,71.8519]

#%%
sns.set(style='whitegrid')

fig, ax = plt.subplots(2,2,figsize=(8,8))
x_ax = range(11)

ax[0,0].set_title('5 channels', fontsize=18)
ax[0,0].plot(x[0,:], label='140 max')
ax[0,0].plot(x[1,:], label='140 min')
ax[0,0].plot(x[2,:], label='60 max')
ax[0,0].plot(x[3,:], label='60 min')
ax[0,0].set_xlabel('data length', fontsize=18)
#ax[0,0].set_xticks(x_ax, ('500ms','400ms','300ms','200ms','180ms','160ms','140ms','120ms','100ms','80ms','60ms'))
ax[0,0].set_ylim((48, 100))
ax[0,0].set_ylabel('acc', fontsize=18)
ax[0,0].legend(loc='best', fontsize=18)

ax[0,1].set_title('4 channels', fontsize=18)
ax[0,1].plot(x[4,:], label='140 max')
ax[0,1].plot(x[5,:], label='140 min')
ax[0,1].plot(x[6,:], label='60 max')
ax[0,1].plot(x[7,:], label='60 min')
ax[0,1].set_xlabel('data length', fontsize=18)
ax[0,1].set_ylim((48, 100))
ax[0,1].set_ylabel('acc', fontsize=18)
ax[0,1].legend(loc='best', fontsize=18)

ax[1,0].set_title('3 channels', fontsize=18)
ax[1,0].plot(x[8,:], label='140 max')
ax[1,0].plot(x[9,:], label='140 min')
ax[1,0].plot(x[10,:], label='60 max')
ax[1,0].plot(x[11,:], label='60 min')
ax[1,0].set_xlabel('data length', fontsize=18)
ax[1,0].set_ylim((48, 100))
ax[1,0].set_ylabel('acc', fontsize=18)
ax[1,0].legend(loc='best', fontsize=18)

ax[1,1].set_title('2 channels', fontsize=18)
ax[1,1].plot(x[12,:], label='140 max')
ax[1,1].plot(x[13,:], label='140 min')
ax[1,1].plot(x[14,:], label='60 max')
ax[1,1].plot(x[15,:], label='60 min')
ax[1,1].set_xlabel('data length', fontsize=18)
ax[1,1].set_ylim((48, 100))
ax[1,1].set_ylabel('acc', fontsize=18)
ax[1,1].legend(loc='best', fontsize=18)
