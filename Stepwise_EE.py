# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 09:57:52 2019

Stepwise Estimate Extraction algorithm to improve short-time SSVEP's SNR

@author: Brynhildr
"""

#%% import 3rd-part module
import numpy as np
import scipy.io as io

import time

from mne.filter import filter_data

from sklearn.linear_model import LinearRegression

import matplotlib.pyplot as plt

import copy

#%% timing
start = time.clock()

#%% load data
eeg = io.loadmat(r'F:\SSVEP\dataset\preprocessed_data\weisiwen\raw_data.mat')

data = eeg['raw_data'][2,:,:,:]

data *= 1e6  # reset unit

del eeg

# in future versions, chan_info will be combined into raw_data.mat
chans = io.loadmat(r'F:\SSVEP\dataset\preprocessed_data\weisiwen\chan_info.mat')
chans = chans['chan_info'].tolist()

# basic info
sfreq = 1000

#%% Data preprocessing
# filtering
f_data = np.zeros((data.shape[0], data.shape[1], 3700))
for i in range(data.shape[0]):
    f_data[i,:,:] = filter_data(data[i,:,:3700], sfreq=sfreq, l_freq=5,
                      h_freq=40, n_jobs=1)

del i, sfreq

# get data for linear regression
w = f_data[:,:,2000:3000]          

# get data for comparision
signal_data = f_data[:,:,3200:]   # 500ms after 200ms's duration

del f_data, data

#%% Basic function definition
# multi-linear regression
def mlr(model_input, model_target, data_input, data_target):
    '''
    model_input: (n_chans, n_trials, n_times)
    model_target: (n_trials, n_times)
    data_input: (n_chans, n_trials, n_times)
    data_target: (n_trials, n_times)
    '''
    if model_input.ndim == 3:
        # regression intercept: (n_trials)
        RI = np.zeros((model_input.shape[1]))
        # estimate signal: (n_trials, n_times)
        estimate = np.zeros((model_input.shape[1], data_input.shape[2]))
        # regression coefficient: (n_trials, n_chans)
        RC = np.zeros((model_input.shape[1], model_input.shape[0]))
        
        for i in range(model_input.shape[1]):    # i for trials
            # basic operating unit: (n_chans, n_times).T, (1, n_times).T
            L = LinearRegression().fit(model_input[:,i,:].T, model_target[i,:].T)
            RI = L.intercept_
            RC = L.coef_
            estimate[i,:] = (np.mat(RC) * np.mat(data_input[:,i,:])).A + RI           
    elif model_input.ndim == 2:      # avoid reshape error
        RI = np.zeros((model_input.shape[0]))
        estimate = np.zeros((model_input.shape[0], data_input.shape[1]))
        RC = np.zeros((model_input.shape[0]))
        
        for i in range(model_input.shape[0]):    
            L = LinearRegression().fit(np.mat(model_input[i,:]).T, model_target[i,:].T)
            RI = L.intercept_
            RC = L.coef_
            estimate[i,:] = RC * data_input[i,:] + RI
    # extract SSVEP from raw data
    extract = data_target - estimate
    
    return extract, estimate

# compute time-domain snr
def snr_time(data):
    '''
    data:(n_trials, n_times)
    '''
    snr = np.zeros((data.shape[1]))             # (n_times)
    ex = np.mat(np.mean(data, axis=0))          # one-channel data: (1, n_times)
    
    temp = np.mat(np.ones((1, data.shape[0])))  # (1, n_trials)
    minus = (temp.T * ex).A                     # (n_trials, n_times)
    
    ex = (ex.A) ** 2                            # signal's power
    var = np.mean((data - minus)**2, axis=0)    # noise's power (avg)
    
    snr = ex/var
    
    return snr

#%% Initialization
# pick target signal channel
data_target = signal_data[:, chans.index('POZ'), :]
signal_data = np.delete(signal_data, chans.index('POZ'), axis=1)

w_target = w[:,chans.index('POZ'),:]
w = np.delete(w, chans.index('POZ'), axis=1)

del chans[chans.index('POZ')]

# config the variables
snr = snr_time(data_target)
msnr = np.mean(snr)
compare_snr = np.zeros((len(chans)))
max_loop = len(chans)

remain_chans = []
snr_change = []

temp_snr = []
core_data = []
core_w = []

# significant loop mark
j = 1

#%% Begin loop of Stepwise EE
active = True
while active and len(chans) <= max_loop:
    # initialization
    compare_snr = np.zeros((len(chans)))
    mtemp_snr = np.zeros((len(chans)))
    
    # add 1 channel respectively & compare the snr (Forward EE)
    for i in range(len(chans)):
        # avoid reshape error in multi-dimension array
        if j == 1:
            temp_w = w[:,i,:]
            temp_data = signal_data[:,i,:]
        else:
            temp_w = np.zeros((j, w.shape[0], w.shape[2]))
            temp_w[:j-1, :, :] = core_w
            temp_w[j-1, :, :] = w[:,i,:]
            
            temp_data = np.zeros((j, signal_data.shape[0], signal_data.shape[2]))
            temp_data[:j-1, :, :] = core_data
            temp_data[j-1, :, :] = signal_data[:,i,:]
        # multi-linear regression & snr computation
        temp_extract, temp_estimate = mlr(temp_w, w_target, temp_data, data_target)
        del temp_w, temp_data, temp_estimate
        temp_snr = snr_time(temp_extract)
        # compare the snr with original one
        mtemp_snr[i] = np.mean(temp_snr)
        compare_snr[i] = mtemp_snr[i] - msnr
    # keep the channels which can improve snr most
    chan_index = np.max(np.where(compare_snr == np.max(compare_snr)))
    remain_chans.append(chans.pop(chan_index))
    snr_change.append(np.max(compare_snr))
    del temp_extract, compare_snr, mtemp_snr, temp_snr
        
    # avoid reshape error at the beginning of Forward EE while refreshing data
    if j == 1: 
        core_w = w[:, chan_index, :]
        core_data = signal_data[:, chan_index, :]
        
        # refresh data
        signal_data = np.delete(signal_data, chan_index, axis=1)
        w = np.delete(w ,chan_index, axis=1)
        
        # significant loop mark
        print('Complete ' + str(j) + 'th loop')
        
    # begin stepwise part (delete & add) 
    if j == 2:  
        # save new data
        temp_core_w = np.zeros((j, w.shape[0], w.shape[2]))
        temp_core_w[0, :, :] = core_w
        temp_core_w[1, :, :] = w[:, chan_index, :]
        core_w = copy.deepcopy(temp_core_w)
        del temp_core_w

        temp_core_data = np.zeros((j, signal_data.shape[0], signal_data.shape[2]))
        temp_core_data[0, :, :] = core_data
        temp_core_data[1, :, :] = signal_data[:, chan_index, :]
        core_data = copy.deepcopy(temp_core_data)
        del temp_core_data
    
        # add judge condition to stop program while achieving the target
        if snr_change[-1] < np.max(snr_change):
            print('Stepwise EE complete!')
            end = time.clock()
            print('Total running time: ' + str(end - start) + 's')
            # if this judge is not passed, then there's no need to continue
            active = False
    
        # delete 1st channel, then add a new one
        else:
            # refresh data
            signal_data = np.delete(signal_data, chan_index, axis=1)
            w = np.delete(w, chan_index, axis=1)
            
            # initialization
            temp_1_chans = copy.deepcopy(remain_chans)
            temp_1_data = copy.deepcopy(core_data)
            temp_1_w = copy.deepcopy(core_w)
        
            # delete 1st channel
            del temp_1_chans[0]
            temp_1_data = np.delete(temp_1_data, 0, axis=0)
            temp_1_w = np.delete(temp_1_w, 0, axis=0)
        
            # add one channel
            temp_2_compare_snr = np.zeros((temp_1_data.shape[1]))
            for k in range(w.shape[1]):
                temp_2_w = np.zeros((2, w.shape[0], w.shape[2]))
                temp_2_w[0, :, :] = temp_1_w
                temp_2_w[1, :, :] = w[:, k, :]
                
                temp_2_data = np.zeros((2, signal_data.shape[0], signal_data.shape[2]))
                temp_2_data[0, :, :] = temp_1_data
                temp_2_data[1, :, :] = signal_data[:, k, :]
                
                # mlr & compute snr
                temp_2_extract, temp_2_estimate = mlr(temp_2_w, w_target, temp_2_data, data_target)
                temp_2_snr = snr_time(temp_2_extract)
                mtemp_2_snr = np.mean(temp_2_snr)
                temp_2_compare_snr[k] = mtemp_2_snr - msnr
        
            # keep the best choice
            temp_2_chan_index = np.max(np.where(temp_2_compare_snr == np.max(temp_2_compare_snr)))
        
            # judge if there's any improvement
            if temp_2_compare_snr[temp_2_chan_index] > snr_change[-1]:  # has improvement
                # refresh data
                chan_index = temp_2_chan_index
                remain_chans.append(chans.pop(chan_index))
                snr_change.append(temp_2_compare_snr[temp_2_chan_index])
                
                # delete useless data & add new data
                core_w = np.delete(core_w, 0, axis=0)
                
                temp_2_core_w = np.zeros((2, w.shape[0], w.shape[2]))
                temp_2_core_w[0, :, :] = core_w
                temp_2_core_w[1, :, :] = w[:, chan_index, :]
                core_w = copy.deepcopy(temp_2_core_w)
                del temp_2_core_w
                
                core_data = np.delete(core_data, 0, axis=0)
            
                temp_2_core_data = np.zeros((2, signal_data.shape[0], signal_data.shape[2]))
                temp_2_core_data[0, :, :] = core_data
                temp_2_core_data[1, :, :] = signal_data[:, chan_index, :]
                core_data = copy.deepcopy(temp_2_core_data)
                del temp_2_core_data
                
                signal_data = np.delete(signal_data, chan_index, axis=1)
                w = np.delete(w, chan_index, axis=1)
                
                # release RAM
                del remain_chans[0]
                del temp_2_chan_index, temp_2_extract, temp_2_estimate, temp_2_snr
                del mtemp_2_snr, temp_2_compare_snr, temp_2_w, temp_2_data
                del temp_1_chans, temp_1_data, temp_1_w
                
                # significant loop mark
                print('Complete ' + str(j) + 'th loop')
                
            else:  # no improvement
                # release RAM
                del temp_1_chans, temp_1_data, temp_1_w
                # reset
                print("Already best in 2 channels' contidion!")

    # now we have at least 3 elements in remain_chans,
    # delete one channel, then add a new one
    if j > 2:
        # save data
        temp_core_w = np.zeros((j, w.shape[0], w.shape[2]))
        temp_core_w[:j-1, :, :] = core_w
        temp_core_w[j-1, :, :] = w[:, chan_index, :]
        core_w = copy.deepcopy(temp_core_w)
        del temp_core_w

        temp_core_data = np.zeros((j, signal_data.shape[0], signal_data.shape[2]))
        temp_core_data[:j-1, :, :] = core_data
        temp_core_data[j-1, :, :] = signal_data[:, chan_index, :]
        core_data = copy.deepcopy(temp_core_data)
        del temp_core_data
        # add judge condition to stop program while achieving the target
        if snr_change[-1] < np.max(snr_change):
            print('Stepwise EE complete!')
            end = time.clock()
            print('Total running time: ' + str(end - start) + 's')
            # if this judge is not passed, then there's no need to continue
            active = False
        
        # now the last snr_change is still the largest in the total sequence
        else:
            # refresh data
            signal_data = np.delete(signal_data, chan_index, axis=1)
            w = np.delete(w, chan_index, axis=1)
        
            # initialization (make copies)
            temp_3_chans = copy.deepcopy(remain_chans)
            temp_3_data = copy.deepcopy(core_data)
            temp_3_w = copy.deepcopy(core_w)
            
            temp_3_compare_snr = np.zeros((len(temp_3_chans)-1))
            temp_3_chan_index = []
        
            # delete one channel except the latest one
            for l in range(len(temp_3_chans)-1):
                # initialization (make copies)
                temp_4_chans = copy.deepcopy(remain_chans)
                temp_4_data = copy.deepcopy(core_data)
                temp_4_w = copy.deepcopy(core_w)
            
                # delete one channel
                del temp_4_chans[l]
                temp_4_data = np.delete(temp_4_data, l, axis=0)
                temp_4_w = np.delete(temp_4_w, l, axis=0)
            
                # add one channel
                temp_4_compare_snr = np.zeros((signal_data.shape[1]))
                for m in range(signal_data.shape[1]):
                    temp_5_w = np.zeros((j, w.shape[0], w.shape[2]))
                    temp_5_w[:j-1, :, :] = temp_4_w
                    temp_5_w[j-1, :, :] = w[:, m, :]
                
                    temp_5_data = np.zeros((j, signal_data.shape[0], signal_data.shape[2]))
                    temp_5_data[:j-1, :, :] = temp_4_data
                    temp_5_data[j-1, :, :] = signal_data[:, m, :]
                
                    # mlr & compute snr
                    temp_5_extract, temp_5_estimate = mlr(temp_5_w, w_target, temp_5_data, data_target)
                    temp_5_snr = snr_time(temp_5_extract)
                    mtemp_5_snr = np.mean(temp_5_snr)
                    temp_4_compare_snr[m] = mtemp_5_snr - msnr
                
                # keep the best choice
                temp_4_chan_index = np.max(np.where(temp_4_compare_snr == np.max(temp_4_compare_snr)))
                temp_3_chan_index.append(str(temp_4_chan_index))
                temp_3_compare_snr[l] = temp_4_compare_snr[temp_4_chan_index]
        
            # judge if there's improvement
            if np.max(temp_3_compare_snr) > np.max(snr_change):  # has improvement
                # find index
                delete_chan_index = np.max(np.where(temp_3_compare_snr == np.max(temp_3_compare_snr)))
                add_chan_index = int(temp_3_chan_index[delete_chan_index])
            
                # operate (refresh data)
                del remain_chans[delete_chan_index]
                remain_chans.append(chans[add_chan_index])
                chan_index = add_chan_index
                snr_change.append(temp_3_compare_snr[delete_chan_index])
                
                # delete useless data & add new data
                core_w = np.delete(core_w, delete_chan_index, axis=0)
                
                temp_6_core_w = np.zeros((core_w.shape[0]+1, core_w.shape[1], core_w.shape[2]))
                temp_6_core_w[:core_w.shape[0], :, :] = core_w
                temp_6_core_w[core_w.shape[0], :, :] = w[:, add_chan_index, :]
                core_w = copy.deepcopy(temp_6_core_w)
                del temp_6_core_w
                
                core_data = np.delete(core_data, delete_chan_index, axis=0)
                
                temp_6_core_data = np.zeros((core_data.shape[0]+1, core_data.shape[1], core_data.shape[2]))
                temp_6_core_data[:core_data.shape[0], :, :] = core_data
                temp_6_core_data[core_data.shape[0], :, :] = signal_data[:, add_chan_index, :]
                core_data = copy.deepcopy(temp_6_core_data)
                del temp_6_core_data
                
                signal_data = np.delete(signal_data, add_chan_index, axis=1)
                w = np.delete(w, add_chan_index, axis=1)
                del chans[add_chan_index]
                
                # release RAM
                del temp_3_chans, temp_3_data, temp_3_w, temp_3_compare_snr, temp_3_chan_index
                del temp_4_chans, temp_4_data, temp_4_w, temp_4_compare_snr, temp_4_chan_index
                del temp_5_data, temp_5_w, temp_5_extract, temp_5_estimate, mtemp_5_snr, temp_5_snr
                
                # significant loop mark
                print('Complete ' + str(j) + 'th loop')
                
            else:  # no improvement
                # release RAM
                del temp_3_chans, temp_3_data, temp_3_w, temp_3_compare_snr, temp_3_chan_index
                del temp_4_chans, temp_4_data, temp_4_w, temp_4_compare_snr, temp_4_chan_index
                del temp_5_data, temp_5_w, temp_5_extract, temp_5_estimate, mtemp_5_snr, temp_5_snr
                
                # reset
                print('Complete ' + str(j) + 'th loop')
                print("Already best in " + str(j) + " channels' condition!")
    
    j += 1
    
#%% Algorithm operating results
remain_chans = remain_chans[:len(remain_chans)-1]