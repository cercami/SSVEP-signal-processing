# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 12:07:56 2019
    SRCA - Spatial Regression Component Analysis
1. Basic operating functions:
    (1) mlr: Multi_linear regression
    (2) snr_time: SNR in time domain
    (3) pearson_corr: Pearson Correlation Coefficient (Mean of sequence)
    (4) fisher_score: Fisher Score
    (5) sinw: make sine wave
    (6) template_phase: best initial phase of signal template
    (7) template_corr: compute correlation between real signal and template
    (8) apply_SRCA: apply SRCA model
    
2. Two kinds of recursive algorithm to choose channels for SRCA optimization
    (1) stepwise_SRCA | including SNR, Corr and CCA method, intra-class optimization
    (2) stepwise_SRCA_fs | fisher score method, inter-class optimization

3. Target identification
    (1) standard CCA
    (2) TRCA: including standard TRCA, filter-bank TRCA and extended-TRCA
        for normal signal and SRCA signal
    (3) DCPM: 5 different descrimination indices for normal and SRCA signal
    (4) corr_detect: single channel detection


@ author: Brynhildr
@ email: brynhildrw@gmail.com
version 1.0
"""

#%% Import third part module
import numpy as np
from numpy import linalg as LA
from numpy import corrcoef as CORR

from sklearn import linear_model

import copy
import time
from math import pi

#%% Basic operating function
# multi-linear regression
def mlr(model_input, model_target, data_input, data_target, regression='OLS',
        alpha=1.0, l1_ratio=1.0):
    '''
    the main process of spatial regression component analysis (SRCA)

    Parameters
    ----------
    model_input : (n_chans, n_trials, n_times) / (n_trials, n_times)
        rest-state data of regression channels.
    model_target : (n_trials, n_times)
        rest-state data of target channel.
    data_input : (n_chans, n_trials, n_times)
        mission-state data of regression channels.
    data_target : (n_trials, n_times)
        mission-state data of target channel.
    regression : str, optional
        OLS, Ridge, Lasso or ElasticNet. The default is 'OLS'.
    alpha : float, optional
        parameters used in Ridge, Lasso and EN regression. The default is 1.0.
    l1_ratio : float, optional
        parameters used in EN regression. The default is 1.0.

    Returns
    -------
    extract : (n_trials, n_times)
        SRCA filtered data.
    '''
    if model_input.ndim == 3:
        # estimate signal: (n_trials, n_times)
        estimate = np.zeros((data_input.shape[1], data_input.shape[2]))
        for i in range(model_input.shape[1]):    # i for trials
            # basic operating unit: (n_chans, n_times).T, (1, n_times).T
            if regression == 'OLS':
                L = linear_model.LinearRegression().fit(model_input[:,i,:].T, model_target[i,:].T)
            elif regression == 'Ridge':
                L = linear_model.Ridge(alpha=alpha).fit(model_input[:,i,:].T, model_target[i,:].T)
            elif regression == 'Lasso':
                L = linear_model.Lasso(alpha=alpha).fit(model_input[:,i,:].T, model_target[i,:].T)
            elif regression == 'ElasticNet':
                L = linear_model.ElasticNet(alpha=alpha, l1_ratio=l1_ratio).fit(
                    model_input[:,i,:].T, model_target[i,:].T)
            RI = L.intercept_
            RC = L.coef_
            estimate[i,:] = (np.mat(RC) * np.mat(data_input[:,i,:])).A + RI
    elif model_input.ndim == 2:      # avoid reshape error
        RI = np.zeros((model_input.shape[0]))
        estimate = np.zeros((model_input.shape[0], data_input.shape[1]))
        RC = np.zeros((model_input.shape[0]))
        for i in range(model_input.shape[0]):
            if regression == 'OLS':
                L = linear_model.LinearRegression().fit(np.mat(model_input[i,:]).T, model_target[i,:].T)
            elif regression == 'Ridge':
                L = linear_model.Ridge(alpha=alpha).fit(np.mat(model_input[i,:]).T, model_target[i,:].T)
            elif regression == 'Lasso':
                L = linear_model.Lasso(alpha=alpha).fit(np.mat(model_input[i,:]).T, model_target[i,:].T)
            elif regression == 'ElasticNet':
                L = linear_model.ElasticNet(alpha=alpha, l1_ratio=l1_ratio).fit(
                    np.mat(model_input[i,:]).T, model_target[i,:].T)
            RI = L.intercept_
            RC = L.coef_
            estimate[i,:] = RC * data_input[i,:] + RI
    # extract SSVEP from raw data
    extract = data_target - estimate
    return extract

# compute time-domain snr
def snr_time(data):
    '''
    Compute the mean of SSVEP's SNR in time domain

    Parameters
    ----------
    data : (n_trials, n_times)

    Returns
    -------
    snr : float
        the mean of SNR sequence.
    '''
    snr = np.zeros((data.shape[1]))             # (n_times)
    ex = np.mat(np.mean(data, axis=0))          # one-channel data: (1, n_times)
    temp = np.mat(np.ones((1, data.shape[0])))  # (1, n_trials)
    minus = (temp.T * ex).A                     # (n_trials, n_times)
    ex = (ex.A) ** 2                            # signal's power
    var = np.mean((data - minus)**2, axis=0)    # noise's power (avg)
    snr = ex/var
    return snr

# compute time-domain Pearson Correlation Coefficient
def pearson_corr(data):
    '''
    Compute the mean of Pearson correlation coefficient in time domain

    Parameters
    ----------
    data : (n_trials, n_times)

    Returns
    -------
    corr : (n_times,)
        corr sequence.
    '''
    template = data.mean(axis=0)
    n_trials = data.shape[0]
    corr = np.zeros((n_trials))
    for i in range(n_trials):
        corr[i] = np.sum(np.tril(np.corrcoef(template, data[i,:]),-1))
    del i
    return corr

# compute Fisher Score
def fisher_score(data):
    '''
    Compute the mean of Fisher Score in time domain

    Parameters
    ----------
    data : (n_events, n_trials, n_times)

    Returns
    -------
    fs : (n_times,)
        fisher score sequence.
    '''
    # initialization
    sampleNum = data.shape[1]    # n_trials
    featureNum = data.shape[-1]  # n_times
    groupNum = data.shape[0]     # n_events
    miu = data.mean(axis=1)      # (n_events, n_times)
    all_miu = miu.mean(axis=0)
    # inter-class divergence
    ite_d = np.sum(sampleNum * (miu - all_miu)**2, axis=0)
    # intra-class divergence
    itr_d= np.zeros((groupNum, featureNum))
    for i in range(groupNum):
        for j in range(featureNum):
            itr_d[i,j] = np.sum((data[i,:,j] - miu[i,j])**2)
    # fisher score
    fs = (ite_d) / np.sum(itr_d, axis=0)
    return fs

# make sine wave
def sinw(freq, time, phase, sfreq=1000):
    '''
    make sine wave

    Parameters
    ----------
    freq : float
        frequency / Hz.
    time : float
        time lenght / s.
    phase : float
        0-2.
    sfreq : float/int, optional
        sampling frequency. The default is 1000.

    Returns
    -------
    wave : (time*sfreq,)
        sequence.
    '''
    n_point = int(time*sfreq)
    time_point = np.linspace(0, (n_point-1)/sfreq, n_point)
    wave = np.sin(2*pi*freq*time_point + pi*phase)
    return wave

# choose best template phase
def template_phase(data, freq, step=100, sfreq=1000):
    '''
    choose best template initial phase for training dataset

    Parameters
    ----------
    data : (n_trials, n_times)
    freq : float
        frequency of template.
    step : int, optional
        stepwidth = 1 / steps. The default is 100.
    sfreq : float, optional
        sampling frequency of signal. The default is 1000.

    Returns
    -------
    best_phase : float
        0-1.
    '''
    time = data.shape[-1] / sfreq
    phase = [2*x/step for x in range(step)]
    corr = np.zeros((step))
    sig_template = np.mean(data, axis=0)
    for i in range(step):
        tar_template = sinw(freq=freq, time=time, phase=phase[i], sfreq=sfreq)
        corr[i] = np.sum(np.tril(np.corrcoef(sig_template, tar_template), -1))
    del i
    index = np.max(np.where(corr == np.max(corr)))
    best_phase = index*2/100
    return best_phase

# compute correlation with artificial template
def template_corr(data, freq, phase, sfreq=1000):
    '''
    Parameters
    ----------
    data : (n_trials, n_times)
    freq : float
        frequency of template
    phase : float
        0-1, initial phase of template.
    sfreq : float, optional
        sampling frequency of signal. The default is 1000.

    Returns
    -------
    corr : (n_trials,)
        correlation sequence.
    '''
    n_trials = data.shape[0]
    time_length = data.shape[-1] / sfreq
    template = sinw(freq, time_length, phase, sfreq)
    corr = np.zeros((n_trials))
    for i in range(n_trials):
        corr[i] = np.sum(np.tril(np.corrcoef(template, data[i,:]),-1))
    return corr

# apply SRCA model
def apply_SRCA(data, tar_chans, model_chans, chans, regression='OLS', sp=1140):
    '''
    Apply SRCA model in test dataset
    
    Parameters
    ----------
    data : (n_trials, n_chans, n_times)
        test dataset.
    tar_chans : list
        names of target channels.
    model_chans : list
        names of SRCA channels for all target channels.
    chans : list
        names of all channels.
    regression : str, optional
        OLS, Ridge, Lasso or ElasticNet. The default is 'OLS'.
    sp : int, optional
        start point of mission state. The default is 1140.
        
    Returns
    -------
    f_data : (n_trials, n_chans, n_times)
        SRCA filtered data.
    '''
    n_trials = data.shape[0]
    n_chans = len(tar_chans)
    n_times = data.shape[-1] - sp
    f_data = np.zeros((n_trials, n_chans, n_times))
    for ntc in range(len(tar_chans)):
        target_channel = tar_chans[ntc]
        model_chan = model_chans[ntc]
        w_i = np.zeros((n_trials, len(model_chan), 1000))
        sig_i = np.zeros((n_trials, len(model_chan), n_times))
        for nc in range(len(model_chan)):
            w_i[:, nc, :] = data[:, chans.index(model_chan[nc]), :1000]
            sig_i[:, nc, :] = data[:, chans.index(model_chan[nc]), sp:]
        del nc
        w_o = data[:, chans.index(target_channel), :1000]
        sig_o = data[:, chans.index(target_channel), sp:]
        w_i = np.swapaxes(w_i, 0, 1)
        sig_i = np.swapaxes(sig_i, 0, 1)
        w_ex_s = mlr(model_input=w_i, model_target=w_o, data_input=sig_i,
                     data_target=sig_o, regression=regression)
        f_data[:, ntc, :] = w_ex_s
    return f_data

# zero mean normalization (if neccessary)
def zero_mean(data):
    '''
    
    Zero mean normalization
    Parameters
    ----------
    data : (n_events, n_trials, n_chans, n_times)
        input data array (z-scored)

    Returns
    -------
    data : (n_events, n_trials, n_chans, n_times)
        data after z-scored

    ''' 
    data -= data.mean(axis=1, keepdims=True)
    return data

#%% Stepwise SRCA
def stepwise_SRCA(chans, mpara, w, w_target, signal_data, data_target, method,
                  regression, freq=None, phase=None, sfreq=1000):
    '''
    Stepward recursive algorithm to achieve SRCA
    The combination of Forward and Backward process:
        (1)this time form an empty set; 
        (2)add one channel respectively and pick the best one; 
        (3)add one channel respectively and delete one respectively (except the just-added one)
            keep the best choice;
        (4)repeat those process until there will be no better choice
            i.e. the convergence point of the recursive algorithm
    Parameters:
        chans: list of channels; the list order corresponds to the data array's
        mpara: float; the mean of original signal's parameters in time domain(0-500ms)
        w: background part input data array (n_trials, n_chans, n_times)
        w_target: background part target data array (n_trials, n_times)
        signal_data: signal part input data array (n_trials, n_chans, n_times)
        data_target: signal part target data array (n_trials, n_times)
    Returns:
        model_chans: list of channels which should be used in SRCA
        para_change: list of parameter's alteration
    '''
    # initialize variables
    print('Running Stepwise SRCA...')
    start = time.perf_counter()   
    j = 1   
    compare_para = np.zeros((len(chans)))
    max_loop = len(chans)   
    remain_chans = []
    para_change = []
    temp_para = []
    core_data = []
    core_w = []
    # begin loop
    active = True
    while active and len(chans) <= max_loop:
        # initialization
        compare_para = np.zeros((len(chans)))
        mtemp_para = np.zeros((len(chans)))
        # add 1 channel respectively & compare the parameter
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
            # multi-linear regression & parameter computation
            temp_extract = mlr(model_input=temp_w, model_target=w_target,
                               data_input=temp_data, data_target=data_target,
                               regression=regression)
            del temp_w, temp_data
            if method == 'SNR':
                temp_para = snr_time(temp_extract)
            elif method == 'Corr':
                temp_para = pearson_corr(temp_extract)
            elif method == 'CCA':
                temp_para = template_corr(temp_extract, freq=freq, phase=phase, sfreq=sfreq)
            # compare the parameter with original one
            mtemp_para[i] = np.mean(temp_para)
            compare_para[i] = mtemp_para[i] - mpara
        # keep the channels which can improve parameters most
        chan_index = np.max(np.where(compare_para == np.max(compare_para)))
        remain_chans.append(chans.pop(chan_index))
        para_change.append(np.max(compare_para))
        del temp_extract, compare_para, mtemp_para, temp_para      
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
            if para_change[-1] < np.max(para_change):
                print('Stepwise complete!')
                end = time.perf_counter()
                print('Recursive running time: ' + str(end - start) + 's')
                # if this judgement is not passed, then there's no need to continue
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
                temp_2_compare_para = np.zeros((signal_data.shape[1]))
                for k in range(signal_data.shape[1]):
                    temp_2_w = np.zeros((2, w.shape[0], w.shape[2]))
                    temp_2_w[0, :, :] = temp_1_w
                    temp_2_w[1, :, :] = w[:, k, :]
                    temp_2_data = np.zeros((2, signal_data.shape[0], signal_data.shape[2]))
                    temp_2_data[0, :, :] = temp_1_data
                    temp_2_data[1, :, :] = signal_data[:, k, :]
                    # mlr & compute parameters
                    temp_2_extract = mlr(model_input=temp_2_w, model_target=w_target,
                                         data_input=temp_2_data, data_target=data_target,
                                         regression=regression)
                    if method == 'SNR':
                        temp_2_para = snr_time(temp_2_extract)
                    elif method == 'Corr':
                        temp_2_para = pearson_corr(temp_2_extract)
                    elif method == 'CCA':
                        temp_2_para = template_corr(temp_2_extract, freq=freq, phase=phase, sfreq=sfreq)
                    mtemp_2_para = np.mean(temp_2_para)
                    temp_2_compare_para[k] = mtemp_2_para - mpara
                # keep the best choice
                temp_2_chan_index = np.max(np.where(temp_2_compare_para == np.max(temp_2_compare_para)))
                # judge if there's any improvement
                if temp_2_compare_para[temp_2_chan_index] > para_change[-1]:  # has improvement
                    # refresh data
                    chan_index = temp_2_chan_index
                    remain_chans.append(chans.pop(chan_index))
                    para_change.append(temp_2_compare_para[temp_2_chan_index])
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
                    del temp_2_chan_index, temp_2_extract, temp_2_para
                    del mtemp_2_para, temp_2_compare_para, temp_2_w, temp_2_data
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
            if para_change[-1] < np.max(para_change):
                print('Stepwise complete!')
                end = time.perf_counter()
                print('Recursive running time: ' + str(end - start) + 's')
                # if this judge is not passed, then there's no need to continue
                active = False
            # now the last para_change is still the largest in the total sequence
            else:
                # refresh data
                signal_data = np.delete(signal_data, chan_index, axis=1)
                w = np.delete(w, chan_index, axis=1)
                # initialization (make copies)
                temp_3_chans = copy.deepcopy(remain_chans)
                temp_3_compare_para = np.zeros((len(temp_3_chans)-1))
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
                    temp_4_compare_para = np.zeros((signal_data.shape[1]))
                    for m in range(signal_data.shape[1]):
                        temp_5_w = np.zeros((j, w.shape[0], w.shape[2]))
                        temp_5_w[:j-1, :, :] = temp_4_w
                        temp_5_w[j-1, :, :] = w[:, m, :]               
                        temp_5_data = np.zeros((j, signal_data.shape[0], signal_data.shape[2]))
                        temp_5_data[:j-1, :, :] = temp_4_data
                        temp_5_data[j-1, :, :] = signal_data[:, m, :]
                        # mlr & compute para
                        temp_5_extract = mlr(model_input=temp_5_w, model_target=w_target,
                                             data_input=temp_5_data, data_target=data_target,
                                             regression=regression)
                        if method == 'SNR':
                            temp_5_para = snr_time(temp_5_extract)
                        elif method == 'Corr':
                            temp_5_para = pearson_corr(temp_5_extract)
                        elif method == 'CCA':
                            temp_5_para = template_corr(temp_5_extract, freq=freq, phase=phase, sfreq=sfreq)
                        mtemp_5_para = np.mean(temp_5_para)
                        temp_4_compare_para[m] = mtemp_5_para - mpara
                    # keep the best choice
                    temp_4_chan_index = np.max(np.where(temp_4_compare_para == np.max(temp_4_compare_para)))
                    temp_3_chan_index.append(str(temp_4_chan_index))
                    temp_3_compare_para[l] = temp_4_compare_para[temp_4_chan_index]
                # judge if there's improvement
                if np.max(temp_3_compare_para) > np.max(para_change):  # has improvement
                    # find index
                    delete_chan_index = np.max(np.where(temp_3_compare_para == np.max(temp_3_compare_para)))
                    add_chan_index = int(temp_3_chan_index[delete_chan_index])
                    # operate (refresh data)
                    del remain_chans[delete_chan_index]
                    remain_chans.append(chans[add_chan_index])
                    chan_index = add_chan_index
                    para_change.append(temp_3_compare_para[delete_chan_index])
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
                    del temp_3_chans, temp_3_compare_para, temp_3_chan_index
                    del temp_4_chans, temp_4_data, temp_4_w, temp_4_compare_para, temp_4_chan_index
                    del temp_5_data, temp_5_w, temp_5_extract, mtemp_5_para, temp_5_para
                    # significant loop mark
                    print('Complete ' + str(j) + 'th loop')
                # no improvement
                else:
                    # release RAM
                    del temp_3_chans, temp_3_compare_para, temp_3_chan_index
                    del temp_4_chans, temp_4_data, temp_4_w, temp_4_compare_para, temp_4_chan_index
                    del temp_5_data, temp_5_w, temp_5_extract, mtemp_5_para, temp_5_para
                    # reset
                    print('Complete ' + str(j) + 'th loop')
                    print("Already best in " + str(j) + " channels' condition!")    
        j += 1    
    remain_chans = remain_chans[:len(remain_chans)-1]
    return remain_chans, para_change

def stepwise_SRCA_fs(chans, mfs, w, w_target, signal_data, data_target, regression):
    '''
    Parameters:
        chans: list of channels; the list order corresponds to the data array's
        mfs: float; the mean of original signal's fisher score
        w: background part input data array (n_events, n_trials, n_chans, n_times)
        w_target: background part target data array (n_events, n_trials, n_times)
        signal_data: signal part input data array (n_events, n_trials, n_chans, n_times)
        data_target: signal part target data array (n_events, n_trials, n_times)
    Returns:
        model_chans: list of channels which should be used in MCEE
        para_change: list of fisher score's alteration
    '''
    print('Running Stepwise SRCA...')
    start = time.perf_counter()   
    j = 1   
    max_loop = len(chans)   
    remain_chans = []
    snr_change = []
    core_data = []
    core_w = []
    active = True
    n_trials = w.shape[1]
    n_times_w = w.shape[-1]
    n_times_s = signal_data.shape[-1]
    while active and len(chans) <= max_loop:
        compare_snr = np.zeros((len(chans)))
        mtemp_snr = np.zeros((len(chans)))
        for i in range(len(chans)):
            temp_extract = np.zeros_like(data_target)
            if j == 1:
                temp_w = w[:, :, i,:]
                temp_data = signal_data[:, :, i, :]
                for kk in range(2):
                    temp_extract[kk, :, :] = mlr(temp_w[kk, :, :], w_target[kk, :, :],
                    temp_data[kk, :, :], data_target[kk, :, :], regression)
            else:
                temp_w = np.zeros((j, 2, n_trials, n_times_w))
                temp_w[:-1, :, :, :] = core_w
                temp_w[-1, :, :, :] = w[:, :, i, :]                
                temp_data = np.zeros((j, 2, n_trials, n_times_s))
                temp_data[:-1, :, :, :] = core_data
                temp_data[-1, :, :, :] = signal_data[:, :, i, :]
                for kk in range(2):
                    temp_extract[kk, :, :] = mlr(temp_w[:, kk, :, :], w_target[kk, :, :],
                    temp_data[:, kk, :, :], data_target[kk, :, :], regression)
            mtemp_snr[i] = np.mean(fisher_score(temp_extract))
            compare_snr[i] = mtemp_snr[i] - mfs
        chan_index = np.max(np.where(compare_snr == np.max(compare_snr)))
        remain_chans.append(chans.pop(chan_index))
        snr_change.append(np.max(compare_snr))    
        if j == 1: 
            core_w = w[:, :, chan_index, :]
            core_data = signal_data[:, :, chan_index, :]
            signal_data = np.delete(signal_data, chan_index, axis=2)
            w = np.delete(w ,chan_index, axis=2)
            print('Complete ' + str(j) + 'th loop')         
        if j == 2:  
            temp_core_w = np.zeros((j, 2, n_trials, n_times_w))
            temp_core_w[0, :, :, :] = core_w
            temp_core_w[1, :, :, :] = w[:, :, chan_index, :]
            core_w = copy.deepcopy(temp_core_w)
            temp_core_data = np.zeros((j, 2, n_trials, n_times_s))
            temp_core_data[0, :, :, :] = core_data
            temp_core_data[1, :, :, :] = signal_data[:, :, chan_index, :]
            core_data = copy.deepcopy(temp_core_data)
            if snr_change[-1] < np.max(snr_change):
                print('Stepwise EE complete!')
                end = time.perf_counter()
                print('Recursive running time: ' + str(end - start) + 's')
                active = False
            else:
                signal_data = np.delete(signal_data, chan_index, axis=2)
                w = np.delete(w, chan_index, axis=2)
                temp_1_chans = copy.deepcopy(remain_chans)
                temp_1_data = copy.deepcopy(core_data)
                temp_1_w = copy.deepcopy(core_w)
                del temp_1_chans[0]
                temp_1_data = np.delete(temp_1_data, 0, axis=0)
                temp_1_w = np.delete(temp_1_w, 0, axis=0)
                temp_2_compare_snr = np.zeros((signal_data.shape[2]))
                for k in range(signal_data.shape[2]):
                    temp_2_w = np.zeros((2, 2, n_trials, n_times_w))
                    temp_2_w[0, :, :, :] = temp_1_w
                    temp_2_w[1, :, :, :] = w[:, :, k, :]
                    temp_2_data = np.zeros((2, 2, n_trials, n_times_s))
                    temp_2_data[0, :, :, :] = temp_1_data
                    temp_2_data[1, :, :, :] = signal_data[:, :, k, :]
                    temp_2_extract = np.zeros_like(data_target)
                    for kk in range(2):
                        temp_2_extract[kk, :, :] = mlr(temp_2_w[:, kk, :, :],
                            w_target[kk, :, :], temp_2_data[:, kk, :, :],
                            data_target[kk, :, :], regression)
                    mtemp_2_snr = np.mean(fisher_score(temp_2_extract))
                    temp_2_compare_snr[k] = mtemp_2_snr - mfs
                temp_2_chan_index = np.max(np.where(temp_2_compare_snr == np.max(temp_2_compare_snr)))
                if temp_2_compare_snr[temp_2_chan_index] > snr_change[-1]:
                    chan_index = temp_2_chan_index
                    remain_chans.append(chans.pop(chan_index))
                    snr_change.append(temp_2_compare_snr[temp_2_chan_index])
                    core_w = np.delete(core_w, 0, axis=0)
                    temp_2_core_w = np.zeros((2, 2, n_trials, n_times_w))
                    temp_2_core_w[0, :, :, :] = core_w
                    temp_2_core_w[1, :, :, :] = w[:, :, chan_index, :]
                    core_w = copy.deepcopy(temp_2_core_w)
                    core_data = np.delete(core_data, 0, axis=0)
                    temp_2_core_data = np.zeros((2, 2, n_trials, n_times_s))
                    temp_2_core_data[0, :, :, :] = core_data
                    temp_2_core_data[1, :, :, :] = signal_data[:, :, chan_index, :]
                    core_data = copy.deepcopy(temp_2_core_data)
                    signal_data = np.delete(signal_data, chan_index, axis=2)
                    w = np.delete(w, chan_index, axis=2)
                    del remain_chans[0]
                    print('Complete ' + str(j) + 'th loop')               
                else:
                    print("Already best in 2 channels' contidion!")
        if j > 2:
            temp_core_w = np.zeros((j, 2, n_trials, n_times_w))
            temp_core_w[:-1, :, :, :] = core_w
            temp_core_w[-1, :, :, :] = w[:, :, chan_index, :]
            core_w = copy.deepcopy(temp_core_w)
            temp_core_data = np.zeros((j, 2, n_trials, n_times_s))
            temp_core_data[:-1, :, :, :] = core_data
            temp_core_data[-1, :, :, :] = signal_data[:, :, chan_index, :]
            core_data = copy.deepcopy(temp_core_data)
            if snr_change[-1] < np.max(snr_change):
                print('Stepwise EE complete!')
                end = time.perf_counter()
                print('Recursive running time: ' + str(end - start) + 's')
                active = False
            else:
                signal_data = np.delete(signal_data, chan_index, axis=2)
                w = np.delete(w, chan_index, axis=2)
                temp_3_chans = copy.deepcopy(remain_chans)
                temp_3_compare_snr = np.zeros((len(temp_3_chans)-1))
                temp_3_chan_index = []
                for l in range(len(temp_3_chans)-1):
                    temp_4_chans = copy.deepcopy(remain_chans)
                    temp_4_data = copy.deepcopy(core_data)
                    temp_4_w = copy.deepcopy(core_w)
                    del temp_4_chans[l]
                    temp_4_data = np.delete(temp_4_data, l, axis=0)
                    temp_4_w = np.delete(temp_4_w, l, axis=0)
                    temp_4_compare_snr = np.zeros((signal_data.shape[2]))
                    for m in range(signal_data.shape[2]):
                        temp_5_w = np.zeros((j, 2, n_trials, n_times_w))
                        temp_5_w[:-1, :, :, :] = temp_4_w
                        temp_5_w[-1, :, :, :] = w[:, :, m, :]               
                        temp_5_data = np.zeros((j, 2, n_trials, n_times_s))
                        temp_5_data[:-1, :, :, :] = temp_4_data
                        temp_5_data[-1, :, :, :] = signal_data[:, :, m, :]
                        temp_5_extract = np.zeros_like(data_target)
                        for kk in range(2):
                            temp_5_extract[kk, :, :] = mlr(temp_5_w[:, kk, :, :],
                                w_target[kk, :, :], temp_5_data[:, kk, :, :],
                                data_target[kk, :, :], regression)
                        mtemp_5_snr = np.mean(fisher_score(temp_5_extract))
                        temp_4_compare_snr[m] = mtemp_5_snr - mfs
                    temp_4_chan_index = np.max(np.where(temp_4_compare_snr == np.max(temp_4_compare_snr)))
                    temp_3_chan_index.append(str(temp_4_chan_index))
                    temp_3_compare_snr[l] = temp_4_compare_snr[temp_4_chan_index]
                if np.max(temp_3_compare_snr) > np.max(snr_change):
                    delete_chan_index = np.max(np.where(temp_3_compare_snr == np.max(temp_3_compare_snr)))
                    add_chan_index = int(temp_3_chan_index[delete_chan_index])
                    del remain_chans[delete_chan_index]
                    remain_chans.append(chans[add_chan_index])
                    chan_index = add_chan_index
                    snr_change.append(temp_3_compare_snr[delete_chan_index])
                    core_w = np.delete(core_w, delete_chan_index, axis=0)                   
                    temp_6_core_w = np.zeros((core_w.shape[0]+1, 2, core_w.shape[2], core_w.shape[-1]))
                    temp_6_core_w[:-1, :, :, :] = core_w
                    temp_6_core_w[-1, :, :, :] = w[:, :, add_chan_index, :]
                    core_w = copy.deepcopy(temp_6_core_w)
                    core_data = np.delete(core_data, delete_chan_index, axis=0)              
                    temp_6_core_data = np.zeros((core_data.shape[0]+1, 2, core_data.shape[2], core_data.shape[-1]))
                    temp_6_core_data[:-1, :, :, :] = core_data
                    temp_6_core_data[-1, :, :, :] = signal_data[:, :, add_chan_index, :]
                    core_data = copy.deepcopy(temp_6_core_data)
                    signal_data = np.delete(signal_data, add_chan_index, axis=2)
                    w = np.delete(w, add_chan_index, axis=2)
                    del chans[add_chan_index]
                    print('Complete ' + str(j) + 'th loop')
                else:
                    print('Complete ' + str(j) + 'th loop')
                    print("Already best in " + str(j) + " channels' condition!")    
        j += 1    
    remain_chans = remain_chans[:len(remain_chans)-1]
    return remain_chans, snr_change


#%% Canonical Correlation Analysis
def sCCA():
    pass

#%% Target identification: TRCA method
def TRCA_compute(data):
    '''
    Task-related component analysis (TRCA)

    Parameters
    ----------
    data : (n_events, n_trials, n_chans, n_times)
        input data array (default z-scored after bandpass filtering).

    Returns
    -------
    w : (n_events, n_chans)
        eigenvector refering to the largest eigenvalue.

    '''
    n_events = data.shape[0]
    n_trials= data.shape[1]
    n_chans = data.shape[2]
    n_times = data.shape[-1]
    # spatial filter W initialization
    w = np.zeros((n_events, n_chans))
    for ne in range(n_events):
        # matrix Q: inter-channel covariance
        q = np.zeros((n_chans, n_chans))
        temp_Q = data[ne,...].swapaxes(0,1).reshape((n_chans,-1), order='C')
        q = np.dot(temp_Q, temp_Q.T) / (n_trials*n_times)
        # matrix S: inter-channels' inter-trial covariance
        s = np.zeros_like(q)
        for nt_i in range(n_trials):
            for nt_j in range(n_trials):
                if nt_i != nt_j:
                    data_i = data[ne, nt_i,...]
                    data_j = data[ne, nt_j,...]
                    s += np.dot(data_i, data_j.T)/(n_times)
        # generalized eigenvalue problem
        e_va, e_vec = LA.eig(np.dot(LA.inv(q), s))
        w_index = np.max(np.where(e_va == np.max(e_va)))
        #w[ne, :] = e_vec[:, e_va.argsort()[::-1]][:,0].T
        w[ne, :] = e_vec[:, w_index].T
        #e_value = np.array(sorted(e_va, reverse=True))
    print('TRCA spatial filter complete!')
    return w

# filter-bank TRCA
def fb_TRCA(tr_fb_data, te_fb_data):
    '''
    TRCA is the method that extracts task-related components efficiently 
        by maximizing the reproducibility during the task period
    Parameters:
        tr_fb_data: (n_events, n_bands, n_trials, n_chans, n_times) |
            training dataset (after filter bank) 
        te_fb_data: (n_events, n_bands, n_trials, n_chans, n_times) |
            test dataset (after filter bank)
    Returns:
        accuracy: int | the number of correct identifications
        
    '''
    # template data: (n_events, n_bands, n_chans, n_times)|basic element: (n_chans, n_times)
    template = np.mean(tr_fb_data, axis=2)  
    # basic parameters
    n_events = tr_fb_data.shape[0]
    n_bands = tr_fb_data.shape[1]
    n_chans = tr_fb_data.shape[3]
    n_times = tr_fb_data.shape[4]   
    # Matrix Q: inter-channel covariance
    q = np.zeros((n_events, n_bands, n_chans, n_chans))
    # all events(n), all bands(m)
    for x in range(n_events):  # x for events (n)
        for y in range(n_bands):  # y for bands (m)
            temp = np.zeros((n_chans, int(tr_fb_data.shape[2]*n_times)))
            for z in range(n_chans):  # z for channels
                # concatenated matrix of all trials in training dataset
                temp[z,:] = tr_fb_data[x,y,:,z,:].flatten()
            # compute matrix Q | (Toeplitz matrix): (n_chans, n_chans)
            # for each event & band, there should be a unique Q
            # so the total quantity of Q is n_bands*n_events (here is 30=x*y)
            q[x,y,:,:] = np.cov(temp)
            del temp, z
        del y
    del x
    # Matrix S: inter-channels' inter-trial covariance
    # all events(n), all bands(m), inter-channel(n_chans, n_chans)
    s = np.zeros((n_events, n_bands, n_chans, n_chans))
    for u in range(n_events):  # u for events
        for v in range(n_bands):  # v for bands
            # at the inter-channels' level, obviouly the square will also be a Toeplitz matrix
            # i.e. (n_chans, n_chans), here the shape of each matrix should be (9,9)
            for w in range(n_chans):  # w for channels (j1)
                for x in range(n_chans):  # x for channels (j2)
                    cov = []
                    # for each event & band & channel, there should be (trials^2-trials) values
                    # here trials = 10, so there should be 90 values in each loop
                    for y in range(tr_fb_data.shape[2]):  # y for trials (h1)
                        temp = np.zeros((2, n_times))
                        temp[0,:] = tr_fb_data[u,v,y,w,:]
                        for z in range(tr_fb_data.shape[2]):  # z for trials (h2)
                            if z != y:  # h1 != h2, INTER-trial covariance
                                temp[1,:] = tr_fb_data[u,v,z,x,:]
                                cov.append(np.sum(np.tril(np.cov(temp), -1)))
                            else:
                                continue
                        del z, temp
                    del y
                    # the basic element S(j1j2) of Matrix S
                    # is the sum of inter-trial covariance (h1&h2) of 1 events & 1 band in 1 channel
                    # then form a square (n_chans,n_chans) to describe inter-channels' information
                    # then form a data cube containing various bands and events' information      
                    # of course the 1st value should be the larger one (computed in 1 channel)
                    # according to the spatial location of different channels
                    # there should also be size differences
                    # (e.g. PZ & POZ's values are significantly larger)
                    s[u,v,w,x] = np.sum(cov)
                    del cov
                del x
            del w
        del v
    del u   
    # Spatial filter W
    # all events(n), all bands(m)
    w = np.zeros((n_events, n_bands, n_chans))
    for y in range(n_events):
        for z in range(n_bands):
            # Square Q^-1 * S
            qs = np.mat(q[y,z,:,:]).I * np.mat(s[y,z,:,:])
            # Eigenvalues & eigenvectors
            e_value, e_vector = np.linalg.eig(qs)
            # choose the eigenvector which refers to the largest eigenvalue
            w_index = np.max(np.where(e_value == np.max(e_value)))
            # w will maximum the task-related componont from multi-channel's data
            w[y,z,:] = e_vector[:,w_index].T
            del w_index
        del z
    del y
    # from now on, never use w as loop mark because we have variable named w
    # Test dataset operating
    # basic element of r is (n_bands, n_events)
    r = np.zeros((n_events, te_fb_data.shape[2], n_bands, n_events))
    for v in range(n_events): # events in test dataset
        for x in range(te_fb_data.shape[2]):  # trials in test dataset (of one event)
            for y in range(n_bands):  # bands are locked
                # (vth event, zth band, xth trial) test data to (all events(n), zth band(m)) training data
                for z in range(n_events):
                    temp_test = np.mat(te_fb_data[v,y,x,:,:]).T * np.mat(w[z,y,:]).T
                    temp_template = np.mat(template[z,y,:,:]).T * np.mat(w[z,y,:]).T
                    r[v,x,y,z] = np.sum(np.tril(np.corrcoef(temp_test.T, temp_template.T),-1))
                del z, temp_test, temp_template
            del y
        del x
    del v
    # Feature for target identification
    r = r**2
    # identification function a(m)
    a = np.matrix([(m+1)**-1.25+0.25 for m in range(n_bands)])
    rou = np.zeros((n_events, te_fb_data.shape[2], n_events))
    for y in range(n_events):
        for z in range(te_fb_data.shape[2]):  # trials in test dataset (of one event)
            # (yth event, zth trial) test data | will have n_events' value, here is 3
            # the location of the largest value refers to the class of this trial
            rou[y,z,:] = a * np.mat(r[y,z,:,:])    
    accuracy = []
    # compute accuracy
    for x in range(rou.shape[0]):  # ideal classification
        for y in range(rou.shape[1]):
            if np.max(np.where(rou[x,y,:] == np.max(rou[x,y,:]))) == x:  # correct
                accuracy.append(1)   
    return accuracy

# TRCA for origin data
def TRCA_off(train_data, test_data):
    '''
    TRCA with target identification process in offline situation

    Parameters
    ----------
    train_data : (n_events, n_trials, n_chans, n_times)
        training dataset.
    test_data : (n_events, n_trials, n_chans, n_times)
        test dataset.

    Returns
    -------
    accuracy : int
        the number of correct identifications.

    '''
    # template data: (n_events, n_chans, n_times)
    template = train_data.mean(axis=1)
    # basic parameters
    n_events = train_data.shape[0]
    n_tests = test_data.shape[1]
    # spatial filter W: (n_events, n_chans)
    w = TRCA_compute(train_data)
    # target identification
    r = np.zeros((n_events, n_tests, n_events))
    for nete in range(n_events):  # n_events in test dataset
        for nte in range(n_tests):
            for netr in range(n_events):  # n_events in training dataset
                temp_test = np.dot(w[netr, :], test_data[nete, nte, ...])
                temp_template = np.dot(w[netr, :], template[netr, ...])
                r[nete, nte, netr] = np.sum(np.tril(CORR(temp_test, temp_template),-1))
    accuracy = []
    # compute accuracy
    for ne in range(n_events):
        for nt in range(n_tests):
            if np.max(np.where(r[ne, nt, :] == np.max(r[ne, nt, :]))) == ne:
                accuracy.append(1)
    accuracy = np.sum(accuracy) / (n_events*n_tests)
    print('TRCA identification complete!')
    return accuracy

# correlation between 2-D matrices
def pearson_corr2(dataA, dataB):
    '''
    Compute Pearson Correlation Coefficients between 2D matrices

    Parameters
    ----------
    dataA : (n_chans, n_times)
    dataB : (n_chans, n_times)

    Returns
    -------
    corr2 : float
        2-D correlation coefficient.
    '''
    # basic information
    n_chans = dataA.shape[0]
    n_times = dataA.shape[-1]
    # 2D correlation
    numerator = 0
    denominatorA = 0
    denominatorB = 0
    meanA = dataA.mean()  # the mean of all values in matrix A
    meanB = dataB.mean()  # the same
    for nc in range(n_chans):
        for nt in range(n_times):
            numerator += (dataA[nc, nt] - meanA) * (dataB[nc, nt] - meanB)
            denominatorA += (dataA[nc, nt] - meanA)**2
            denominatorB += (dataB[nc, nt] - meanB)**2
    corr2 = numerator / np.sqrt(denominatorA * denominatorB)
    return corr2

# TRCA for SRCA data
def SRCA_TRCA(train_data, test_data, tar_chans, model_chans, chans,
              regression='OLS', alpha=1.0, l1_ratio=1.0, sp=1140):
    '''
    TRCA for SRCA data
    class A using SRCA model A should be more adapted then class B using model A

    Parameters
    ----------
    train_data : (n_events, n_trials, n_chans, n_times)
        training dataset.
    test_data : (n_events, n_trials, n_chans, n_times)
        test dataset.
    tar_chans : list
        names of target channels.
    model_chans : list
        names of SRCA channels for all target channels.
    chans : list
        names of all channels.
    regression : str, optional
        OLS, Ridge, Lasso or ElasticNet regression. The default is 'OLS'.
    alpha : float, optional
        parameters used in Ridge, Lasso and EN regression. The default is 1.0.
    l1_ratio : float, optional
        parameters used in EN regression. The default is 1.0.
    sp : int, optional
        start point of mission state. The default is 1140.

    Returns
    -------
    accuracy : int
        the number of correct identifications.
    '''
    # config correct srca process on training dataset
    n_events = train_data.shape[0]
    n_trains = train_data.shape[1]
    n_chans = len(tar_chans)
    n_times = train_data.shape[-1] - sp
    model_sig = np.zeros((n_events, n_trains, n_chans, n_times))
    for ne in range(n_events):
        temp_model = model_chans[ne::n_events]  # length: len(tar_chans)
        model_sig[ne, ...] = apply_SRCA(train_data[ne, ...], tar_chans,
        temp_model, chans, regression, sp)
    del ne, temp_model
    # apply different srca models on one trial's data
    n_tests = test_data.shape[1]
    target_sig = np.zeros((n_events, n_events, n_tests, n_chans, n_times))
    for ne_tr in range(n_events):      # n_events in SRCA model
        temp_model = model_chans[ne_tr::n_events]
        for ne_re in range(n_events):  # n_events in real data
            target_sig[ne_tr, ne_re, ...] = apply_SRCA(test_data[ne_re, ...],
            tar_chans, temp_model, chans, regression, sp)
    del ne_tr, ne_re, n_chans, n_times
    # template data: (n_events, n_chans, n_times)
    template = model_sig.mean(axis=1)
    # Spatial filter W: (n_events, n_chans)
    w = TRCA_compute(model_sig)
    # target identification
    r = np.zeros((n_events, n_tests, n_events))  # (n_events srca, n_tests, n_events test)
    for nes in range(n_events):                  # n_events in srca model
        for nte in range(n_tests):               # n_tests
            for ner in range(n_events):          # n_events in real test dataset
                temp_test = np.dot(w[nes, :], target_sig[nes, ner, nte,...])
                temp_template = np.dot(w[nes, :], template[nes, ...])
                r[nes, nte, ner] = np.sum(np.tril(CORR(temp_test, temp_template),-1))
    # compute accuracy
    accuracy = []
    for nes in range(n_events):
        for nte in range(n_tests):
            if np.max(np.where(r[nes, nte, :] == np.max(r[nes, nte, :]))) == nes:
                accuracy.append(1)
    accuracy = np.sum(accuracy) / (n_events*n_tests)
    print('SRCA TRCA identification complete!')
    return accuracy

# ensemble TRCA for origin data
def eTRCA(train_data, test_data):
    '''
    ensemble-TRCA without filter banks for original signal
    Parameters:
        train_data: (n_events, n_trials, n_chans, n_times) | training dataset
        test_data: (n_events, n_trials, n_chans, n_times) | test dataset
    Returns:
        accuracy: int | the number of correctt identification
    '''
    # template data: (n_events, n_chans, n_times)
    template = train_data.mean(axis=1)
    # basic parameters
    n_events = train_data.shape[0]
    n_tests = test_data.shape[1]
    # spatial filter W: (n_events, n_chans)
    w = TRCA_compute(train_data)
    # target identification
    r = np.zeros((n_events, n_tests, n_events))
    for nete in range(n_events):
        for nte in range(n_tests):
            for netr in range(n_events):
                temp_test = np.dot(w, test_data[nete, nte, ...])
                temp_template = np.dot(w, template[netr, ...])
                r[nete, nte, netr] = pearson_corr2(temp_test, temp_template)
    accuracy = []
    for x in range(n_events):
        for y in range(n_tests):
            if np.max(np.where(r[x,y,:] == np.max(r[x,y,:]))) == x:  # correct
                accuracy.append(1)
    accuracy = np.sum(accuracy) / (n_events*n_tests)
    print('eTRCA identification complete!')
    return accuracy

# ensembel TRCA for SRCA data
def SRCA_eTRCA(train_data, test_data, tar_chans, model_chans, chans,
               regression='OLS', alpha=1.0, l1_ratio=1.0, sp=1140):
    '''
    Ensemble-TRCA without filter banks
    Parameters:
        train_data: (n_events, n_trials, n_chans, n_times) | training dataset
        test_data: (n_events, n_trials, n_chans, n_times) | test dataset
        tar_chans: str list | names of target channels
        model_chans: str list | names of SRCA channels for all target channels
        chans: str list | names of all channels
        regression: str | OLS, Ridge, Lasso or ElasticNet regression, default OLS
        alpha: float | default 1.0, parameters used in Ridge, Lasso and EN regression
        l1_ratio: float | default 1.0, parameters used in EN regression
        sp: int | start point of mission state (default 1140)
    Returns:
        accuracy: int | the number of correct identifications
    '''  
    # config correct srca process on training dataset
    n_events = train_data.shape[0]
    n_trains = train_data.shape[1]
    n_chans = len(tar_chans)
    n_times = train_data.shape[-1] - sp
    model_sig = np.zeros((n_events, n_trains, n_chans, n_times))
    for ne in range(n_events):
        temp_model = model_chans[ne::n_events]  # length: len(tar_chans)
        model_sig[ne, :, :, :] = apply_SRCA(train_data[ne, :, :, :], tar_chans,
                                            temp_model, chans, regression, sp)
    del ne, temp_model
    # apply different srca models on one trial's data
    n_tests = test_data.shape[1]
    # (n_events srca, n_events test, n_tests, n_chans, n_times)
    target_sig = np.zeros((n_events, n_events, n_tests, n_chans, n_times))
    for nes in range(n_events):  # n_events in SRCA model
        temp_model = model_chans[nes::n_events]
        for ner in range(n_events):
            target_sig[nes, ner, :, :, :] = apply_SRCA(test_data[ner, :, :, :],
                                        tar_chans, temp_model, chans, regression, sp)
        del ner
    del nes, n_chans, n_times
    # template data: (n_events, n_chans, n_times)
    template = np.mean(model_sig, axis=1)
    # Matrix Q: (n_events, n_chans, n_chans) | inter-channel covariance
    q = matrix_Q(model_sig)
    print('Matrix Q complete!')
    # Matrix S: (n_events, n_chans, n_chans) | inter-channels' inter-trial covariance
    s = matrix_S(model_sig)
    print('Matrix S complete!')
    # Spatial filter W: (n_events, n_chans)
    w = spatial_W(q, s)
    print('Spatial filter complete!')
    # Ensemble target identification
    r = np.zeros((n_events, n_tests, n_events))  # (n_events srca, n_tests, n_events test)
    for nes in range(n_events):  # n_events in srca model
        for nte in range(n_tests):  # n_tests
            for ner in range(n_events):  # n_events in test dataset
                temp_test = np.mat(w) * np.mat(target_sig[nes, ner, nte, :, :])
                temp_template = np.mat(w) * np.mat(template[nes, :, :])
                r[nes, nte, ner] = pearson_corr2(temp_test, temp_template)
            del ner
        del nte
    del nes
    # Compute accuracy
    accuracy = []
    for nes in range(n_events):
        for nte in range(n_tests):
            if np.max(np.where(r[nes, nte, :] == np.max(r[nes, nte, :]))) == nes:
                accuracy.append(1)
    print('SRCA eTRCA identification complete!')
    return accuracy


# Target identification: DCPM
# discrimination index 1: 2D correlation
def di1(dataA, dataB):
    '''
    Correlation coefficient of 2D matrices
    Parameters:
        dataA & dataB: (n_chans, n_times)
    Return:
        coef: float
    '''
    coef = pearson_corr2(dataA, dataB)
    return coef

# discrimination index 2: Euclidean distance
def di2(dataA, dataB):
    '''
    Euclidean of 2D matrices
    Parameters:
        dataA & dataB: (n_chans, n_times)
    Return:
        coef: float
    '''
    dataA, dataB = dataA.A, dataB.A
    coef = np.sqrt(np.sum((dataA - dataB)**2))
    return coef

# discrimination index 3: CCA coefficient
def di3(dataA, dataB):
    '''
    Standard CCA coefficient using unified framework
    Parameters:
        dataA & dataB: (n_chans, n_times)
    Return:
        coef: float
    '''
    x = np.mat(np.swapaxes(dataA, 0, 1))       # (n_times, n_chans)
    y = np.mat(np.swapaxes(dataB, 0, 1))       # (n_times, n_chans)
    # filter for template: u
    projectionX = y * (y.T * y).I * y.T        # (n_chans, n_chans)
    matrixAX = x.T * projectionX * x           # (n_chans, n_chans)
    matrixBX = x.T * x                         # (n_chans, n_chans)
    baX = matrixBX.I * matrixAX                # GEPs
    e_valueX, e_vectorX = np.linalg.eig(baX)
    u_index = np.max(np.where(e_valueX == np.max(e_valueX)))
    u = e_vectorX[:, u_index].T                # (1, n_chans)
    del u_index, projectionX, matrixAX, matrixBX, baX, e_valueX
    # filter for data: v
    projectionY = x * (x.T * x).I * x.T        # (n_chans, n_chans)
    matrixAY = y.T * projectionY * y           # (n_chans, n_chans)
    matrixBY = y.T * y                         # (n_chans, n_chans)
    baY = matrixBY.I * matrixAY                # GEPs
    e_valueY, e_vectorY = np.linalg.eig(baY)
    v_index = np.max(np.where(e_valueY == np.max(e_valueY)))
    v = e_vectorY[:, v_index].T                # (1, n_chans)
    del v_index, projectionY, matrixAY, matrixBY, baY, e_valueY
    # compute coefficient
    coef = np.sum(np.tril(np.corrcoef(u*dataA, v*dataB), -1))
    return coef

# discrimination index 4: 1D correlation & CCA coefficient (template filter)
def di4(dataA, dataB):
    pass

# discrimination index 5: 1D correlation & CCA coefficient (data filter)
def di5(dataA, dataB):
    pass

# DCPM for origin data
def DCPM(train_data, test_data, di=['1','2','3','4','5']):
    '''
    Discriminative canonical pattern matching algorithm (DCPM) for origin data
    Parameters:
        train_data: (n_events, n_trials, n_chans, n_times) | training dataset
        test_data: (n_events, n_trials, n_chans, n_times) | test dataset
        di: discrimination index | 5 different indices, recommended 1 or 2
    Returns:
        acc: int | the number of correct identifications
    '''
    # basic information
    n_events = train_data.shape[0]
    n_trains = train_data.shape[1]
    n_tests = test_data.shape[1]
    # pattern data preparation
    x1 = np.swapaxes(train_data[0, :, :, :], 0, 2)  # (n_times, n_chans, n_trials)
    x1 = np.swapaxes(x1, 0, 1)                     # (n_chans, n_times, n_trials)
    x2 = np.swapaxes(train_data[1, :, :, :], 0, 2)  # (n_times, n_chans, n_trials)
    x2 = np.swapaxes(x2, 0, 1)                     # (n_chans, n_times, n_trials)
    pattern1 = np.mat(np.mean(train_data[0, :, :, :], axis=0))  # (n_chans, n_times)
    pattern2 = np.mat(np.mean(train_data[1, :, :, :], axis=0))  # (n_chans, n_times)
    # covariance matrices
    sigma11 = pattern1 * pattern1.T  # (n_chans, n_chans)
    sigma12 = pattern1 * pattern2.T  # (n_chans, n_chans)
    sigma21 = pattern2 * pattern1.T  # (n_chans, n_chans)
    sigma22 = pattern2 * pattern2.T  # (n_chans, n_chans)
    # variance matrices
    var1, var2 = np.zeros_like(sigma11), np.zeros_like(sigma22)
    for i in range(n_trains):
        var1 += (x1[:,:,i] - pattern1) * (x1[:,:,i] - pattern1).T
        var2 += (x2[:,:,i] - pattern2) * (x2[:,:,i] - pattern2).T
    var1 /= (n_trains - 1)
    var2 /= (n_trains - 1)
    # discriminative spatial pattern (DSP) % projection W
    sb = sigma11 + sigma22 - sigma12 - sigma21    # (n_chans, n_chans)
    sw = var1 + var2                              # (n_chans, n_chans)
    e_value, e_vector = np.linalg.eig(sw.I * sb)  # (n_chans, n_chans)
    #e_index = e_value.argsort()  # rise-up sorting
    #n_loop, n_per = 0, 0
    #while n_loop < 9:
    #    n_per += e_value[e_index[n_loop]]/np.sum(e_value)
    w_index = np.max(np.where(e_value == np.max(e_value)))
    w = e_vector[:, w_index].T
    # apply DSP projection
    wx1 = w * pattern1             # (1, n_times)
    wx2 = w * pattern2             # (1, n_times)
    # target identification
    rou = np.zeros((n_events, n_tests, n_events))
    for nete in range(n_events):          # n_events in test dataset
        for nte in range(n_tests):
            if '1' in di:             # 2D correlation
                rou[nete, nte, 0] += di1(wx1, w*test_data[nete, nte, :, :])
                rou[nete, nte, 1] += di1(wx2, w*test_data[nete, nte, :, :])
            if '2' in di:             # Euclidean distance
                rou[nete, nte, 0] += di2(wx1, w*test_data[nete, nte, :, :])
                rou[nete, nte, 1] += di2(wx2, w*test_data[nete, nte, :, :])
            if '3' in di:             # CCA coefficient
                rou[nete, nte, 0] += di3(wx1, w*test_data[nete, nte, :, :])
                rou[nete, nte, 1] += di3(wx2, w*test_data[nete, nte, :, :])
            if '4' in di:             # Correlation & CCA (template)
                rou[nete, nte, 0] += di4(wx1, w*test_data[nete, nte, :, :])
                rou[nete, nte, 1] += di4(wx2, w*test_data[nete, nte, :, :])
            if '5' in di:             # Correlation & CCA (data)
                rou[nete, nte, 0] += di5(wx1, w*test_data[nete, nte, :, :])
                rou[nete, nte, 1] += di5(wx2, w*test_data[nete, nte, :, :])
        del nte
    del nete
    # compute accuracy
    acc = []
    for nete in range(n_events):
        for nte in range(n_tests):
            if np.max(np.where(rou[nete, nte, :] == np.max(rou[nete, nte, :]))) == nete:
                acc.append(1)
    print('DCPM indentification complete!')
    return acc

# DCPM for SRCA data
def SRCA_DCPM(train_data, test_data, tar_chans, model_chans, chans,
              regression='OLS', sp=1140, di=['1','2','3','4','5']):
    '''
    Discriminative canonical pattern matching algorithm (DCPM) for SRCA data
    Parameters:
        train_data: (n_events, n_trials, n_chans, n_times) | training dataset
        test_data: (n_events, n_trials, n_chans, n_times) | test dataset
        tar_chans: str list | names of target channels
        model_chans: str list | names of SRCA channels for all target channels
        chans: str list | names of all channels
        regression: str | OLS, Ridge, Lasso or ElasticNet regression, default OLS
        sp: int | start point of mission state (default 1140)
        di: discrimination index | 5 different indices
    Returns:
        acc: int | the number of correct identifications
    '''
    # config correct SRCA process on training/test dataset
    n_events = train_data.shape[0]
    n_trains = train_data.shape[1]
    n_tests = test_data.shape[1]
    n_chans = len(tar_chans)
    n_times = train_data.shape[-1] - sp
    model_sig = np.zeros((n_events, n_trains, n_chans, n_times))
    target_sig = np.zeros((n_events, n_tests, n_chans, n_times))
    for ne in range(n_events):
        model_sig[ne, :, :, :] = apply_SRCA(train_data[ne, :, :, :], tar_chans,
                                            model_chans, chans, regression, sp)
        target_sig[ne, :, :, :] = apply_SRCA(test_data[ne, :, :, :], tar_chans,
                                             model_chans, chans, regression, sp)
    del n_chans, n_times
    # pattern data preparation
    x1 = np.swapaxes(model_sig[0, :, :, :], 0, 2)  # (n_times, n_chans, n_trials)
    x1 = np.swapaxes(x1, 0, 1)                     # (n_chans, n_times, n_trials)
    x2 = np.swapaxes(model_sig[1, :, :, :], 0, 2)  # (n_times, n_chans, n_trials)
    x2 = np.swapaxes(x2, 0, 1)                     # (n_chans, n_times, n_trials)
    pattern1 = np.mat(np.mean(model_sig[0, :, :, :], axis=0))  # (n_chans, n_times)
    pattern2 = np.mat(np.mean(model_sig[1, :, :, :], axis=0))  # (n_chans, n_times)
    # covariance matrices
    sigma11 = pattern1 * pattern1.T  # (n_chans, n_chans)
    sigma12 = pattern1 * pattern2.T  # (n_chans, n_chans)
    sigma21 = pattern2 * pattern1.T  # (n_chans, n_chans)
    sigma22 = pattern2 * pattern2.T  # (n_chans, n_chans)
    # variance matrices
    var1, var2 = np.zeros_like(sigma11), np.zeros_like(sigma22)
    for i in range(n_trains):
        var1 += (x1[:,:,i] - pattern1) * (x1[:,:,i] - pattern1).T
        var2 += (x2[:,:,i] - pattern2) * (x2[:,:,i] - pattern2).T
    var1 /= (n_trains - 1)
    var2 /= (n_trains - 1)
    # discriminative spatial pattern (DSP) % projection W
    sb = sigma11 + sigma22 - sigma12 - sigma21    # (n_chans, n_chans)
    sw = var1 + var2                              # (n_chans, n_chans)
    e_value, e_vector = np.linalg.eig(sw.I * sb)  # (n_chans, n_chans)
    w_index = np.max(np.where(e_value == np.max(e_value)))
    w = e_vector[:, w_index].T
    # apply DSP projection
    wx1 = w * pattern1             # (1, n_times)
    wx2 = w * pattern2             # (1, n_times)
    # target identification
    rou = np.zeros((n_events, n_tests, n_events))
    for nete in range(n_events):          # n_events in test dataset
        for nte in range(n_tests):
            if '1' in di:             # 2D correlation
                rou[nete, nte, 0] += di1(wx1, w*target_sig[nete, nte, :, :])
                rou[nete, nte, 1] += di1(wx2, w*target_sig[nete, nte, :, :])
            if '2' in di:             # Euclidean distance
                rou[nete, nte, 0] += di2(wx1, w*target_sig[nete, nte, :, :])
                rou[nete, nte, 1] += di2(wx2, w*target_sig[nete, nte, :, :])
            if '3' in di:             # CCA coefficient
                rou[nete, nte, 0] += di3(wx1, w*target_sig[nete, nte, :, :])
                rou[nete, nte, 1] += di3(wx2, w*target_sig[nete, nte, :, :])
            if '4' in di:             # Correlation & CCA (template)
                rou[nete, nte, 0] += di4(wx1, w*target_sig[nete, nte, :, :])
                rou[nete, nte, 1] += di4(wx2, w*target_sig[nete, nte, :, :])
            if '5' in di:             # Correlation & CCA (data)
                rou[nete, nte, 0] += di5(wx1, w*target_sig[nete, nte, :, :])
                rou[nete, nte, 1] += di5(wx2, w*target_sig[nete, nte, :, :])
        del nte
    del nete
    # compute accuracy
    acc = []
    for nete in range(n_events):
        for nte in range(n_tests):
            if np.max(np.where(rou[nete, nte, :] == np.max(rou[nete, nte, :]))) == nete:
                acc.append(1)
    print('SRCA DCPM indentification complete!')
    return acc


#%% Correlation detect for single-channel data
def corr_detect(test_data, template):
    '''
    Offline Target identification for single-channel data
        (using Pearson correlation coefficients)
    Parameters:
        test_data: array | (n_events, n_trials, n_times)
        template: array | (n_events, n_times)
    Returns:
        acc: int | the total number of correct identifications
        mr: square | (n_events (test dataset), n_events (template)),
            the mean of Pearson correlation coefficients between N th events'
            test data and M th events' template
        rou: array | (n_events (test dataset), n_trials, n_events (template))
            details of mr
    '''
    # initialization
    n_events = template.shape[0]
    n_trials = test_data.shape[1]
    acc = []
    mr = np.zeros((n_events, n_events))
    rou = np.zeros((n_events, n_trials, n_events))   
    # compute Pearson correlation coefficients & target identification
    for nete in range(n_events):  # n_events' loop in test dataset
        for ntte in range(n_trials):  # n_trials' loop in test dataset
            for netr in range(n_events):  # n_events' loop in template (training dataset)
                rou[nete,ntte,netr] = np.sum(np.tril(np.corrcoef(test_data[nete,ntte,:],template[netr,:]),-1))
            del netr
            if np.max(np.where([rou[nete,ntte,:] == np.max(rou[nete,ntte,:])])) == nete:  # correct
                acc.append(1)
        del ntte
    del nete
    acc = np.sum(acc) 
    for j in range(n_events):
        for k in range(n_events):
            mr[j,k] = np.mean(rou[j,:,k])
    return acc, mr, rou