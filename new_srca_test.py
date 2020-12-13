'''
Author: Brynhildr Wu
Date: 2020-11-14 23:20:18
LastEditTime: 2020-11-15 00:06:21
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: \SSVEP-signal-processing\new_srca_test.py
'''
# -*- coding: utf-8 -*-
# %%
import numpy as np
from numpy import linalg as LA
from numpy import corrcoef as CORR
from numpy import newaxis as NA

from sklearn import linear_model

import copy
import time
from math import pi

# spatial regression component analysis (main function)
def srca(model_input, model_target, data_input, data_target, regression='OLS',
        alpha=1.0, l1_ratio=1.0):
    '''
    the main process of spatial regression component analysis (SRCA)

    Parameters
    ----------
    model_input : (n_trials, n_chans, n_times)
        rest-state data of regression channels.
    model_target : (n_trials, n_times)
        rest-state data of target channel.
    data_input : (n_trials, n_chans, n_times)
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
    n_trials = data_input.shape[0]
    n_times = data_input.shape[-1]
    estimate = np.zeros((n_trials, n_times))  # estimate signal
    for i in range(n_trials):  # basic operating unit: (n_times, n_chans), (n_times, 1)
        if regression == 'OLS':
            L = linear_model.LinearRegression().fit(model_input[i, ...].T, model_target[i, :].T)
        elif regression == 'Ridge':
            L = linear_model.Ridge(alpha=alpha).fit(model_input[i, ...].T, model_target[i, :].T)
        elif regression == 'Lasso':
            L = linear_model.Lasso(alpha=alpha).fit(model_input[i, ...].T, model_target[i, :].T)
        elif regression == 'ElasticNet':
            L = linear_model.ElasticNet(alpha=alpha, l1_ratio=l1_ratio).fit(model_input[i, ...].T,
            model_target[i, :].T)
        RI = L.intercept_
        RC = L.coef_
        estimate[i, :] = np.dot(RC, data_input[i, ...]) + RI
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
        w_ex_s = srca(model_input=w_i, model_target=w_o, data_input=sig_i,
                      data_target=sig_o, regression=regression)
        f_data[:, ntc, :] = w_ex_s
    return f_data


# %% new srca test
def SRCA_train(chans, para, w_in, w_target, data_in, data_target, method='SNR',
                regression='OLS', alpha=1.0, l1_ratio=1.0, freq=None, phase=None, sfreq=1000):
    '''
    Stepwise recursive algorithm to train SRCA model
    The combination of Forward and Backward process:
        (1) form an empty set; 
        (2) add one channel respectively and pick the best one; 
        (3) add one channel respectively and delete one respectively (except the just-added one)
            keep the best choice;
        (4) repeat (3) until there will be no better choice
            i.e. the convergence point of the recursive algorithm

    Parameters
    ----------
    chans : list
        the list order corresponds to the data array's.
    para : float
        the mean of original signal's parameters in time domain.
    w_in : (n_trials, n_chans, n_times)
        rest-state input data array.
    w_target : (n_trials, n_times)
        rest-state target data array.
    data_in : (n_trials, n_chans, n_times)
        mission-state input data array.
    data_target : (n_trials, n_times)
        mission-state target data array.
    method : str
        SNR, Corr or CCA
    regression : str, optional
        OLS, Ridge, Lasso or ElasticNet. The default is 'OLS'.
    alpha : float, optional
        parameters used in Ridge, Lasso and EN regression. The default is 1.0.
    l1_ratio : float, optional
        parameters used in EN regression. The default is 1.0.
    freq : int/float, optional
        parameters used if method = 'CCA'. The default is None.
    phase : int/float, optional
        parameters used if method = 'CCA'. The default is None.
    sfreq : int/float, optional
        sampling frequency. The default is 1000
    
    Returns
    -------
    model_chans : list
        list of channels which should be used in SRCA
    para_change : list
        list of parameter's alteration
    '''
    # initialize variables
    start = time.perf_counter()   
    loop = 1
    remain_chans = []
    para_change = []
    max_loop = len(chans)

    # begin loop
    print('Stepwise SRCA training...')
    active = True
    while active and len(chans) <= max_loop:
        compare_para = [0 for x in range(len(chans))]  # initialization
        
        # sequence growth 
        if loop == 1:
            # add 1 channel (step forward)
            for step_f in range(len(chans)):
                temp_w = w_in[:, step_f, :]
                temp_data = data_in[:, step_f, :]
                temp_extract = srca(temp_w, w_target, temp_data, data_target, regression)
                del temp_w, temp_data
                if method == 'SNR':
                    temp_para = snr_time(temp_extract)
                elif method == 'Corr':
                    temp_para = pearson_corr(temp_extract)
                elif method == 'CCA':
                    temp_para = template_corr(temp_extract, freq, phase, sfreq)
                # compare the parameter with original one
                compare_para[step_f] = temp_para.mean() - para
            # keep the best choice
            chan_index = compare_para.index(max(compare_para))
            remain_chans.append(chans.pop(chan_index))
            para_change.append(max(compare_para))
            del temp_extract, compare_para, temp_para
            # save new data
            core_w = w_in[:, chan_index, :]
            core_data = data_in[:, chan_index, :]
            # refresh data
            data_in = np.delete(data_in, chan_index, axis=1)
            w_in = np.delete(w_in, chan_index, axis=1)
            del chan_index
            print('Complete ' + str(loop) + 'th loop')
            loop += 1

        # contain stepwise part
        elif loop == 2:
            # add 1 channel (step forward)
            for step_f in range(len(chans)):
                temp_w = np.concatenate((core_w[:, NA, :],
                        w_in[:, step_f, :][:, NA, :]), axis=1)
                temp_data = np.concatenate((core_data[:, NA, :],
                        data_in[:, step_f, :][:, NA, :]), axis=1)
                temp_extract = srca(temp_w, w_target, temp_data, data_target, regression)
                del temp_w, temp_data
                if method == 'SNR':
                    temp_para = snr_time(temp_extract)
                elif method == 'Corr':
                    temp_para = pearson_corr(temp_extract)
                elif method == 'CCA':
                    temp_para = template_corr(temp_extract, freq, phase, sfreq)
                # compare the parameter with original one
                compare_para[step_f] = temp_para.mean() - para
            del temp_extract, temp_para
            # judge condition
            temp_para_change = max(compare_para)
            if temp_para_change < max(para_change):  # start to decay
                print('Training complete!')
                end = time.perf_counter()
                print('Training time: ' + str(end - start) + 's')
                active = False
            else:  # still has improvement, begin stepwise
                # keep the best choice
                para_change.append(max(compare_para))
                chan_index = compare_para.index(max(compare_para))
                remain_chans.append(chans.pop(chan_index))
                del compare_para
                # save new data
                core_w = np.concatenate((core_w[:, NA, :],
                        w_in[:, chan_index, :][:, NA, :]), axis=1)
                core_data = np.concatenate((core_data[:, NA, :],
                        data_in[:, chan_index, :][:, NA, :]), axis=1)
                # refresh data
                data_in = np.delete(data_in, chan_index, axis=1)
                w_in = np.delete(w_in, chan_index, axis=1)
                del chan_index
                # delete 1 channel (step backward, except the just-added one)
                for step_b in range(len(remain_chans) - 1):
                    temp_chans = remain_chans[:]
                    del temp_chans[step_b]
                    temp_w = np.delete(core_w, step_b, axis=1)
                    temp_data = np.delete(core_data, step_b, axis=1)
                    # then add 1 channel immediately (stepwise)
                    compare_wise_para = [0 for x in range(len(chans))]
                    for step_w in range(len(chans)):
                        wise_w = np.concatenate((temp_w[:, NA, :], 
                                w_in[:, step_w, :][:, NA, :]), axis=1)
                        wise_data = np.concatenate((temp_data[:, NA, :],
                                data_in[:, step_w, :][:, NA, :]), axis=1)
                        wise_extract = srca(wise_w, w_target, wise_data, data_target, regression)
                        del wise_w, wise_data
                        if method == 'SNR':
                            wise_para = snr_time(wise_extract)
                        elif method == 'Corr':
                            wise_para = pearson_corr(wise_extract)
                        elif method == 'CCA':
                            wise_para = template_corr(wise_extract, freq, phase, sfreq)
                        # compare the parameter with original one
                        compare_wise_para[step_w] = wise_para.mean() - para
                    # judge condition
                    temp_para_wise_change = max(compare_wise_para)
                    if temp_para_wise_change < max(para_change)
                    # keep the best choice
                    chan_wise_index = compare_wise_para.index(max(compare_wise_para))
                    





            