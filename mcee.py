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
    (2) TRCA: including standard TRCA and extended-TRCA
        for normal signal and SRCA signal
    (3) DCPM: 5 different descrimination indices for normal and SRCA signal
    (4) corr_detect: single channel detection


@ author: Brynhildr
@ email: brynhildrw@gmail.com
version 1.0
"""

# %% Import third part module
import numpy as np
from numpy import linalg as LA
from numpy import corrcoef as CORR
from numpy import newaxis as NA
from numpy import (sin, cos)

from sklearn import linear_model

import copy
import time
from math import pi

# %% Basic operating function
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
        time length / s.
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
    wave = sin(2*pi*freq*time_point + pi*phase)
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

# zero mean normalization (if necessary)
def zero_mean(data, axis):
    '''
    
    Zero mean normalization
    Parameters
    ----------
    data : ndarray
        input data array
    axis : int/None
        if None, data is a sequence.

    Returns
    -------
    data : ndarray
        data after z-scored

    '''
    if axis == None:
        data -= data.mean()
    else:
        data -= data.mean(axis=axis, keepdims=True)
    return data

def corr_coef(X, y):
    """

    Parameters
    ----------
    X : (..., n_points)
        input data array. (could be sequence)
    y : (1, n_points)
        input data vector/sequence

    Returns
    -------
    corrcoef : float
        Pearson's Correlation Coefficient(mean of sequence or single number).
    """
    cov_yx = y @ X.T
    # Note: 'int' object has no attribute 'ndim', but the dimension of 'float' object is 0
    if cov_yx.ndim == 0:  
        var_xx = np.sqrt(X @ X.T)
    # try:
    #     dim = cov_yx.ndim  
    # except AttributeError:
    #     var_xx = np.sqrt(X @ X.T)
    else:
        var_xx = np.sqrt(np.diagonal(X @ X.T)) 
    var_yy = np.sqrt(float(y @ y.T))
    corrcoef = cov_yx / (var_xx*var_yy)

    return corrcoef.mean()

# %% Stepwise SRCA
def SRCA_train(chans, mpara, w, w_target, signal_data, data_target, method='SNR',
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
    mpara : float
        the mean of original signal's parameters in time domain.
    w : (n_trials, n_chans, n_times)
        background part input data array.
    w_target : (n_trials, n_times)
        background part target data array.
    signal_data : (n_trials, n_chans, n_times)
        signal part input data array.
    data_target : (n_trials, n_times)
        signal part target data array.
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
    print('Stepwise SRCA training...')
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
            temp_extract = srca(temp_w, w_target, temp_data, data_target, regression)
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


# %% Canonical Correlation Analysis
def sin_model(base_freq, n_bands, time, phase=0, sfreq=1000):
    """

    Parameters
    ----------
    base_freq : int/float
        frequency / Hz
    n_bands : int
    time : float
        time length / s
    phase : float, optional
        0-2. The default is 0.
    sfreq : float/int, optional
        sampling frequency. The default is 1000.

    Returns
    -------
    model : ndarray, (2*n_bands, time*sfreq)
        sine/cosine wave model.
    """
    n_point = int(time*sfreq)
    model = np.zeros((int(2*n_bands), n_point))
    time_point = np.linspace(0, (n_point-1)/sfreq, n_point)
    
    for i in range(n_bands):
        model[i*2, :] = sin(2*pi*(i+1)*base_freq*time_point + pi*phase)
        model[i*2+1, :] = cos(2*pi*(i+1)*base_freq*time_point + pi*phase)
    
    return model

def CCA_compute(data, model, mode='data'):
    """

    Parameters
    ----------
    data : ndarray, (n_chans, n_times)
    model : ndarray, (2*n_bands, n_times)
    mode : str
        'data' or 'model'. The default is 'data'.

    Returns
    -------
    w : (n_chans or n_bands,)
        spatial filters for eeg data (or model)

    """
    if mode == 'data':
        Z = data.T
        projection = model.T @ LA.inv(model @ model.T) @ model
    elif mode == 'model':
        Z = model.T
        projection = data.T @ LA.inv(data @ data.T) @ data

    # unified framework: type I
    matrix_A = Z.T @ projection @ projection.T @ Z
    matrix_B = Z.T @ Z

    # solve generalized eigenvalue problem
    e_va, e_vec = LA.eig(LA.inv(matrix_A) @ matrix_B)
    w_index = np.argmax(e_va)
    w = e_vec[:, w_index].T

    return w

def sCCA(data, base_freq, n_bands, sfreq=1000):
    """

    Parameters
    ----------
    data : ndarray, (n_events, n_trials, n_chans, n_points)
        test dataset
    base_sfreq : list of float/int
    n_bands : int

    Returns
    -------
    accuracy : float, 0-1
    """
    n_events = data.shape[0]
    n_tests = data.shape[1]
    n_points = data.shape[-1]
    signal_time = n_points/sfreq
    r = np.zeros((n_events, n_tests, n_events))
    filtered_data = np.zeros((n_events, n_events, n_tests, data.shape[2], n_points))

    for nete in range(n_events):
        for nte in range(n_tests):
            # preparation
            test = data[nete, nte, ...]
            for netr in range(n_events):
                model = sin_model(base_freq[netr], n_bands, signal_time)
                # compute spatial filters 
                w_X = CCA_compute(test, model, mode='data')
                w_Y = CCA_compute(test, model, mode='model')

                # dimensionality reduction
                trans_X, trans_Y = w_X @ test, w_Y @ model
                filtered_data[nete, netr, nte, ...] = trans_X

                # compute correlation
                r[nete, nte, netr] = corr_coef(trans_X, trans_Y)
    
    # compute accuracy
    accuracy = 0
    for nete in range(n_events):
        for nte in range(n_tests):
            if np.argmax(r[nete, nte, :]) == nete:
                accuracy += 1
    accuracy /= (n_events*n_tests)

    return accuracy, filtered_data

def itCCA(train_data, test_data):
    """

    Parameters
    ----------
    train_data : ndarray, (n_events, n_trains, n_chans, n_points)
        training dataset
    test_data : ndarray, (n_events, n_tests, n_chans, n_points)
        test dataset
    
    Returns
    -------
    accuracy : float, 0-1
    """
    n_events = train_data.shape[0]
    n_tests = test_data.shape[1]
    template = train_data.mean(axis=1)  # (n_events, n_chans, n_points)
    r = np.zeros((n_events, n_tests, n_events))
    filtered_data = np.zeros((n_events, n_events, n_tests, test_data.shape[2], test_data.shape[-1]))

    for nete in range(n_events):
        for nte in range(n_tests):
            # preparation
            test = test_data[nete, nte, ...]
            for netr in range(n_events):
                model = template[netr, ...]
                # compute spatial filters
                w_X = CCA_compute(test, model, mode='data')
                w_Y = CCA_compute(test, model, mode='model')
                
                # dimensionality reduction
                trans_X, trans_Y = w_X @ test, w_Y @ model
                filtered_data[nete, netr, nte, ...] = trans_X

                # compute correlation
                r[nete, nte, netr] = corr_coef(trans_X, trans_Y)
    
    # compute accuracy
    accuracy = 0
    for nete in range(n_events):
        for nte in range(n_tests):
            if np.argmax(r[nete, nte, :]) == nete:
                accuracy += 1
    accuracy /= (n_events*n_tests)

    return accuracy, filtered_data


# %% Target identification: TRCA method (series)
# pre-functions
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
    # basic information
    n_events = data.shape[0]
    n_trials= data.shape[1]
    n_chans = data.shape[2]
    n_times = data.shape[-1]

    # compute spatial filter W 
    w = np.zeros((n_events, n_chans))
    for ne in range(n_events):
        # matrix Q: inter-channel covariance
        q = np.zeros((n_chans, n_chans))
        temp_Q = data[ne, ...].swapaxes(0,1).reshape((n_chans,-1), order='C')
        q = (temp_Q@temp_Q.T) / (n_trials*n_times)
        # matrix S: inter-channels' inter-trial covariance
        s = np.zeros_like(q)
        for nt_i in range(n_trials):
            for nt_j in range(n_trials):
                if nt_i != nt_j:
                    data_i = data[ne, nt_i, ...]
                    data_j = data[ne, nt_j, ...]
                    s += (data_i@data_j.T) / n_times
        # generalized eigenvalue problem
        e_va, e_vec = LA.eig(LA.inv(q)@s)
        w_index = np.argmax(e_va)
        #w[ne, :] = e_vec[:, e_va.argsort()[::-1]][:,0].T
        w[ne, :] = e_vec[:, w_index].T
        #e_value = np.array(sorted(e_va, reverse=True))
    return w

def pearson_corr2(data_A, data_B):
    """

    Parameters
    ----------
    data_A : ndarray, (n_chans, n_times)
    data_B : ndarray, (n_chans, n_times)

    Returns
    -------
    corr2 : float
        2-D correlation coefficient.
    """
    mean_A = data_A.mean()
    mean_B = data_B.mean()
    numerator = np.sum((data_A-mean_A) * (data_B-mean_B))
    denominator_A = np.sum((data_A-mean_A)**2)
    denominator_B = np.sum((data_B-mean_B)**2)
    corr2 = numerator / np.sqrt(denominator_A*denominator_B)

    return corr2

# For origin data
def TRCA(train_data, test_data):
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
    accuracy : float, 0-1

    '''
    # basic parameters
    n_events, n_tests = test_data.shape[0], test_data.shape[1]
    template = train_data.mean(axis=1)  # template data: (n_events, n_chans, n_times)
    w = TRCA_compute(train_data)        # spatial filter W: (n_events, n_chans)
    filtered_data = np.zeros((n_events, n_events, n_tests, test_data.shape[-2], test_data.shape[-1]))

    # target identification
    r = np.zeros((n_events, n_tests, n_events))
    for netr in range(n_events):  # n_events in training dataset
        for nte in range(n_tests):
            for nete in range(n_events):  # n_events in test dataset
                temp_test = w[netr, :] @ test_data[nete, nte, ...]
                temp_template = w[netr, :] @ template[netr, ...]
                r[netr, nte, nete] = corr_coef(temp_test, temp_template)
                filtered_data[nete, netr, nte, ...] = temp_test
    # compute accuracy
    accuracy = []
    for nete in range(n_events):  # n_events in training dataset
        for nte in range(n_tests):
            if np.argmax(r[:, nte, nete]) == nete:
                accuracy.append(1)
    accuracy = np.sum(accuracy) / (n_events*n_tests)

    return accuracy, filtered_data

def eTRCA(train_data, test_data):
    '''
    ensemble-TRCA for original signal

    Parameters
    ----------
        train_data: (n_events, n_trials, n_chans, n_times) | training dataset
        test_data: (n_events, n_trials, n_chans, n_times) | test dataset
    
    Returns
    -------
        accuracy: int | the number of correct identification
    '''
    # basic parameters
    n_events = train_data.shape[0]
    n_tests = test_data.shape[1]

    template = train_data.mean(axis=1)  # template data: (n_events, n_chans, n_times)
    w = TRCA_compute(train_data)        # spatial filter W: (n_events, n_chans)

    # target identification
    r = np.zeros((n_events, n_tests, n_events))
    for netr in range(n_events):  # n_events in training dataset
        for nte in range(n_tests):
            for nete in range(n_events):  # n_events in test dataset
                temp_test = w @ test_data[nete, nte, ...]
                temp_template = w @ template[netr, ...]
                r[netr, nte, nete] = pearson_corr2(temp_test, temp_template)

    # compute accuracy
    accuracy = []
    for nete in range(n_events):  # n_events in training dataset
        for nte in range(n_tests):
            if np.argmax(r[:, nte, nete]) == nete:
                accuracy.append(1)
    accuracy = np.sum(accuracy) / (n_events*n_tests)

    return accuracy

def TRCA_R():
    '''
    reference-TRCA without filter banks for original signal

    Parameters
    ----------
        train_data: (n_events, n_trials, n_chans, n_times) | training dataset
        test_data: (n_events, n_trials, n_chans, n_times) | test dataset
    
    Returns
    -------
        accuracy: int | the number of correct identification
    '''
    

    pass

def eTRCA_R():
    pass

def split_TRCA(stepwidth, train_data, test_data, mode='total'):
    """

    Parameters
    ----------
    train_data : (n_events, n_trials, n_chans, n_times)
        training dataset.
    test_data : (n_events, n_trials, n_chans, n_times)
        test dataset.
    stepwidth : int
        step size of division operation based on the period of signals
    mode : str
        'partial' or 'total', the default is 'total'

    Returns
    -------
    accuracy : float, 0-1

    """
    # basic parameters
    n_events = test_data.shape[0]
    n_tests = test_data.shape[1]
    n_chans = test_data.shape[2]
    n_times = test_data.shape[-1]
    template = train_data.mean(axis=1)  # (n_events, n_chans, n_times)
    seg_num = int(n_times/stepwidth)
    
    # split data
    seg_test_data = np.zeros((1, n_events, n_tests, n_chans, stepwidth))
    seg_template = np.zeros((1, n_events, n_chans, stepwidth))
    for i in range(seg_num):
        # seg_test_data: (seg_num, n_events, n_trials, n_chans, stepwidth)
        seg_test_data = np.concatenate((seg_test_data,
            test_data[NA, ..., i*stepwidth:(i+1)*stepwidth]), axis=0)
        # seg_template: (seg_num, n_events, n_chans, stepwidth)
        seg_template = np.concatenate((seg_template,
            template[NA, ..., i*stepwidth:(i+1)*stepwidth]), axis=0)
    seg_test_data = np.delete(seg_test_data, 0, axis=0)
    seg_template = np.delete(seg_template, 0, axis=0)

    if mode == 'partial':  # only divide the test dataset
        # compute spatial filter w
        w = TRCA_compute(train_data)

        # split target identification
        rou = np.zeros((seg_num, n_events, n_tests, n_events))
        for seg in range(seg_num):
            for netr in range(n_events):  # n_events in training dataset
                for nte in range(n_tests):
                    for nete in range(n_events):  # n_events in test dataset
                        temp_test = w[netr, :] @ seg_test_data[seg, nete, nte, ...]
                        temp_template = w[netr, :] @ seg_template[seg, netr, ...]
                        rou[seg, netr, nte, nete] = corr_coef(temp_test, temp_template)

    elif mode == 'total':  # divide both the training and test dataset
        # compute spatial filter w
        seg_w = np.zeros((1, n_events, n_chans))
        for i in range(seg_num):
            temp_w = TRCA_compute(train_data[..., i*stepwidth:(i+1)*stepwidth])
            seg_w = np.concatenate((seg_w, temp_w[NA, ...]), axis=0)
        seg_w = np.delete(seg_w, 0, axis=0)

        # split target identification
        rou = np.zeros((seg_num, n_events, n_tests, n_events))
        for seg in range(seg_num):
            for netr in range(n_events):  # n_events in training dataset
                for nte in range(n_tests):
                    for nete in range(n_events):  # n_events in test dataset
                        temp_test = seg_w[seg, netr, :] @ seg_test_data[seg, nete, nte, ...]
                        temp_template = seg_w[seg, netr, :] @ seg_template[seg, netr, ...]
                        rou[seg, netr, nte, nete] = corr_coef(temp_test, temp_template)
        
    # compute accuracy
    accuracy = []
    r = np.sum(rou, axis=0)
    for nete in range(n_events):  # n_events in training dataset
        for nte in range(n_tests):
            if np.argmax(r[:, nte, nete]) == nete:
                accuracy.append(1)
    accuracy = np.sum(accuracy) / (n_events*n_tests)

    return rou, accuracy

def split_eTRCA(stepwidth, train_data, test_data, mode='total'):

    # basic parameters
    n_events = test_data.shape[0]
    n_tests = test_data.shape[1]
    n_chans = test_data.shape[2]
    n_times = test_data.shape[-1]
    template = train_data.mean(axis=1)  # (n_events, n_chans, n_times)
    seg_num = int(n_times/stepwidth)
    
    # split data
    seg_test_data = np.zeros((1, n_events, n_tests, n_chans, stepwidth))
    seg_template = np.zeros((1, n_events, n_chans, stepwidth))
    for i in range(seg_num):
        # seg_test_data: (seg_num, n_events, n_trials, n_chans, stepwidth)
        seg_test_data = np.concatenate((seg_test_data,
                        test_data[NA, ..., i*stepwidth:(i+1)*stepwidth]), axis=0)
        # seg_template: (seg_num, n_events, n_chans, stepwidth)
        seg_template = np.concatenate((seg_template,
                        template[NA, ..., i*stepwidth:(i+1)*stepwidth]), axis=0)
    seg_test_data = np.delete(seg_test_data, 0, axis=0)
    seg_template = np.delete(seg_template, 0, axis=0)

    if mode == 'partial':  # only divide the test dataset
        # compute spatial filter w
        w = TRCA_compute(train_data)

        # split target identification
        rou = np.zeros((seg_num, n_events, n_tests, n_events))
        for seg in range(seg_num):
            for netr in range(n_events):  # n_events in training dataset
                for nte in range(n_tests):
                    for nete in range(n_events):  # n_events in test dataset
                        temp_test = w @ seg_test_data[seg, nete, nte, ...]
                        temp_template = w @ seg_template[seg, netr, ...]
                        rou[seg, netr, nte, nete] = pearson_corr2(temp_test, temp_template)

    elif mode == 'total':  # divide both the training and test dataset
        # compute spatial filter w
        seg_w = np.zeros((1, n_events, n_chans))
        for i in range(seg_num):
            temp_w = TRCA_compute(train_data[..., i*stepwidth:(i+1)*stepwidth])
            seg_w = np.concatenate((seg_w, temp_w[NA, ...]), axis=0)
        seg_w = np.delete(seg_w, 0, axis=0)

        # split target identification
        rou = np.zeros((seg_num, n_events, n_tests, n_events))
        for seg in range(seg_num):
            for netr in range(n_events):  # n_events in training dataset
                for nte in range(n_tests):
                    for nete in range(n_events):  # n_events in test dataset
                        temp_test = seg_w[seg, ...] @ seg_test_data[seg, nete, nte, ...]
                        temp_template = seg_w[seg, ...] @ seg_template[seg, netr, ...]
                        rou[seg, netr, nte, nete] = pearson_corr2(temp_test, temp_template)

    # compute accuracy
    accuracy = []
    r = np.sum(rou, axis=0)
    for nete in range(n_events):  # n_events in training dataset
        for nte in range(n_tests):
            if np.argmax(r[:, nte, nete]) == nete:
                accuracy.append(1)
    accuracy = np.sum(accuracy) / (n_events*n_tests)

    return rou, accuracy

def split_TRCA_R():
    pass

def split_eTRCA_R():
    pass


# For SRCA data
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
    # basic parameters
    n_events = train_data.shape[0]
    n_trains = train_data.shape[1]
    n_chans = len(tar_chans)
    n_times = train_data.shape[-1] - sp

    # config correct srca process on training dataset
    model_sig = np.zeros((n_events, n_trains, n_chans, n_times))
    for ne in range(n_events):
        temp_model = model_chans[ne::n_events]  # length: len(tar_chans)
        model_sig[ne, ...] = apply_SRCA(train_data[ne, ...], tar_chans, temp_model, chans,
                                        regression, sp)
    del ne, temp_model

    # apply different srca models on one trial's data
    n_tests = test_data.shape[1]
    target_sig = np.zeros((n_events, n_events, n_tests, n_chans, n_times))
    for nes in range(n_events):      # n_events in SRCA model
        temp_model = model_chans[nes::n_events]
        for ner in range(n_events):  # n_events in real data
            target_sig[nes, ner, ...] = apply_SRCA(test_data[ner, ...], tar_chans, temp_model,
                                                        chans, regression, sp)
    del nes, ner

    template = model_sig.mean(axis=1)  # template data: (n_events, n_chans, n_times)
    w = TRCA_compute(model_sig)        # Spatial filter W: (n_events, n_chans)

    # target identification
    r = np.zeros((n_events, n_tests, n_events))  # (n_events srca, n_tests, n_events test)
    for nes in range(n_events):                  # n_events in srca model
        for nte in range(n_tests):               # n_tests
            for ner in range(n_events):          # n_events in real test dataset
                temp_test = w[nes, :] @ target_sig[nes, ner, nte,...]
                temp_template = w[nes, :] @ template[nes, ...]
                r[nes, nte, ner] = corr_coef(temp_test, temp_template)

    # compute accuracy
    accuracy = []
    for ner in range(n_events):
        for nte in range(n_tests):
            if np.argmax(r[:, nte, ner]) == ner:
                accuracy.append(1)
    accuracy = np.sum(accuracy) / (n_events*n_tests)

    return accuracy

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
        model_sig[ne, :, :, :] = apply_SRCA(train_data[ne, ...], tar_chans,
        temp_model, chans, regression, sp)
    del ne, temp_model

    # apply different srca models on one trial's data
    n_tests = test_data.shape[1]
    target_sig = np.zeros((n_events, n_events, n_tests, n_chans, n_times))
    for nes in range(n_events):  # n_events in SRCA model
        temp_model = model_chans[nes::n_events]
        for ner in range(n_events):
            target_sig[nes, ner, ...] = apply_SRCA(test_data[ner, ...],
            tar_chans, temp_model, chans, regression, sp)
    del nes, ner

    template = model_sig.mean(axis=1)
    w = TRCA_compute(model_sig)

    # Ensemble target identification
    r = np.zeros((n_events, n_tests, n_events))  # (n_events srca, n_tests, n_events test)
    for nes in range(n_events):                  # n_events in srca model
        for nte in range(n_tests):               # n_tests
            for ner in range(n_events):          # n_events in test dataset
                temp_test = np.dot(w, target_sig[nes, ner, nte, ...])
                temp_template = np.dot(w, template[nes, ...])
                r[nes, nte, ner] = pearson_corr2(temp_test, temp_template)
    del nes, nte, ner

    # Compute accuracy
    accuracy = []
    for ner in range(n_events):
        for nte in range(n_tests):
            if np.argmax(r[:, nte, ner]) == ner:
                accuracy.append(1)
    accuracy = np.sum(accuracy) / (n_events*n_tests)

    return accuracy

def SRCA_TRCA_R():
    pass

def SRCA_eTRCA_R():
    pass

def split_SRCA_TRCA(stepwidth, train_data, test_data, tar_chans, model_chans, chans,
                    regression='OLS', alpha=1.0, l1_ratio=1.0, sp=1140, mode='total'):
    
    # basic parameters
    n_events = test_data.shape[0]
    n_trains = train_data.shape[1]
    n_tests = test_data.shape[1]
    n_chans = len(tar_chans)
    n_times = train_data.shape[-1] - sp
    seg_num = int(n_times/stepwidth)

    # config correct srca process on training dataset
    model_sig = np.zeros((n_events, n_trains, n_chans, n_times))
    for ne in range(n_events):
        temp_model = model_chans[ne::n_events]  # length: len(tar_chans)
        model_sig[ne, ...] = apply_SRCA(train_data[ne, ...], tar_chans, temp_model,
            chans, regression, sp)
    template = model_sig.mean(axis=1)  # template data: (n_events, n_chans, n_times)
    del ne

    # apply different srca models on one trial's data
    n_tests = test_data.shape[1]
    target_sig = np.zeros((n_events, n_events, n_tests, n_chans, n_times))
    for ne_tr in range(n_events):      # n_events in SRCA model
        temp_model = model_chans[ne_tr::n_events]
        for ne_re in range(n_events):  # n_events in real data
            target_sig[ne_tr, ne_re, ...] = apply_SRCA(test_data[ne_re, ...], tar_chans,
                temp_model, chans, regression, sp)
    del ne_tr, ne_re

    # split data
    seg_target_data = np.zeros((1, n_events, n_events, n_tests, n_chans, stepwidth))
    seg_template = np.zeros((1, n_events, n_chans, stepwidth))
    for i in range(seg_num):
        # seg_target_data: (seg_num, n_events, n_events, n_trials, n_chans, stepwidth)
        seg_target_data = np.concatenate((seg_target_data,
            target_sig[NA, ..., i*stepwidth:(i+1)*stepwidth]), axis=0)
        # seg_template: (seg_num, n_events, n_chans, stepwidth)
        seg_template = np.concatenate((seg_template,
            template[NA, ..., i*stepwidth:(i+1)*stepwidth]), axis=0)
    seg_target_data = np.delete(seg_target_data, 0, axis=0)
    seg_template = np.delete(seg_template, 0, axis=0)
    
    if mode == 'partial':  # only divide the test dataset
        # compute spatial filter w
        w = TRCA_compute(model_sig)

        # split target identification
        rou = np.zeros((seg_num, n_events, n_tests, n_events))
        for seg in range(seg_num):
            for nes in range(n_events):                  # n_events in srca model
                for nte in range(n_tests):               # n_tests
                    for ner in range(n_events):          # n_events in real test dataset
                        temp_test = w[nes, :] @ seg_target_data[seg, nes, ner, nte, ...]
                        temp_template = w[nes, :] @ seg_template[seg, nes, ...]
                        rou[seg, nes, nte, ner] = corr_coef(temp_test, temp_template)
    
    elif mode == 'total':  # divide both the training and test dataset
        # compute spatial filter w
        seg_w = np.zeros((1, n_events, n_chans))
        for i in range(seg_num):
            temp_w = TRCA_compute(model_sig[..., i*stepwidth:(i+1)*stepwidth])
            seg_w = np.concatenate((seg_w, temp_w[NA, ...]), axis=0)
        seg_w = np.delete(seg_w, 0, axis=0)

        # split target identification
        rou = np.zeros((seg_num, n_events, n_tests, n_events))
        for seg in range(seg_num):
            for nes in range(n_events):                  # n_events in srca model
                for nte in range(n_tests):               # n_tests
                    for ner in range(n_events):          # n_events in real test dataset
                        temp_test = seg_w[seg, nes, :] @ seg_target_data[seg, nes, ner, nte, ...]
                        temp_template = seg_w[seg, nes, :] @ seg_template[seg, nes, ...]
                        rou[seg, nes, nte, ner] = corr_coef(temp_test, temp_template)

    # compute accuracy
    accuracy = []
    r = np.sum(rou, axis=0)
    for ner in range(n_events):
        for nte in range(n_tests):
            if np.argmax(r[:, nte, ner]) == ner:
                accuracy.append(1)
    accuracy = np.sum(accuracy) / (n_events*n_tests)

    return rou, accuracy

def split_SRCA_eTRCA(stepwidth, train_data, test_data, tar_chans, model_chans, chans,
                    regression='OLS', alpha=1.0, l1_ratio=1.0, sp=1140, mode='total'):

    # basic parameters
    n_events = test_data.shape[0]
    n_trains = train_data.shape[1]
    n_tests = test_data.shape[1]
    n_chans = len(tar_chans)
    n_times = train_data.shape[-1] - sp
    seg_num = int(n_times/stepwidth)

    # config correct srca process on training dataset
    model_sig = np.zeros((n_events, n_trains, n_chans, n_times))
    for ne in range(n_events):
        temp_model = model_chans[ne::n_events]  # length: len(tar_chans)
        model_sig[ne, ...] = apply_SRCA(train_data[ne, ...], tar_chans, temp_model,
            chans, regression, sp)
    template = model_sig.mean(axis=1)  # template data: (n_events, n_chans, n_times)
    del ne

    # apply different srca models on one trial's data
    n_tests = test_data.shape[1]
    target_sig = np.zeros((n_events, n_events, n_tests, n_chans, n_times))
    for nes in range(n_events):      # n_events in SRCA model
        temp_model = model_chans[nes::n_events]
        for ner in range(n_events):  # n_events in real data
            target_sig[nes, ner, ...] = apply_SRCA(test_data[ner, ...], tar_chans,
                temp_model, chans, regression, sp)
    del nes, ner

    # split data
    seg_target_data = np.zeros((1, n_events, n_events, n_tests, n_chans, stepwidth))
    seg_template = np.zeros((1, n_events, n_chans, stepwidth))
    for i in range(seg_num):
        # seg_target_data: (seg_num, n_events, n_events, n_trials, n_chans, stepwidth)
        seg_target_data = np.concatenate((seg_target_data,
            target_sig[NA, ..., i*stepwidth:(i+1)*stepwidth]), axis=0)
        # seg_template: (seg_num, n_events, n_chans, stepwidth)
        seg_template = np.concatenate((seg_template,
            template[NA, ..., i*stepwidth:(i+1)*stepwidth]), axis=0)
    seg_target_data = np.delete(seg_target_data, 0, axis=0)
    seg_template = np.delete(seg_template, 0, axis=0)
    
    if mode == 'partial':  # only divide the test dataset
        # compute spatial filter w
        w = TRCA_compute(model_sig)

        # split target identification
        rou = np.zeros((seg_num, n_events, n_tests, n_events))
        for seg in range(seg_num):
            for nes in range(n_events):                  # n_events in srca model
                for nte in range(n_tests):               # n_tests
                    for ner in range(n_events):          # n_events in real test dataset
                        temp_test = w @ seg_target_data[seg, nes, ner, nte, ...]
                        temp_template = w @ seg_template[seg, nes, ...]
                        rou[seg, nes, nte, ner] = pearson_corr2(temp_test, temp_template)
    
    elif mode == 'total':  # divide both the training and test dataset
        # compute spatial filter w
        seg_w = np.zeros((1, n_events, n_chans))
        for i in range(seg_num):
            temp_w = TRCA_compute(model_sig[..., i*stepwidth:(i+1)*stepwidth])
            seg_w = np.concatenate((seg_w, temp_w[NA, ...]), axis=0)
        seg_w = np.delete(seg_w, 0, axis=0)

        # split target identification
        rou = np.zeros((seg_num, n_events, n_tests, n_events))
        for seg in range(seg_num):
            for nes in range(n_events):                  # n_events in srca model
                for nte in range(n_tests):               # n_tests
                    for ner in range(n_events):          # n_events in real test dataset
                        temp_test = seg_w[seg, ...] @ seg_target_data[seg, nes, ner, nte, ...]
                        temp_template = seg_w[seg, ...] @ seg_template[seg, nes, ...]
                        rou[seg, nes, nte, ner] = pearson_corr2(temp_test, temp_template)

    # compute accuracy
    accuracy = []
    r = np.sum(rou, axis=0)
    for ner in range(n_events):
        for nte in range(n_tests):
            if np.argmax(r[:, nte, ner]) == ner:
                accuracy.append(1)
    accuracy = np.sum(accuracy) / (n_events*n_tests)

    return rou, accuracy

def split_SRCA_TRCA_R():
    pass

def split_SRCA_eTRCA_R():
    pass

def cvep_TI(code_num, code_length, gap_length, train_data, test_data):
    """
    """
    # if test_data.shape[-1] != (code_num-1)*(code_length+gap_length) + code_length:
    #     raise Exception('Please check the length of input data!')
    # # split data according to different codes
    # code_data = 1

    pass

# %% Target identification: DCPM
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
    x1 = np.swapaxes(train_data[0, ...], 0, 2)  # (n_times, n_chans, n_trials)
    x1 = np.swapaxes(x1, 0, 1)                     # (n_chans, n_times, n_trials)
    x2 = np.swapaxes(train_data[1, ...], 0, 2)  # (n_times, n_chans, n_trials)
    x2 = np.swapaxes(x2, 0, 1)                     # (n_chans, n_times, n_trials)
    pattern1 = np.mat(np.mean(train_data[0, ...], axis=0))  # (n_chans, n_times)
    pattern2 = np.mat(np.mean(train_data[1, ...], axis=0))  # (n_chans, n_times)
    # covariance matrices
    sigma11 = pattern1 * pattern1.T  # (n_chans, n_chans)
    sigma12 = pattern1 * pattern2.T  # (n_chans, n_chans)
    sigma21 = pattern2 * pattern1.T  # (n_chans, n_chans)
    sigma22 = pattern2 * pattern2.T  # (n_chans, n_chans)
    # variance matrices
    var1, var2 = np.zeros_like(sigma11), np.zeros_like(sigma22)
    for i in range(n_trains):
        var1 += (x1[...,i] - pattern1) * (x1[...,i] - pattern1).T
        var2 += (x2[...,i] - pattern2) * (x2[...,i] - pattern2).T
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
    acc = np.sum(acc)/(n_tests*n_events)
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


# %% Correlation detect for single-channel data
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
