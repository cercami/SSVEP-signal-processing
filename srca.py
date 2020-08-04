# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 12:07:56 2019

1. Basic operating functions:
    (1) Multi_linear regression: mlr
    (2) SNR in time domain: snr_time
    (3) Fisher score
    
2. Three kinds of recursive algorithm to choose channels for optimization
    (1) Backward: backward_SRCA
    (2) Forward: forward_SRCA
    (3) Stepwise: stepwise_SRCA

3. Target identification
    (1) TRCA & fbTRCA
    (2) Correlation dectection
    (3) Ensembled-TRCA
    (4) extended-CCA

updating...

@author: Brynhildr
"""

#%% Import third part module
import numpy as np
from sklearn import linear_model
import copy
import time

#%% Basic operating function
# Extract optimized signals from linear model
def SRCA_lm_extract(model_input, model_target, data_input, data_target,
                 regression='OLS', alpha=0.5, l1_ratio=0.5):
    '''
    Use different linear models to achieve SRCA extraction
    Parameters:
        model_input: (n_trials, n_chans, n_times)
        model_target: (n_trials, n_times)
        data_input: (n_trials, n_chans, n_times)
        data_target: (n_trials, n_times)
        regression | linear models (str):
            OLS : Ordinary Least Squares
            Ridge : Ridge Regression
            Lasso : Lasso Regression
            EN: ElasticNet Regression
            updating...
        alpha: float | default 0.5, parameters used in Ridge, Lasso and EN regression
        l1_ratio: float | default 0.5, parameters used in EN regression
    Returns:
        extract: (n_trials, n_times) | filtered data
    '''
    # basic information
    n_times = data_input.shape[-1]
    n_trials = model_input.shape[0]
    estimate = np.zeros((n_trials, n_times))  # initialization
    for i in range(n_trials):
        if model_input.ndim == 3:             # (n_trials, n_chans, n_times)
            x = np.mat(model_input[i,:,:]).T  # (n_times, n_chans)
            z = data_input[i,:,:]             # (n_chans, n_times)
        elif model_input.ndim == 2:           # (n_trials, n_times) for single channel
            x = np.mat(model_input[i,:]).T    # (n_times, 1)
            z = data_input[i,:]               # (1, n_times)
        y = np.mat(model_target[i,:]).T       # (n_times,1)
        if regression == 'OLS':               # ordinaru least squares
            L = linear_model.LinearRegression().fit(x,y)
        elif regression == 'Ridge':           # Ridge Regression
            L = linear_model.Ridge(alpha=alpha).fit(x,y)
        elif regression == 'Lasso':           # Lasso Regression
            L = linear_model.Lasso(alpha=alpha).fit(x,y)
        elif regression == 'EN':              # ElasticNet Regression
            L = linear_model.ElasticNet(alpha=1, l1_ratio=l1_ratio).fit(x,y)
        RI = L.intercept_
        RC = L.coef_
        if model_input.ndim == 3:
            estimate[i,:] = np.mat(RC) * np.mat(z) + RI
        elif model_input.ndim == 2:
            estimate[i,:] = RC * z + RI
    extract = data_target - estimate          # extract optimized data
    return extract

# Compute time-domain SNR
def snr_time(data):
    '''
    Compute the mean of SSVEP's SNR in time domain
    Parameters:
        data: (n_trials, n_times)
    Returns:
        snr: float | the mean of SNR sequence
    '''
    snr = np.zeros((data.shape[1]))             # (n_times)
    ex = np.mat(np.mean(data, axis=0))          # one-channel data: (1, n_times)
    temp = np.mat(np.ones((1, data.shape[0])))  # (1, n_trials)
    minus = (temp.T * ex).A                     # (n_trials, n_times)
    ex = (ex.A) ** 2                            # signal's power
    var = np.mean((data - minus)**2, axis=0)    # noise's power (avg)
    snr = ex/var
    return snr

# Compute time-domain Pearson Correlation Coefficient
def pearson_corr(data):
    '''
    Compute the mean of Pearson correlation coefficient in time domain
    Parameters:
        data: (n_trials, n_times)
    Returns:
        mcorr float | the mean of corr sequence
    '''
    template = np.mean(data, axis=0)
    n_trials = data.shape[0]
    corr = np.zeros((n_trials))
    for i in range(n_trials):
        corr[i] = np.sum(np.tril(np.corrcoef(template, data[i,:]),-1))
    return corr

# Compute Fisher Score
def fisher_score(data):
    '''
    Compute the mean of Fisher Score in time domain
    Parameters:
        data: (n_events, n_trails, n_times)
    Returns:
        fs: float | the mean of fisher score
    '''
    # initialization
    sampleNum = data.shape[1]    # n_trials
    featureNum = data.shape[-1]  # n_times
    groupNum = data.shape[0]     # n_events
    miu = np.mean(data, axis=1)  # (n_events, n_times)
    all_miu = np.mean(miu, axis=0)
    # inter-class divergence
    ite_d = np.sum(sampleNum * (miu - all_miu)**2, axis=0)
    # intra-class divergence
    itr_d= np.zeros((groupNum, featureNum))
    for i in range(groupNum):
        for j in range(featureNum):
            itr_d[i,j] = np.sum((data[i,:,j] - miu[i,j])**2)
    # fisher score
    fs = (ite_d) / np.sum(sampleNum * itr_d, axis=0)
    return fs

# Apply SRCA model
def applySRCA(data, tar_chans, model_chans, regression, sp=1140):
    '''
    Apply SRCA model in test dataset
    Parameters:
        data: (n_trials, n_chans, n_times) | test dataset
        tar_chans: str list | names of target channels
        model_chans: str list | names of SRCA channels for each target channel
        regression: str | OLS, Ridge, Lasso or ElasticNet
        sp: int | start point of mission state (default 1140)
    Returns:
        f_data: (n_trials, n_chans, n_times) | filtered data
    '''
    
    pass

#%% Target identification: TRCA method
def fbtrca(tr_fb_data, te_fb_data):
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
    # from now on, never use w as loop mark for we have variable named w
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
    acc = []
    # compute accuracy
    for x in range(rou.shape[0]):  # ideal classification
        for y in range(rou.shape[1]):
            if np.max(np.where(rou[x,y,:] == np.max(rou[x,y,:]))) == x:  # correct
                acc.append(1)  
    return acc

def pure_trca(train_data, test_data):
    '''
    TRCA without filter banks
    Parameters:
        train_data: (n_events, n_trials, n_chans, n_times) | training dataset
        test_data: (n_events, n_trials, n_chans, n_times) | test dataset
    Returns:
        accuracy: int | the number of correct identifications
    '''
    # template data: (n_events, n_chans, n_times) | basic element: (n_chans, n_times)
    template = np.mean(train_data, axis=1) 
    # basic parameters
    n_events = train_data.shape[0]
    n_chans = train_data.shape[2]
    n_times = train_data.shape[3]   
    # Matrix Q: inter-channel covariance
    q = np.zeros((n_events, n_chans, n_chans))
    for x in range(n_events):
        temp = np.zeros((n_chans, int(train_data.shape[1]*n_times)))
        for y in range(n_chans):
            # concatenated matrix of all trials in training dataset
            temp[y,:] = train_data[x,:,y,:].flatten()
        q[x,:,:] = np.cov(temp) 
    # Matrix S: inter-channels' inter-trial covariance
    s = np.zeros((n_events, n_chans, n_chans))
    for v in range(n_events):  # v for events
        for w in range(n_chans):  # w for channels (j1)
            for x in range(n_chans):  # x for channels (j2)
                cov = []
                for y in range(train_data.shape[1]):  # y for trials (h1)
                    temp = np.zeros((2, n_times))
                    temp[0,:] = train_data[v,y,w,:]
                    for z in range(train_data.shape[1]):  # z for trials (h2)
                        if z != y:  # h1 != h2
                            temp[1,:] = train_data[v,z,x,:]
                            cov.append(np.sum(np.tril(np.cov(temp),-1)))
                        else:
                            continue
                s[v,w,x] = np.sum(cov) 
    # Spatial filter W
    w = np.zeros((n_events, n_chans))
    for z in range(n_events):
        # Square Q^-1 * S
        qs = np.mat(q[z,:,:]).I * np.mat(s[z,:,:])
        # Eigenvalues & eigenvectors
        e_value, e_vector = np.linalg.eig(qs)
        # choose the eigenvector refering to the largest eigenvalue
        w_index = np.max(np.where(e_value == np.max(e_value)))
        w[z,:] = e_vector[:,w_index].T 
    # Test dataset operating
    r = np.zeros((n_events, test_data.shape[1], n_events))
    for x in range(n_events):  # n_events in test dataset
        for y in range(test_data.shape[1]):
            for z in range(n_events):
                temp_test = np.mat(test_data[x,y,:,:]).T * np.mat(w[z,:]).T
                temp_template = np.mat(template[z,:,:]).T * np.mat(w[z,:]).T
                r[x,y,z] = np.sum(np.tril(np.corrcoef(temp_test.T, temp_template.T),-1))  
    # Compute accuracy
    acc = []
    for x in range(r.shape[0]):  # ideal classification
        for y in range(r.shape[1]):
            if np.max(np.where(r[x,y,:] == np.max(r[x,y,:]))) == x:  # correct
                acc.append(1)               
    return acc
    
# Correlation detect for single-channel data
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
    acc = np.sum(acc)
    for j in range(n_events):
        for k in range(n_events):
            mr[j,k] = np.mean(rou[j,:,k])       
    return acc, mr, rou

#%% Spatial Related Component Analysis: SRCA
class SRCA:
    # initialise the spatial related component analysis process
    def __init__(self, data=None, target=None, chan_info=None, model_num=None,
                 model_length=None, signal_num=None, signal_length=None, start_point=None):
        '''
        target | list: name of target channels
        chan_info | list: names of all channels
        data | (n_events, n_trials, n_chans, n_times)
        '''
        # basic infomation
        self.target = target
        self.n_events = data.shape[0]        
        srca_chans = copy.deepcopy(chan_info)
        del srca_chans[chan_info.index(self.target)]
        self.train_chans = srca_chans
        del srca_chans
        
        # initialise model data (train dataset)
        tr_w = data[:, :model_num, :, (start_point-model_length):start_point]
        tr_w_temp = copy.deepcopy(tr_w)
        # model input: (n_events, n_trials, n_chans, n_times)
        self.tr_w_i = np.delete(tr_w_temp, chan_info.index(target), axis=2)
        # model output: (n_events, n_trials, n_times)
        self.tr_w_o = tr_w[:, :, chan_info.index(target), :]
        del tr_w, tr_w_temp      
        
        # initialise mission data (train dataset)
        tr_sig = data[:, :model_num, :, start_point:(start_point+signal_length)]
        tr_sig_temp = copy.deepcopy(tr_sig)
        # model input: (n_events, n_trials, n_chans, n_times)
        self.tr_sig_i = np.delete(tr_sig_temp, chan_info.index(target), axis=2)
        # model output: (n_events, n_trials, n_times)
        self.tr_sig_o = tr_sig[:, chan_info.index(target), :]
        del tr_sig, tr_sig_temp       
        
        # initialise model data (test dataset)
        te_w = data[:, -signal_num:, :, (start_point-model_length):start_point]
        te_w_temp = copy.deepcopy(te_w)
        # model input: (n_events, n_trials, n_chans, n_times)
        self.te_w_i = np.delete(te_w_temp, chan_info.index(target), axis=2)
        # model output: (n_events, n_trials, n_times)
        self.te_w_o = te_w[:, :, chan_info.index(target), :]
        del te_w, te_w_temp
        
        # initialise mission data (test dataset) 
        te_sig = data[:, -signal_num:, :, start_point:(start_point+signal_length)]
        te_sig_temp = copy.deepcopy(te_sig)
        # model input: (n_events, n_trials, n_chans, n_times)
        self.te_sig_i = np.delete(te_sig_temp, chan_info.index(target), axis=2)
        # model output: (n_events, n_trials, n_times)
        self.te_sig_o = te_sig[:, :, chan_info.index(target), :]
        del te_sig, te_sig_temp
        pass
    
    # config unique requirement
    def prepare(self, evaluation='SNR'):
        # compute parameters
        if evaluation == 'SNR':     # Signal-Noise Ratio
            self.tr_para = np.zeros((self.n_events))
            self.ori_te_para = np.zeros_like(self.tr_para)
            for i in range(self.n_events):
                self.tr_para[i] = np.mean(snr_time(self.tr_sig_o[i, :, :]))
                self.ori_te_para[i] = np.mean(snr_time(self.te_sig_o[i, :, :]))
        elif evaluation == 'Corr':  # Pearson Correlation Coefficient
            self.tr_para = np.zeros((self.n_events))
            self.ori_te_para = np.zeros_like(self.tr_para)
            for i in range(self.n_events):
                self.tr_para[i] = np.mean(pearson_corr(self.tr_sig_o[i, :, :]))
                self.ori_te_para[i] = np.mean(pearson_corr(self.te_sig_o[i, :, :]))
        elif evaluation == 'FS':    # Fisher Score
            self.tr_para = np.mean(self.tr_sig_o)
            self.ori_te_para = np.mean(self.te_sig_o)
        self.remain_chans = []
        self.para_change = []
        pass
    
    # stepwise preparation
    def move_forward(self, evaluation='SNR', alpha=0.5, l1_ratio=0.5,
                     regression='OLS', first=False, loop=None):
        n_tc = len(self.train_chans)
        compare_para = np.zeros((n_tc))
        for i in range(n_tc):  # add 1 channel respectively
            # data preparation
            if first:  # avoid reshape error
                temp_w = self.tr_w_i[:, i, :]
                temp_sig = self.tr_sig_i[:, i, :]
            # based on data from last turn
            elif first==False and self.train_chans[i] not in self.remain_chans :
                temp_w = np.zeros((self.tr_w_i.shape[0], loop, self.tr_w_i.shape[-1]))
                temp_w[:, :(loop-1), :] = self.core_w
                temp_w[:, -1, :] = self.tr_w_i[:, i, :]
                
                temp_sig = np.zeros((self.tr_sig_i.shape[0], loop, self.tr_sig_i.shape[-1]))
                temp_sig[:, :(loop-1), :] = self.core_sig
                temp_sig[:, -1, :] = self.tr_sig_i[:, i, :]
            else:
                continue
            # multi-linear regression & extraction
            temp_estimate, temp_extract = SRCA_lm_extract(temp_w, self.tr_w_o,
                temp_sig, self.tr_sig_o, regression, alpha, l1_ratio)
            del temp_estimate, temp_w, temp_sig  # release RAM
            # parameter computaion
            if evaluation == 'SNR':
                temp_para = np.mean(snr_time(temp_extract))
            elif evaluation == 'Corr':
                temp_para = np.mean(pearson_corr(temp_extract))
            compare_para[i] = temp_para - self.tr_para
            del temp_para
        # keep the best choice & refresh channel list
        chan_index = np.max(np.where(compare_para == np.max(compare_para)))
        self.remain_chans.append(self.train_chans[chan_index])
        self.para_change.append(np.max(compare_para))
        self.add_chan = chan_index
        pass
    
    def move_stepwise(self, evaluation='SNR', alpha=0.5, l1_ratio=0.5,
                      regression='OLS', first=False, loop=None):
        del_num = len(self.remain_chans) - 1  # except the last one
        add_num = len(self.train_chans)       # except the same channels
        compare_para = np.zeros((del_num, add_num))
        # delete 1 channel from existed data
        for i in range(del_num):
            # initialization (make copies)
            #temp_del_chan = copy.deepcopy(self.remain_chans)
            temp_del_sig = copy.deepcopy(self.core_sig)
            temp_del_w = copy.deepcopy(self.core_w)
            # delete one channel except the last one
            #del temp_del_chan[i]
            temp_del_sig = np.delete(temp_del_sig, i, axis=1)
            temp_del_w = np.delete(temp_del_w, i, axis=1)
            # then add one channel
            for j in range(len(self.train_chans)):
                if self.train_chans[j] not in self.remain_chans:
                    # initialization (make copies)
                    #temp_add_chan = copy.deepcopy(self.train_chans)
                    temp_add_w = np.zeros((self.tr_w_i.shape[0], loop, self.tr_w_i.shape[-1]))
                    temp_add_w[:, :(loop-1), :] = temp_del_w
                    temp_add_w[:, :-1, :] = self.tr_w_i[:, j, :]
                    
                    temp_add_sig = np.zeros((self.tr_sig_i.shape[0], loop, self.tr_sig_i.shape[-1]))
                    temp_add_sig[:, :(loop-1), :] = temp_del_sig
                    temp_add_sig[:, :-1, :] = self.tr_sig_i[:, j, :]
                    # multi-linear regression & extraction
                    temp_estimate, temp_extract = SRCA_lm_extract(temp_add_w,
                        self.tr_w_o, temp_add_sig, self.tr_sig_o, regression=regression,
                        alpha=alpha, l1_ratio=l1_ratio)
                    del temp_estimate, temp_add_w, temp_add_sig
                    # parameter computation
                    if evaluation == 'SNR':
                        temp_para = np.mean(snr_time(temp_extract))
                    elif evaluation == 'Corr':
                        temp_para = np.mean(pearson_corr(temp_extract))
                    compare_para[i,j] = temp_para - self.tr_para
                else:
                    compare_para[i,j] = -999
            # keep the best choice (del & add)
            del_item, add_item = np.where(compare_para == np.max(compare_para))
            del_item = np.max(del_item)
            add_item = np.max(add_item)
            # 
            pass
        pass
    
    # judge condition
    def check_end(self, loop, stop=True):
        if stop:  # maybe need to stop training
            if self.para_change[-1] < np.max(self.para_change):  # no improvement
                print('Stepwise optimization complete!')
                self.end  = time.clock()
                print('Training time: ' + str(self.end - self.start) + 's')
                status = False
            else:  # still has improvement
                print(str(loop) + 'th loop complete!')
                status = True
        else:  # no need to stop training
            if self.para_change[-1] < np.max(self.para_change):  # no improvement
                print('Already best in ' + str(loop) + 'th turn')
            else:  # still has improvement
                # save temp data
                print('fuck')
            status = True
        return status    
        pass
    
    # update train data
    def update(self, first=False, loop=None):
        if first:
            self.core_w = self.tr_w_i[:, self.add_chan, :]
            self.core_sig = self.tr_sig_i[:, self.add_chan, :]
        else:
            temp_core_w = np.zeros((self.core_w.shape[0], loop, self.core_w.shape[-1]))
            temp_core_w[:, :(loop-1), :] = self.core_w
            temp_core_w[:, -1, :] = self.tr_w_i[:, self.chan_index, :]
            self.core_w = temp_core_w
            del temp_core_w
            
            temp_core_sig = np.zeros((self.core_sig.shape[0], loop, self.core_sig.shape[-1]))
            temp_core_sig[:, :(loop-1), :] = self.core_sig
            temp_core_sig[:, -1, :] = self.tr_sig_i[:, self.chan_index, :]
            self.core_sig = temp_core_sig
            del temp_core_sig   
        pass
    
    # stepwise recursion
    def train(self, evaluation='SNR', regression='OLS', mode='stepwise',
              alpha=0.5, l1_ratio=0.5):
        print('SRCA training ({} + {} + {})...'.format(mode, evaluation, regression))
        self.start = time.perf_counter()
        max_loop = len(self.train_chans)
        # begin loop
        active = True
        loop = 1
        if mode == 'stepwise':
            for i in range(self.n_events):
                self.
            while active and len(self.train_chans) <= max_loop:
                if loop == 1:  # 1st round: just step forward
                    self.move_forward(evaluation=evaluation, regression=regression,
                                  alpha=alpha, l1_ratio=l1_ratio, first=True)
                    self.update(first=True)
                    print('1st loop complete')
                elif loop == 2:  # 2nd round: begin stepwise
                    # step forward
                    self.move_forward(evaluation=evaluation, regression=regression,
                                  alpha=alpha, l1_ratio=l1_ratio, loop=loop)
                    self.update() 
                    # step backward then forward
                    self.move_stepwise(evaluation=evaluation, regression=regression,
                                   alpha=alpha, l1_ratio=l1_ratio)
                    active = self.check_end(loop=loop, stop=False)  # check improved or not
                    self.update()
                else:  # num of model_chans > 2
                    self.move_forward(evaluation=evaluation, regression=regression,
                                  alpha=alpha, l1_ratio=l1_ratio)
                    active = self.check_end(loop=loop)  # check whether target is achieved
                    self.update()
                    self.move_stepwise(evaluation=evaluation, regression=regression,
                                   alpha=alpha, l1_ratio=l1_ratio)
                    active = self.check_end(loop=loop, stop=False)  # check improved or not
                    self.update()
                loop += 1
        elif mode == 'backward':
            pass
        elif mode == 'forward':
            pass
        pass

    # Fisher score recursive
    def fs_train(self, evaluation='snr', regression='OLS',
                       alpha=0.5, l1_ratio=0.5):
        pass
  
    # apply srca models into test dataset
    def apply_model(self):
        pass
    
    # check training effects
    def performance_evaluation(self, cross_validation=5, tr_te=True):
        #return acc
        pass
    