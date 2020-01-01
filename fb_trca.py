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

#%% Load data
eeg = io.loadmat(r'I:\SSVEP\dataset\preprocessed_data\weisiwen\raw_data.mat')
data = eeg['raw_data']
chans = eeg['chan_info'].tolist()
data *= 1e6  

del eeg

sfreq = 1000

sig = data[:, :, :, 3140:3240]   

n_events = sig.shape[0]
n_trials = sig.shape[1]
n_times = sig.shape[3]

del data

# pick channels from parietal and occipital areas
tar_chans = ['POZ','OZ ']
n_chans = len(tar_chans)

pk_sig = np.zeros((n_events, n_trials, n_chans, n_times))
for i in range(n_chans):
    pk_sig[:,:,i,:] = sig[:, :, chans.index(tar_chans[i]), :]
del i

#%% Filter bank:8-88Hz (10 bands/ 8Hz each): IIR method
# each band from m*8Hz-2Hz to 90Hz (i.e. with 2Hz's buffer)
n_bands = 5
l_freq = [(8*(x+1)-2) for x in range(n_bands)]
h_freq = 90

# 5-D tension
fb_data = np.zeros((n_events, n_bands, n_trials, n_chans, n_times))

# design Chebyshev I bandpass filter
for i in range(n_bands):
    b, a = signal.cheby1(N=5, rp=0.5, Wn=[l_freq[i], h_freq], btype='bandpass',
                         analog=False, output='ba', fs=sfreq)
    # filter data forward and backward to achieve zero-phase
    fb_data[:,i,:,:,:] = signal.filtfilt(b=b, a=a, x=pk_sig, axis=-1) 
    
    del b, a
del i

#%% Task-Related Component Analysis: Main function
# cross-validation
N = 10

acc_1 = []
acc_2 = []
acc_3 = []

print('Running TRCA algorithm...')

for i in range(N):   
    # Divide dataset
    a = i * 10
    # training dataset: (n_events, n_bands, n_trials, n_chans, n_times)
    tr_fb_data = fb_data[:,:,a:a+int(n_trials/N),:,:]
    # test dataset: (n_events, n_bands, n_trials, n_chans, n_times)
    te_fb_data = copy.deepcopy(fb_data)
    te_fb_data = np.delete(te_fb_data, [a,a+1,a+2,a+3,a+4,a+5,a+6,a+7,a+8,a+9], axis=2)
    # template data: (n_events, n_bands, n_chans, n_times)| basic unit: (n_chans, n_times)
    template = np.mean(tr_fb_data, axis=2)

    # Matrix Q: inter-channel covariance
    q = np.zeros((n_events, n_bands, n_chans, n_chans))
    # all events(n), all bands(m)
    for x in range(n_events):  # x for events (n)
        for y in range(n_bands):  # y for bands (m)
            temp = np.zeros((n_chans, int(n_trials/N*n_times)))
            for z in range(n_chans):  # z for channels
                # concatenated matrix of all trials in training dataset
                temp[z,:] = tr_fb_data[2,y,:,z,:].flatten()
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
                    for y in range(int(n_trials/N)):  # y for trials (h1)
                        temp = np.zeros((2, n_times))
                        temp[0,:] = tr_fb_data[u,v,y,w,:]
                        for z in range(int(n_trials/N)):  # z for trials (h2)
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
    r = np.zeros((n_events, n_trials-int(n_trials/N), n_bands, n_events))
    for v in range(n_events): # events in test dataset
        for x in range(n_trials-int(n_trials/N)):  # trials in test dataset (of one event)
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
    rou = np.zeros((n_events, n_trials-int(n_trials/N), n_events))

    for y in range(n_events):
        for z in range(n_trials-int(n_trials/N)):  # trials in test dataset (of one event)
            # (yth event, zth trial) test data | will have n_events' value, here is 3
            # the location of the largest value refers to the class of this trial
            rou[y,z,:] = a * np.mat(r[y,z,:,:])
      
    # compute accuracy
    for x in range(rou.shape[0]):  # ideal classification
        if x == 0:  # should be in 1st class
            for y in range(rou.shape[1]):
                if np.max(np.where(rou[x,y,:] == np.max(rou[x,y,:]))) == 0:  # correct
                    acc_1.append(1)
        if x == 1:  # should be in 2nd class
            for y in range(rou.shape[1]):
                if np.max(np.where(rou[x,y,:] == np.max(rou[x,y,:]))) == 1:  # correct
                    acc_2.append(1)
        if x == 2:  # should be in 3rd class
            for y in range(rou.shape[1]):
                if np.max(np.where(rou[x,y,:] == np.max(rou[x,y,:]))) == 2:  # correct
                    acc_3.append(1)

    # end                                
    print(str(i+1) + 'th cross-validation complete!\n')

acc = np.sum(acc_1) + np.sum(acc_2) + np.sum(acc_3)