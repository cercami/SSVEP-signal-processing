# -*- coding: utf-8 -*-
"""
Created on Tue Dec 24 15:46:50 2019

Task-Related Component Analysis

@author: Brynhildr
"""

#%% Import third part module
import numpy as np
import scipy.io as io
from scipy import signal

import copy

import matplotlib.pyplot as plt
import seaborn as sns

from mne.filter import filter_data

#%% Load data
eeg = io.loadmat(r'I:\SSVEP\dataset\preprocessed_data\weisiwen\raw_data.mat')
data = eeg['raw_data']
chans = eeg['chan_info'].tolist()
data *= 1e6  

del eeg

sfreq = 1000

sig = data[:, :, :, 3000:]   

n_events = sig.shape[0]
n_trials = sig.shape[1]
n_times = sig.shape[3]


del data

# pick channels from parietal and occipital areas
tar_chans = ['PZ ','PO5','PO3','POZ','PO4','PO6','O1 ','OZ ','O2 ']
n_chans = len(tar_chans)

pk_sig = np.zeros((n_events, n_trials, n_chans, n_times))
for i in range(n_chans):
    pk_sig[:,:,i,:] = sig[:, :, chans.index(tar_chans[i]), :]
del i

#%% Filter bank:8-88Hz (10 bands/ 8Hz each)
#%% IIR method
# each band contain 8Hz's useful information with 1Hz's buffer zone
n_bands = 10
# for example, 1st band is actually 7Hz-17Hz
sl_freq = [(8*(x+1)-4) for x in range(10)]  # the lowest stop frequencies
pl_freq = [(8*(x+1)-2) for x in range(10)]  # the lowest pass frequencies
ph_freq = [(8*(x+2)+2) for x in range(10)]  # the highest pass frequencies
sh_freq = [(8*(x+2)+4) for x in range(10)]  # the highest stop frequencies

# 5-D tension
fb_data = np.zeros((n_bands, n_events, n_trials, n_chans, n_times))

# make data through filter bank
for i in range(n_bands):
    # design Chebyshev-II iir band-pass filter
    b, a = signal.iirdesign(wp=[pl_freq[i], ph_freq[i]], ws=[sl_freq[i], sh_freq[i]],
                        gpass=3, gstop=18, analog=False, ftype='cheby1', fs=sfreq)
    # filter data forward and backward to achieve zero-phase
    fb_data[i,:,:,:,:] = signal.filtfilt(b=b, a=a, x=pk_sig, axis=-1)
    del a, b
    
del i, sl_freq, pl_freq, ph_freq, sh_freq
print('Filter bank construction complete!')

#%% IIR method 2
# each band from m*8Hz to 90Hz
n_bands = 10
l_freq = [(8*(x+1)-2) for x in range(n_bands)]
h_freq = 90

sns.set(style='whitegrid')

fig, ax = plt.subplots(2, 1, figsize=(8, 6))

# 5-D tension
fb_data = np.zeros((n_bands, n_events, n_trials, n_chans, n_times))

# design Chebyshev I bandpass filter
for i in range(n_bands):
    b, a = signal.cheby1(N=5, rp=0.5, Wn=[l_freq[i], h_freq], btype='bandpass',
                         analog=False, output='ba', fs=sfreq)
    # filter data forward and backward to achieve zero-phase
    fb_data[i,:,:,:,:] = signal.filtfilt(b=b, a=a, x=pk_sig, axis=-1)
    # plot figures
    w, h = signal.freqz(b, a)
    freq = (w*sfreq) / (2*np.pi)
    
    ax[0].plot(freq, 20*np.log10(abs(h)), label='band %d'%(i+1))
    ax[0].set_title('Frequency Response', fontsize=16)
    ax[0].set_ylabel('Amplitude/dB', fontsize=16)
    ax[0].set_xlabel('Frequency/Hz', fontsize=16)
    ax[0].set_xlim([0, 200])
    ax[0].set_ylim([-60, 5])
    ax[0].legend(loc='best')

    ax[1].plot(freq, np.unwrap(np.angle(h))*180/np.pi, label='band %d'%(i+1))
    ax[1].set_ylabel('Anlge(degrees)', fontsize=16)
    ax[1].set_xlabel('Frequency/Hz', fontsize=16)
    ax[1].set_xlim([0, 200])
    ax[1].legend(loc='best')
    
    del b, a, w, h, freq
del i
plt.show()

#%% FIR method
# each band contain 8Hz's useful information with 1Hz's buffer zone
n_bands = 10

l_freq = [(8*(x+1)-2) for x in range(n_bands)]
h_freq = 90

# 5-D tension
fb_data = np.zeros((n_bands, n_events, n_trials, n_chans, n_times))

# make data through filter bank
for i in range(n_bands):
    for j in range(pk_sig.shape[0]):
        fb_data[i,j,:,:,:] = filter_data(pk_sig[j,:,:,:], sfreq=sfreq,
               l_freq=l_freq[i], h_freq=h_freq[i], n_jobs=4, filter_length='500ms',
               l_trans_bandwidth=2, h_trans_bandwidth=2, method='fir',
               phase='zero', fir_window='hamming', fir_design='firwin2',
               pad='reflect_limited')

#%%
k=0
plt.plot(np.mean(np.mean(fb_data[0,k,:,:,:]-np.mean(fb_data[0,k,:,:,:]), axis=0), axis=0), label='band 1')
plt.plot(np.mean(np.mean(fb_data[1,k,:,:,:]-np.mean(fb_data[1,k,:,:,:]), axis=0), axis=0), label='band 2')
plt.plot(np.mean(np.mean(fb_data[2,k,:,:,:]-np.mean(fb_data[2,k,:,:,:]), axis=0), axis=0), label='band 3')
plt.plot(np.mean(np.mean(fb_data[3,k,:,:,:]-np.mean(fb_data[3,k,:,:,:]), axis=0), axis=0), label='band 4')
plt.plot(np.mean(np.mean(fb_data[4,k,:,:,:]-np.mean(fb_data[4,k,:,:,:]), axis=0), axis=0), label='band 5')
plt.plot(np.mean(np.mean(fb_data[5,k,:,:,:]-np.mean(fb_data[5,k,:,:,:]), axis=0), axis=0), label='band 6')
plt.plot(np.mean(np.mean(fb_data[6,k,:,:,:]-np.mean(fb_data[6,k,:,:,:]), axis=0), axis=0), label='band 7')
plt.plot(np.mean(np.mean(fb_data[7,k,:,:,:]-np.mean(fb_data[7,k,:,:,:]), axis=0), axis=0), label='band 8')
plt.plot(np.mean(np.mean(fb_data[8,k,:,:,:]-np.mean(fb_data[8,k,:,:,:]), axis=0), axis=0), label='band 9')
plt.plot(np.mean(np.mean(fb_data[9,k,:,:,:]-np.mean(fb_data[9,k,:,:,:]), axis=0), axis=0), label='band 10')
plt.legend(loc='best')

#%% Task-Related Component Analysis: Main function
# cross-validation
N = 10

acc_1 = []
acc_2 = []
acc_3 = []

print('Running TRCA algorithm...')

for i in range(N):
    a = i*10
    
    # devide training dataset: (n_bands, n_events, n_trials, n_chans, n_times)
    tr_fb_data = fb_data[:,:,a:a+int(n_trials/N),:,:]
    # devide test dataset: (n_bands, n_events, n_trials, n_chans, n_times)
    te_fb_data = copy.deepcopy(fb_data)
    te_fb_data = np.delete(te_fb_data, [a,a+1,a+2,a+3,a+4,a+5,a+6,a+7,a+8,a+9], axis=2)
    # template signal: (n_bands, n_events, n_chans, n_times)
    template = np.mean(te_fb_data, axis=2)
    del a

    # initialization
    q = np.zeros((n_bands, n_events, n_chans, n_chans))
    s = np.zeros((n_bands, n_events, n_chans, n_chans))

    # Matrix Q: Q={x|x=Cov(x1, x2)}, Q=R^(chans*chans)
    for a in range(n_bands):  # j for n_bands
        for b in range(n_events):  # k for n_events
            # flatten high dimension data into (n_chans, n_points)
            temp = np.zeros((n_chans, tr_fb_data.shape[2]*n_times))
            for c in range(n_chans):  # l for n_chans
                temp[c,:] = tr_fb_data[a,b,:,c,:].flatten()
                # compute inter-channel covariance matrix Q
                q[a,b,:,:] = np.cov(temp)
            del temp, c
        del b
    del a
    print('Matrix Q complete!')

    # Matrix S: (n_chans, n_chans)
    for a in range(n_bands):  # j for n_bands
        for b in range(n_events):  # k for n_events
            for c in range(n_chans):  # j1 (channel)
                for d in range(n_chans):  # j2 (channel)
                    cov = []  # initialization
                    for e in range(10):  # h1 (trial)
                        temp = np.zeros((2, n_times))
                        temp[0,:] = tr_fb_data[a,b,e,c,:]
                        for f in range(10):  # h2 (trial)
                            if f != e:  # h1 != h2 (different trials)
                                temp[1,:] = tr_fb_data[a,b,f,d,:]
                                cov.append(np.sum(np.tril(np.cov(temp), -1)))
                            else:
                                continue
                        del f
                    del e
                    s[a,b,c,d] = np.sum(cov)
                del d
            del c
        del b
    del a
    print('Matrix S complete!')

    # Optimal coefficient vector w
    # initialization
    qs = np.zeros((n_bands, n_events, n_chans, n_chans))
    e_value = np.zeros((n_bands, n_events, n_chans))            # eigen values
    e_vector = np.zeros((n_bands, n_events, n_chans, n_chans))  # eigen vectors
    w = np.zeros((n_bands, n_events, n_chans))                  # w

    # Pearson correlation coefficients: rou
    # (n_events(test dataset), n_trials(test dataset), n_events, n_bands)
    r = np.zeros((n_events, te_fb_data.shape[2], n_events, n_bands))

    for j in range(q.shape[0]):  # j for n_bands
        for k in range(q.shape[1]):  # k for n_events
            # transform array to matrix
            temp_q = np.mat(q[j,k,:,:])
            temp_s = np.mat(s[j,k,:,:])
            # Square Q^-1*S
            qs[j,k,:,:] = temp_q.I * temp_s
            del temp_q, temp_s
            # Eigenvalue & Eigenvector
            e_value[j,k,:], e_vector[j,k,:,:] = np.linalg.eig(qs[j,k,:,:])
            e_value = abs(e_value)
            e_vector = abs(e_vector)
            # choose the eigenvector w with largest eigenvalue
            w_index = np.max(np.where(e_value[j,k,:] == np.max(e_value[j,k,:])))
            w[j,k,:] = e_vector[j,k,:,w_index]
            del w_index
        del k
    del j
    print('Optimal coefficient vector computation complete!')

    # test dataset looping
    for j in range(r.shape[0]):  # j for n_events in test dataset
        for k in range(r.shape[1]):  # k for n_trials in test dataset
            # identification looping
            for l in range(r.shape[2]):  # l for n_events in identification
                for m in range(r.shape[3]):  # m for n_bands
                    temp = np.zeros((2, n_times))  # computation target: (2, n_times)
                    temp[0,:] = ((np.mat(te_fb_data[m,j,k,:,:])).T * np.mat(w[m,l,:]).T).T
                    temp[1,:] = ((np.mat(template[m,j,:,:])).T * np.mat(w[m,l,:]).T).T
                    # compute Pearson correlation coefficient
                    r[j,k,l,m] = np.sum(np.tril(np.corrcoef(temp), -1))
                    del temp
                del m
            del l
        del k
    del j
    r = r**2
    print('Maximum inter-trial correaltion complete!')

    # identification function a(m)
    a = np.array([(m+1)**-1.25+0.25 for m in range(10)])

    # initialization
    rou = np.zeros((r.shape[1], r.shape[0], r.shape[2]))
    identity = np.zeros((r.shape[0], r.shape[1]))

    # The feature for target identification: rou
    for j in range(r.shape[0]):  # j for n_events in test dataset
        for k in range(r.shape[1]):  # k for n_trials in test dataset
            for l in range(r.shape[2]):  # l for n_events in identication
                rou[k,j,l] = np.mat(a) * (np.mat(r[j,k,l,:])).T
            # classification
            identity[j,k] = np.max(np.where(rou[k,j,:] == np.max(rou[k,j,:])))
            del l
        del k
    del j
    print('Classification complete!')

    for j in range(identity.shape[0]):  # actual classes
        for k in range(identity.shape[1]):  # trials
            # right classification
            if identity[j,k] == j and j == 0:  
                acc_1.append(1)
            elif identity[j,k] == j and j == 1:
                acc_2.append(1)
            elif identity[j,k] == j  and j == 2:
                acc_3.append(1)
        del k
    del j
    print(str(i+1) + 'th cross-validation complete!')
    
#%%
N = 10

acc_1 = []
acc_2 = []
acc_3 = []

i=0
#%%
a = i*10
    
# devide training dataset: (n_bands, n_events, n_trials, n_chans, n_times)
tr_fb_data = fb_data[:,:,a:a+10,:,:]
# devide test dataset: (n_bands, n_events, n_trials, n_chans, n_times)
te_fb_data = copy.deepcopy(fb_data)
te_fb_data = np.delete(te_fb_data, [a,a+1,a+2,a+3,a+4,a+5,a+6,a+7,a+8,a+9], axis=2)
# template signal: (n_bands, n_events, n_chans, n_times)
template = np.mean(te_fb_data, axis=2)

# initialization
q = np.zeros((n_bands, n_events, n_chans, n_chans))
s = np.zeros((n_bands, n_events, n_chans, n_chans))

# Matrix Q: Q={x|x=Cov(x1, x2)}, Q=R^(chans*chans)
for j in range(n_bands):  # j for n_bands
    for k in range(n_events):  # k for n_events
        # flatten high dimension data into (n_chans, n_points)
        temp = np.zeros((n_chans, tr_fb_data.shape[2]*n_times))
        for l in range(n_chans):  # l for n_chans
            temp[l,:] = tr_fb_data[j,k,:,l,:].flatten()
            # compute inter-channel covariance matrix Q
            q[j,k,:,:] = np.cov(temp)
        del temp, l
    del k
del j
print('Matrix Q complete!')

# Matrix S
s = np.zeros((n_bands,n_events,n_chans,n_chans))
for a in range(n_bands):
    for b in range(n_events):
        for c in range(n_chans):  # c for n_chans (j1)
            for d in range(n_chans):  # d for n_chans (j2)
                cov=[]
                for e in range(10):  # e for n_trials (h1)
                    temp = np.zeros((2, n_times))
                    temp[0,:] = tr_fb_data[a,b,e,c,:]
                    for f in range(10):  # f for n_trials (h2)
                        if f != e:
                            temp[1,:] = tr_fb_data[a,b,f,d,:]
                            cov.append(np.sum(np.tril(np.cov(temp), -1)))
                        else:
                            continue
                    del f
                del e
                s[a,b,c,d] = np.sum(cov)
            del d
        del c
    del b
del a
#%
# Optimal coefficient vector w
# initialization
qs = np.zeros((n_bands, n_events, n_chans, n_chans))
e_value = np.zeros((n_bands, n_events, n_chans))            # eigen values
e_vector = np.zeros((n_bands, n_events, n_chans, n_chans))  # eigen vectors
w = np.zeros((n_bands, n_events, n_chans))                  # w

# Pearson correlation coefficients rou:
# (n_events(test dataset), n_trials(test dataset), n_events, n_bands)
r = np.zeros((n_events, te_fb_data.shape[2], n_events, n_bands))

for j in range(q.shape[0]):  # j for n_bands
    for k in range(q.shape[1]):  # k for n_events
        # transform array to matrix
        temp_q = np.mat(q[j,k,:,:])
        temp_s = np.mat(s[j,k,:,:])
        # Square Q^-1*S
        qs[j,k,:,:] = temp_q.I * temp_s
        del temp_q, temp_s
        # Eigenvalue & Eigenvector
        e_value[j,k,:], e_vector[j,k,:,:] = np.linalg.eig(qs[j,k,:,:])
        # choose the eigenvector w with largest eigenvalue
        w_index = np.max(np.where(e_value[j,k,:] == np.max(e_value[j,k,:])))
        w[j,k,:] = e_vector[j,k,:,w_index]
        del w_index
    del k
del j
print('Optimal coefficient vector computation complete!')

# test dataset looping
for j in range(r.shape[0]):  # j for n_events in test dataset
    for k in range(r.shape[1]):  # k for n_trials in test dataset
        # identification looping
        for l in range(r.shape[2]):  # l for n_events in identification
            for m in range(r.shape[3]):  # m for n_bands
                temp = np.zeros((2, n_times))  # computation target: (2, n_times)
                temp[0,:] = ((np.mat(te_fb_data[m,j,k,:,:])).T * np.mat(w[m,l,:]).T).T
                temp[1,:] = ((np.mat(template[m,j,:,:])).T * np.mat(w[m,l,:]).T).T
                # compute Pearson correlation coefficient
                r[j,k,l,m] = np.sum(np.tril(np.corrcoef(temp), -1))
                del temp
            del m
        del l
    del k
del j
r = r**2
print('Maximum inter-trial correaltion complete!')

# identification function a(m)
a = np.array([((m+1)**-1.25 + 0.25) for m in range(10)])

# initialization
# (n_trials(test dataset), n_events(test dataset), n_events(identification))
rou = np.zeros((r.shape[1], r.shape[0], r.shape[2]))
identity = np.zeros((r.shape[0], r.shape[1]))

# The feature for target identification: rou
for j in range(r.shape[0]):  # j for n_events in test dataset
    for k in range(r.shape[1]):  # k for n_trials in test dataset
        for l in range(r.shape[2]):  # l for n_events in identication
            rou[k,j,l] = np.mat(a) * (np.mat(r[j,k,l,:])).T
        # classification
        identity[j,k] = np.max(np.where(rou[k,j,:] == np.max(rou[k,j,:])))
        del l
    del k
del j
print('Classification complete!')

for j in range(identity.shape[0]):  # actual classes
    for k in range(identity.shape[1]):  # trials
        # right classification
        if identity[j,k] == j and j == 0:  
            acc_1.append(1)
        elif identity[j,k] == j and j == 1:
            acc_2.append(1)
        elif identity[j,k] == j  and j == 2:
            acc_3.append(1)
    del k
del j
print(str(i+1) + 'th cross-validation complete!')

i+=1

#%%
acc_1 = np.sum(acc_1)/900*100
acc_2 = np.sum(acc_2)/900*100
acc_3 = np.sum(acc_3)/900*100 
