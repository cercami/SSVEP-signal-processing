# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 20:06:59 2019
Use benchmark dataset to complete my research
@author: Brynhildr
"""
#%% Import third part module
import numpy as np
from numpy import transpose
import scipy.io as io
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import seaborn as sns

import os

import mne
from mne.filter import filter_data
from sklearn.linear_model import LinearRegression

import signal_processing_function as SPF 

#%% prevent ticking 'F5'
???

#%%*************************Part I: processing data*************************
#%% Load benchmark dataset & relative information
# load data from .mat file
eeg = io.loadmat(r'E:\dataset\data\S15.mat')
info = io.loadmat(r'E:\dataset\Freq_Phase.mat')

data = eeg['data']
# (64, 1500, 40, 6) = (n_chans, n_times, n_events, n_blocks)
# total trials = n_conditions x n_blocks = 240
# all epochs are down-sampled to 250 Hz, HOLLY SHIT!

# reshape data array: (n_events, n_epochs, n_chans, n_times)
data = data.transpose((2, 3, 0, 1))  

# combine data array: np.concatenate(X, Y, axis=)

# condition infomation
sfreq = 250
freqs = info['freqs'].T
phases = info['phases'].T
del eeg, info


#%% load channels information from .txt file
channels = {}
file = open(r'F:\SSVEP\dataset\channel_info\weisiwen_chans.txt')
for line in file.readlines():
    line = line.strip()
    v = str(int(line.split(' ')[0]) - 1)
    k = line.split(' ')[1]
    channels[k] = v
file.close()

del v, k, file, line       # release RAM
     

#%% Load multiple data file & also can be used to process multiple data
# CAUTION: may lead to RAM crash (5-D array takes more than 6125MB)
# Now I know why people need 32G's RAM...PLEASE SKIP THIS PART!!!
filepath = r'E:\dataset\data'

filelist = []
for file in os.listdir(filepath):
    full_path = os.path.join(filepath, file)
    filelist.append(full_path)

i = 0
eeg = np.zeros((35, 64, 1500, 40, 6))
for file in filelist:
    temp = io.loadmat(file)
    eeg[i,:,:,:,:] = temp['data']
    i += 1
    
# add more codes here to achieve multiple data processing (PLEASE DON'T)
    
del temp, i, file, filelist, filepath, full_path


#%% load local data (extract from .cnt file)
eeg = io.loadmat(r'F:\SSVEP\dataset\preprocessed_data\weisiwen\raw_data.mat')

data = eeg['raw_data']
data *= 1e6

channels = eeg['chan_info']

del eeg

# basic info
sfreq = 1000

n_events = data.shape[0]
n_trials = data.shape[1]
n_chans = data.shape[2]
n_times = data.shape[3]


#%% Data preprocessing
# filtering
f_data = np.zeros((n_events, n_trials, n_chans, n_times))
for i in range(n_events):
    f_data[i,:,:,:] = filter_data(data[i,:,:,:], sfreq=sfreq, l_freq=5,
                      h_freq=40, n_jobs=6)

del i

# get data for linear regression
w1 = f_data[:,:,:,0:1000]           # 0-1s
w2 = f_data[:,:,:,1000:2000]        # 1-2s
w3 = f_data[:,:,:,2000:3000]        # 2-3s
w = f_data[:,:,:,0:3000]

# get data for comparision
signal_data = f_data[:,:,:,3000:]   # 3-6s

del f_data, data

#%% Correlation binarization
def binarization(X):
    compare = np.zeros((X.shape[0], X.shape[1]))
    for i in range(n_chans):
        for j in range(n_chans):
            if X[i,j] < 0:
                compare[i,j] = 0
            else:
                compare[i,j] = X[i,j]
    return compare

#%% Inter-channel correlation analysis: Spearman correlation coefficient
w1_corr = SPF.corr_coef(w1, 'spearman')
w2_corr = SPF.corr_coef(w2, 'spearman')
w3_corr = SPF.corr_coef(w3, 'spearman')
w_corr = SPF.corr_coef(w, 'spearman')

sig_corr = SPF.corr_coef(signal_data, mode='spearman')

compare_w1 = binarization(w1_corr - sig_corr)
compare_w2 = binarization(w2_corr - sig_corr)
compare_w3 = binarization(w2_corr - sig_corr)
compare = binarization(w_corr - sig_corr)

data_path = r'F:\SSVEP\dataset\preprocessed_data\weisiwen\64chan_corr_sp.mat'
io.savemat(data_path, {'signal':sig_corr,
                       'w1':w1_corr,
                       'w2':w2_corr,
                       'w3':w3_corr,
                       'w':w_corr,
                       'w1_sub':compare_w1,
                       'w2_sub':compare_w2,
                       'w3_sub':compare_w3,
                       'w_sub':compare})
    
del w1_corr, w2_corr, w3_corr, sig_corr, w_corr
del compare_w1, compare_w2, compare_w3, compare


#%% Reload correlation data (1st time)
corr = io.loadmat(r'F:\SSVEP\dataset\preprocessed_data\weisiwen\64chan_corr.mat')

w1_corr = corr['w1']
w2_corr = corr['w2']
w3_corr = corr['w3']

w1_sub_corr = corr['w1_sub']
w2_sub_corr = corr['w2_sub']
w3_sub_corr = corr['w3_sub']

del corr

#%% Automatically pick estimate channel and target channel


# pick input channels: C1, Cz, C2, C4, CP5
# choose output channels: POz

# w1 model data: 0-1000ms
w1_i = w1[:,:,24:29,:]
w1_o = w1[:,:,47,:]
w1_total = w1[:,:,[24,25,26,27,28,47],:]

# w2 model data: 1000-2000ms
w2_i = w2[:,:,24:29,:]
w2_o = w2[:,:,47,:]
w2_total = w2[:,:,[24,25,26,27,28,47],:]

# w3 model data: 2000-3000ms
w3_i = w3[:,:,24:29,:]
w3_o = w3[:,:,47,:]
w3_total = w3[:,:,[24,25,26,27,28,47],:]

# signal part data: 3000-6000ms
sig_i = signal_data[:,:,24:29,:]
sig_o = signal_data[:,:,47,:]
sig_total = signal_data[:,:,[24,25,26,27,28,47],:]

# release RAM
#del w1_sub_corr, w2_sub_corr, w3_sub_corr
 

#%% Prepare for checkboard plot (Spearman method)
w1_pick_corr = SPF.corr_coef(w1_total, 'spearman')
w2_pick_corr = SPF.corr_coef(w2_total, 'spearman')
w3_pick_corr = SPF.corr_coef(w3_total, 'spearman')

sig_pick_corr = SPF.corr_coef(sig_total, 'spearman')

data_path = r'F:\SSVEP\dataset\preprocessed_data\weisiwen\pick_chan_corr.mat'
io.savemat(data_path, {'w1':w1_pick_corr,
                       'w2':w2_pick_corr,
                       'w3':w3_pick_corr,
                       'sig':sig_pick_corr})
    
del w1_pick_corr, w2_pick_corr, w3_pick_corr, sig_pick_corr


#%% Spatial filter: multi-linear regression method

# regression coefficient, intercept, R^2
rc_w1, ri_w1, r2_w1 = SPF.mlr_analysis(w1_i, w1_o)
# w1 estimate & extract data: (n_events, n_epochs, n_times)
w1_mes_w1, w1_mex_w1 = SPF.sig_extract(rc_w1, w1_i, w1_o, ri_w1)

# the same but w2 part data:
rc_w2, ri_w2, r2_w2 = SPF.mlr_analysis(w2_i, w2_o, w2_i, 0)
w2_mes_w2, w2_mex_w2 = SPF.sig_extract(rc_w2, w2_i, w2_o, ri_w2)

# the same but w3 part data (use w2)
w2_mes_w3, w2_mex_w3 = SPF.sig_extract(rc_w2, w3_i, w3_o, ri_w2)

# the same but w3 part data (use w3)
rc_w3, ri_w3, r2_w3 = SPF.mlr_analysis(w3_i, w3_o, w3_i, 0)
w3_mes_w3, w3_mex_w3 = SPF.sig_extract(rc_w3, w3_i, w3_o, ri_w3)

# signal part data (use w1):
s_mes_w1, s_mex_w1 = SPF.sig_extract(rc_w1, sig_i, sig_o, ri_w1)
# signal part data (use w2):
s_mes_w2, s_mex_w2 = SPF.sig_extract(rc_w2, sig_i, sig_o, ri_w2)
# signal part data (use w3): 
s_mes_w3, s_mex_w3 = SPF.sig_extract(rc_w3, sig_i, sig_o, ri_w3)


#%% Spatial filter: inverse array method
# filter coefficient
sp_w1 = SPF.inv_spa(w1_i, w1_o)
# w1 estimate & extract data: (n_events, n_epochs, n_times)
w1_es_w1, w1_ex_w1 = SPF.sig_extract(sp_w1, w1_i, w1_o, 0)
# w1 model's goodness of fit
gf_w1 = SPF.fit_goodness(w1_o, w1_es_w1, chans=5)

# the same but w2 part data:
sp_w2 = SPF.inv_spa(w2_i, w2_o)
w2_es_w2, w2_ex_w2 = SPF.sig_extract(sp_w2, w2_i, w2_o, 0)
gf_w2 = SPF.fit_goodness(w2_o, w2_es_w2, chans=5)

# the same but w3 part data (use w2):
w2_es_w3, w2_ex_w3 = SPF.sig_extract(sp_w2, w3_i, w3_o, 0)

# the same but w3 part data (use w3):
sp_w3 = SPF.inv_spa(w3_i, w3_o)
w3_es_w3, w3_ex_w3 = SPF.sig_extract(sp_w3, w3_i, w3_o, 0)
gf_w3 = SPF.fit_goodness(w3_o, w3_es_w3, chans=5)

# signal part data (use w1):
s_ies_w1, s_iex_w1 = SPF.sig_extract(sp_w1, sig_i, sig_o, 0)
# signal part data (use w2):
s_ies_w2, s_iex_w2 = SPF.sig_extract(sp_w2, sig_i, sig_o, 0)
# signal part data (use w3):
s_ies_w3, s_iex_w3 = SPF.sig_extract(sp_w3, sig_i, sig_o, 0)


#%% Cosine similarity (background part): normal
# w1 estimate (w1 model) & w1 original, mlr, normal similarity, the same below
w1_w1_m_nsim = SPF.cos_sim(w1_o, w1_mes_w1, mode='normal')
w2_w2_m_nsim = SPF.cos_sim(w2_o, w2_mes_w2, mode='normal')
w2_w3_m_nsim = SPF.cos_sim(w3_o, w2_mes_w3, mode='normal')
w3_w3_m_nsim = SPF.cos_sim(w3_o, w3_mes_w3, mode='normal')

w1_w1_i_nsim = SPF.cos_sim(w1_o, w1_ies_w1, mode='normal')
w2_w2_i_nsim = SPF.cos_sim(w2_o, w2_ies_w2, mode='normal')
w2_w3_i_nsim = SPF.cos_sim(w3_o, w2_ies_w3, mode='normal')
w3_w3_i_nsim = SPF.cos_sim(w3_o, w3_ies_w3, mode='normal')


#%% Cosine similarity (background part): Tanimoto (generalized Jaccard)
# w1 estimate (w1 model) & w1 original, mlr, Tanimoto, the same below
w1_w1_m_tsim = SPF.cos_sim(w1_o, w1_mes_w1, mode='tanimoto')
w2_w2_m_tsim = SPF.cos_sim(w2_o, w2_mes_w2, mode='tanimoto')
w2_w3_m_tsim = SPF.cos_sim(w3_o, w2_mes_w3, mode='tanimoto')
w3_w3_m_tsim = SPF.cos_sim(w3_o, w3_mes_w3, mode='tanimoto')

w1_w1_i_tsim = SPF.cos_sim(w1_o, w1_ies_w1, mode='tanimoto')
w2_w2_i_tsim = SPF.cos_sim(w2_o, w2_ies_w2, mode='tanimoto')
w2_w3_i_tsim = SPF.cos_sim(w3_o, w2_ies_w3, mode='tanimoto')
w3_w3_i_tsim = SPF.cos_sim(w3_o, w3_ies_w3, mode='tanimoto')


#%% Power spectrum density
w1_p, f = SPF.welch_p(s_iex_w1, sfreq=sfreq, fmin=0, fmax=50, n_fft=1000,
                      n_overlap=250, n_per_seg=500)
w2_p, f = SPF.welch_p(s_iex_w2, sfreq=sfreq, fmin=0, fmax=50, n_fft=1000,
                      n_overlap=250, n_per_seg=500)
w3_p, f = SPF.welch_p(s_iex_w3, sfreq=sfreq, fmin=0, fmax=50, n_fft=1000,
                      n_overlap=250, n_per_seg=500)
sig_p, fn = SPF.welch_p(sig_o, sfreq=sfreq, fmin=0, fmax=50, n_fft=3000,
                       n_overlap=250, n_per_seg=500)


#%% Precise FFT transform

#%% Variance
# original signal variance
var_o_t = var_estimation(sig_o)

# extract signal variance (w1 model) 
var_w1_m_t = var_estimation(w1_mex_w1)
var_w1_i_t = var_estimation(w1_iex_w1)

# extract signal variance (w2 model) 
var_w2_m_t = var_estimation(w2_mex_w2)
var_w2_i_t = var_estimation(w2_iex_w2)

# extract signal variance (w3 model) 
var_w3_m_t = var_estimation(w3_mex_w3)
var_w3_i_t = var_estimation(w3_iex_w3)


#%% SNR in time domain
# original signal snr
snr_o_t = SPF.snr_time(sig_o, mode='time')

# extract signal snr (w1 model) 
#snr_w1_m_t = snr_time(s_mex_w1, mode='time')
snr_w1_i_t = SPF.snr_time(s_iex_w1, mode='time')

# extract signal snr (w2 model) 
#snr_w2_m_t = snr_time(s_mex_w2, mode='time')
snr_w2_i_t = SPF.snr_time(s_iex_w2, mode='time')

# extract signal snr (w3 model) 
#snr_w3_m_t = snr_time(s_mex_w3, mode='time')
snr_w3_i_t = SPF.snr_time(s_iex_w3, mode='time')


#%% SNR in frequency domain


#%%*************************Part II: plot figures*************************
#%% Model descrpition Part I: Boxplot & Histogram
fig = plt.figure(figsize=(24,24))
#fig.suptitle(r'$\ Model\ Description$', fontsize=30, fontweight='bold')
gs = GridSpec(6, 7, figure=fig)

# 1. Boxplot of R^2 
X = gf_w1.flatten()
Y = gf_w2.flatten()
Z = gf_w3.flatten()

xmin = min(np.min(X), np.min(Y), np.min(Z)) - 0.05

R2 = np.zeros((len(X) + len(Y) + len(Z)))
R2[0:len(X)] = X
R2[len(X):(len(X)+len(Y))] = Y
R2[(len(X)+len(Y)):(len(X) + len(Y) + len(Z))] = Z
model = ['w1' for i in range(len(X))]+['w2' for i in range(len(Y))]+['w3' for i in range(len(Z))]
R2 = pd.DataFrame({r'$\ model$': model, r'$\ R^2$': R2})

order=['w1', 'w2', 'w3']
sns.set(style='whitegrid')

ax1 = fig.add_subplot(gs[0:4, 0:4])
ax1.set_title(r"$\ 3\ model's\ R^2$", fontsize=30)
ax1.tick_params(axis='both', labelsize=22)
ax1.set_xlim((xmin, 1.05))
ax1 = sns.boxplot(x=r'$\ R^2$', y=r'$\ model$', data=R2, notch=True,
                  linewidth=2.5, orient='h', fliersize=10)
ax1 = sns.swarmplot(x=r'$\ R^2$', y=r'$\ model$', data=R2, color='dimgrey',
                    orient='h', size=5)
ax1.set_xlabel(r'$\ R^2\ values$', fontsize=26)
ax1.set_ylabel(r'$\ Models$', fontsize=26)


# 2. Histogram of R^2
ax2 = fig.add_subplot(gs[4:, 0:4])
ax2.set_title(r'$\ Distribution\ of\ R^2$', fontsize=30)
ax2.set_xlabel(r'$\ R^2\ values$', fontsize=26)
ax2.set_ylabel(r'$\ Frequence$', fontsize=26)
ax2.tick_params(axis='both', labelsize=22)
ax2.set_xlim((xmin, 1.05))
ax2 = sns.kdeplot(X, shade=True, label=r'$\ w1$')
ax2 = sns.kdeplot(Y, shade=True, label=r'$\ w2$')
ax2 = sns.kdeplot(Z, shade=True, label=r'$\ w3$')
ax2.legend(loc='best', fontsize=20)

del X, Y, Z


# 3. Inter-channel correlation (2 parts + comparision)
sns.set(style='white')
X = w1_pick_corr_sp
Y = sig_pick_corr_sp
Z = X - Y

# format decimal number & remove leading zeros & hide the diagonal elements
def func(x, pos):  
    return '{:.4f}'.format(x).replace('0.', '.').replace('1.0000', '').replace('.0000', '')
    
pick_chans = ['C1','Cz','C2','C4','CP5','POz']  # change each time

vmin = min(np.min(X), np.min(Y))
vmax = max(np.max(X), np.max(Y))

ax3 = fig.add_subplot(gs[0:2, 4:])
im, _ = SPF.check_plot(data=X, row_labels=pick_chans, col_labels=pick_chans,
                       ax=ax3, cmap='Blues', vmin=vmin, vmax=vmax)
SPF.check_annotate(im, valfmt=matplotlib.ticker.FuncFormatter(func), size=16)
ax3.set_xlabel(r'$\ w1\ part\ inter-channel\ correlation$', fontsize=30)
ax3.set_ylabel(r'$\ Channels$', fontsize=26)
ax3.tick_params(axis='both', labelsize=22)

ax4 = fig.add_subplot(gs[2:4, 4:])
im, _ = SPF.check_plot(data=Y, row_labels=pick_chans, col_labels=pick_chans,
                       ax=ax4, cmap='Blues', vmin=vmin, vmax=vmax)
SPF.check_annotate(im, valfmt=matplotlib.ticker.FuncFormatter(func), size=16)
ax4.set_xlabel(r'$\ SSVEP\ part\ inter-channel\ correlation$', fontsize=30)
ax4.set_ylabel(r'$\ Channels$', fontsize=26)
ax4.tick_params(axis='both', labelsize=22)

ax5 = fig.add_subplot(gs[4:, 4:])
im, _ = SPF.check_plot(data=Z, row_labels=pick_chans, col_labels=pick_chans,
                       ax=ax5, cmap='Reds', vmin=np.min(Z), vmax=np.max(Z))
SPF.check_annotate(im, valfmt=matplotlib.ticker.FuncFormatter(func), size=16)
ax5.set_xlabel(r'$\ Correlation\ comparision\ (w1-SSVEP)$', fontsize=30)
ax5.set_ylabel(r'$\ Channels$', fontsize=23)
ax5.tick_params(axis='both', labelsize=22)

del X, Y, Z

fig.subplots_adjust(top=0.949, bottom=0.05, left=0.049, right=0.990, 
                    hspace=1.000, wspace=0.7)
plt.savefig(r'D:\dataset\preprocessed_data\weisiwen\model_description.png', dpi=600)


#%% use sns module to plot heatmap
#fig, ax = plt.subplots()                    
#ax = sns.heatmap(Z, annot=True, fmt='.2g', linewidths=0.5, cmap='Reds', cbar=False,
                 #xticklabels=pick_chans, yticklabels=pick_chans, vmin=np.min(Z),
                 #vmax=np.max(Z))
#cbar = ax.figure.colorbar(ax.collections[0])
#cbar.ax.tick_params(labelsize=18)
#plt.xticks(fontsize=16)
#plt.yticks(fontsize=16)


#%% Signal waveform & Time SNR (with zoom-in effect: 0-500ms/0-125points)
# 1. signal waveform:
    # (1) original
    # (2) w1 model extract
    # (3) w2 model extract
    # (4) w3 model extract
fig = plt.figure(figsize=(16,16))
gs = GridSpec(6, 7, figure=fig)

ax1 = fig.add_subplot(gs[,])
ax1.set_title('signal', fontsize=20)
ax1.set_xlabel('time/ms', fontsize=20)
ax1.set_ylabel('SNR', fontsize=20)
ax1.plot(np.mean(sig_o[7,:,:], axis=0), label='origin:3-6s')
ax1.plot(np.mean(s_iex_w1[7,:,:], axis=0), label='w1:0-1s')
ax1.plot(np.mean(s_iex_w2[7,:,:], axis=0), label='w2:1-2s')
ax1.plot(np.mean(s_iex_w3[7,:,:], axis=0), label='w3:2-3s')
ax1.tick_params(axis='both', labelsize=20)
ax1.legend(loc='upper right', fontsize=20)

ax2 = fig.add_subplot(gs[,])
ax2.set_title('time snr', fontsize=20)
ax2.set_xlabel('time/ms', fontsize=20)
ax2.set_ylabel('SNR', fontsize=20)
ax2.plot(snr_o_t[7,:], label='origin:3-6s')
ax2.plot(snr_w1_i_t[7,:], label='w1:0-1s')
ax2.plot(snr_w2_i_t[7,:], label='w2:1-2s')
ax2.plot(snr_w3_i_t[7,:], label='w3:2-3s')
ax2.tick_params(axis='both', labelsize=20)
ax2.legend(loc='best', fontsize=20)


#%% plot PSD
plt.title('signal psd', fontsize=20)
plt.xlabel('frequency/Hz', fontsize=20)
plt.plot(fn[1,1,:], np.mean(sig_p[2,:,:], axis=0), label='origin:3-6s', color='red')
plt.plot(f[1,1,:], np.mean(w1_p[2,:,:], axis=0), label='w1:0-1s', color='blue')
plt.plot(f[1,1,:], np.mean(w2_p[2,:,:], axis=0), label='w2:1-2s', color='yellow')
plt.plot(f[1,1,:], np.mean(w3_p[2,:,:], axis=0), label='w3:2-3s', color='green')
plt.legend(loc='best', fontsize=20)

#%%
def strain(X):
    strain = np.zeros((40,50))
    
    for i in range(X.shape[0]):
        k=0
        for j in range(int(X.shape[1]/100)):
            strain[i,j] = np.mean(X[i,k:k+100])
            k += 100
    return strain

#%%
snr_o_t = strain(snr_o_t)
snr_w1_i_t = strain(snr_w1_i_t)
snr_w2_i_t = strain(snr_w2_i_t)
snr_w3_i_t = strain(snr_w3_i_t)

#%%
plt.plot(snr_o_t[2,:], label='origin')
plt.plot(snr_w1_i_t[2,:], label='w1')
plt.plot(snr_w2_i_t[2,:], label='w2')
plt.plot(snr_w3_i_t[2,:], label='w3')
plt.tick_params(axis='both', labelsize=20)
plt.legend(loc='best', fontsize=20)

#%% 
plt.plot(np.mean(s_ies_w1[2,:,:], axis=0), label='w1', color='dodgerblue')
plt.plot(np.mean(s_ies_w2[2,:,:], axis=0), label='w2', color='green')
plt.plot(np.mean(s_ies_w3[2,:,:], axis=0), label='w3', color='red')
plt.plot(np.mean(sig_o[2,:,:], axis=0), label='origin', color='grey')
plt.legend(loc='best')