# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 13:25:59 2019

This program is used to plot various figures

@author: Brynhildr
"""

#%% import 3rd-part module
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

#%% Fig 1: Full-channels' correlation heatmap (after substraction)
# load data
data = io.loadmat()

# plot

# save figure
fig.subplots_adjust()
plt.savefig(r'F:\SSVEP\figures\weisiwen\full_chan_corr.png', dpi=600)

# release RAM
del data


#%% Fig 2: Heatmaps (inter-channel correlation comparisions)
# load data
data = io.loadmat()

# plot

# save figure
fig.subplots_adjust()
plt.savefig(r'F:\SSVEP\figures\weisiwen\inter-chan-corr.png', dpi=600)

# release RAM
del data


#%% Fig 3: Boxplots (R^2 or goodness of fit of models)
# load data
data = io.loadmat()

# plot

# save figure
fig.subplots_adjust()
plt.savefig(r'F:\SSVEP\figures\weisiwen\goodness.png', dpi=600)

# release RAM
del data


#%% Fig 4: Barplot (Bias of estimation)
# load data
data = io.loadmat()

# plot

# save figure
fig.subplots_adjust()
plt.savefig(r'F:\SSVEP\figures\weisiwen\bias.png', dpi=600)

# release RAM
del data

#%% Fig 5: Barplot (Cosine similarity (Normal & Tanimoto))
# load data
data = io.loadmat()

# plot

# save figure
fig.subplots_adjust()
plt.savefig(r'F:\SSVEP\figures\weisiwen\cos_sim.png', dpi=600)

# release RAM
del data


#%% Fig 6: Waveform of signal (Origin & estimate) (With zoom-in effect)
# load data
data = io.loadmat()

# plot

# save figure
fig.subplots_adjust()
plt.savefig(r'F:\SSVEP\figures\weisiwen\waveform_o&es.png', dpi=600)

# release RAM
del data


#%% Fig 7: Waveform of signal (Origin & extract) (With zoom-in effect)
# load data
data = io.loadmat()

# plot

# save figure
fig.subplots_adjust()
plt.savefig(r'F:\SSVEP\figures\weisiwen\waveform_o&ex.png', dpi=600)

# release RAM
del data


#%% Fig 8: Power of spectrum density (Origin & extract)
# load data
data = io.loadmat()

# plot

# save figure
fig.subplots_adjust()
plt.savefig(r'F:\SSVEP\figures\weisiwen\psd.png', dpi=600)

# release RAM
del data


#%% Fig 9: SNR in time domain (Origin & extract) (full & strain)
# load data
data = io.loadmat()

# plot

# save figure
fig.subplots_adjust()
plt.savefig(r'F:\SSVEP\figures\weisiwen\snr_t.png', dpi=600)

# release RAM
del data


#%% Fig 10: SNR in frequency domain (Line chart)
# load data
data = io.loadmat()

# plot

# save figure
fig.subplots_adjust()
plt.savefig(r'F:\SSVEP\figures\weisiwen\snr_f.png', dpi=600)

# release RAM
del data






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
plt.plot(fs[1,1,:], np.mean(sig_p[2,:,:], axis=0), label='origin:3-6s', color='red')
plt.plot(fs[1,1,:], np.mean(w1_p[2,:,:], axis=0), label='w1:0-1s', color='blue')
plt.plot(fs[1,1,:], np.mean(w2_p[2,:,:], axis=0), label='w2:1-2s', color='yellow')
plt.plot(fs[1,1,:], np.mean(w3_p[2,:,:], axis=0), label='w3:2-3s', color='green')
plt.legend(loc='best', fontsize=20)

#%%
def strain(X):
    strain = np.zeros((X.shape[0],int((X.shape[1]-1)/100)))
    
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