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
data = io.loadmat(r'F:\SSVEP\dataset\preprocessed_data\weisiwen\64chan_corr.mat')
w_corr = data['w']
signal_corr = data['signal']
compare = w_corr - signal_corr

del data, signal_corr, w_corr

chan = io.loadmat(r'F:\SSVEP\dataset\preprocessed_data\weisiwen\chan_info.mat')
pick_chans = chan['chan_info'].tolist()

del chan

# plot
fig, ax = plt.subplots(figsize=(24,24))

im = ax.imshow(compare, cmap='Blues')

ax.set_xticks(np.arange(compare.shape[1]))
ax.set_yticks(np.arange(compare.shape[0]))
    
ax.set_xticklabels(pick_chans, fontsize=24)
ax.set_yticklabels(pick_chans, fontsize=24)

ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)
    
plt.setp(ax.get_xticklabels(), rotation=-60, ha='right', rotation_mode='anchor')

for edge, spine in ax.spines.items():
    spine.set_visible(False)
        
ax.set_xticks(np.arange(compare.shape[1]+1)-.5, minor=True)
ax.set_yticks(np.arange(compare.shape[0]+1)-.5, minor=True)
ax.grid(which='minor', color='w', linestyle='-', linewidth=3)
ax.tick_params(which='minor', bottom=False, left=False)

plt.show()

# save figure
fig.tight_layout()
plt.savefig(r'F:\SSVEP\figures\weisiwen\full_chan_corr.png', dpi=600)

# release RAM
del pick_chans, compare, vmin, vmax, edge


#%% Fig 1: Boxplot
# load data
mlr_A = io.loadmat(r'F:\SSVEP\dataset\preprocessed_data\weisiwen\14-18__POz\MLR_model.mat')
mlr_a = io.loadmat(r'F:\SSVEP\dataset\preprocessed_data\weisiwen\14-18__Oz\MLR_model.mat')
w1_gf_A = mlr_A['r2_w1'].flatten()
w2_gf_A = mlr_A['r2_w2'].flatten()
w3_gf_A = mlr_A['r2_w3'].flatten()
w1_gf_a = mlr_a['r2_w1'].flatten()
w2_gf_a = mlr_a['r2_w2'].flatten()
w3_gf_a = mlr_a['r2_w3'].flatten()
del mlr_A, mlr_a

mlr_B = io.loadmat(r'F:\SSVEP\dataset\preprocessed_data\weisiwen\19-23__POz\MLR_model.mat')
mlr_b = io.loadmat(r'F:\SSVEP\dataset\preprocessed_data\weisiwen\19-23__Oz\MLR_model.mat')
w1_gf_B = mlr_B['r2_w1'].flatten()
w2_gf_B = mlr_B['r2_w2'].flatten()
w3_gf_B = mlr_B['r2_w3'].flatten()
w1_gf_b = mlr_B['r2_w1'].flatten()
w2_gf_b = mlr_B['r2_w2'].flatten()
w3_gf_b = mlr_B['r2_w3'].flatten()
del mlr_B, mlr_b

mlr_C = io.loadmat(r'F:\SSVEP\dataset\preprocessed_data\weisiwen\24_28__POz\MLR_model.mat')
mlr_c = io.loadmat(r'F:\SSVEP\dataset\preprocessed_data\weisiwen\24_28__Oz\MLR_model.mat')
w1_gf_C = mlr_C['r2_w1'].flatten()
w2_gf_C = mlr_C['r2_w2'].flatten()
w3_gf_C = mlr_C['r2_w3'].flatten()
w1_gf_c = mlr_c['r2_w1'].flatten()
w2_gf_c = mlr_c['r2_w2'].flatten()
w3_gf_c = mlr_c['r2_w3'].flatten()
del mlr_C, mlr_c

mlr_D = io.loadmat(r'F:\SSVEP\dataset\preprocessed_data\weisiwen\34-35-36-43-44__POz\MLR_model.mat')
mlr_d = io.loadmat(r'F:\SSVEP\dataset\preprocessed_data\weisiwen\34-35-36-43-44__Oz\MLR_model.mat')
w1_gf_D = mlr_D['r2_w1'].flatten()
w2_gf_D = mlr_D['r2_w2'].flatten()
w3_gf_D = mlr_D['r2_w3'].flatten()
w1_gf_d = mlr_d['r2_w1'].flatten()
w2_gf_d = mlr_d['r2_w2'].flatten()
w3_gf_d = mlr_d['r2_w3'].flatten()
del mlr_D, mlr_d

# data refrom
A = ['w1: POz' for i in range(300)]
B = ['w2: POz' for i in range(300)]
C = ['w3: POz' for i in range(300)]
D = ['w1: Oz' for i in range(300)]
E = ['w2: Oz' for i in range(300)]
F = ['w3: Oz' for i in range(300)]
channel = A+D+A+D+A+D+A+D+B+E+B+E+B+E+B+E+C+F+C+F+C+F+C+F
del A, B, C, D, E, F

A = ['A' for i in range(300)]
B = ['B' for i in range(300)]
C = ['C' for i in range(300)]
D = ['D' for i in range(300)]
P = A+A+B+B+C+C+D+D
label = P+P+P
del A, B, C, D, P

gf = np.zeros((7200))

gf[0:300] = w1_gf_A
gf[300:600] = w1_gf_a
gf[600:900] = w1_gf_B
gf[900:1200] = w1_gf_b
gf[1200:1500] = w1_gf_C
gf[1500:1800] = w1_gf_c
gf[1800:2100] = w1_gf_D
gf[2100:2400] = w1_gf_d

gf[2400:2700] = w2_gf_A
gf[2700:3000] = w2_gf_a
gf[3000:3300] = w2_gf_B
gf[3300:3600] = w2_gf_b
gf[3600:3900] = w2_gf_C
gf[3900:4200] = w2_gf_c
gf[4200:4500] = w2_gf_D
gf[4500:4800] = w2_gf_d

gf[4800:5100] = w3_gf_A
gf[5100:5400] = w3_gf_a
gf[5400:5700] = w3_gf_B
gf[5700:6000] = w3_gf_b
gf[6000:6300] = w3_gf_C
gf[6300:6600] = w3_gf_c
gf[6600:6900] = w3_gf_D
gf[6900:7200] = w3_gf_d

del w1_gf_A, w1_gf_B, w1_gf_C, w1_gf_D
del w2_gf_A, w2_gf_B, w2_gf_C, w2_gf_D
del w3_gf_A, w3_gf_B, w3_gf_C, w3_gf_D
del w1_gf_a, w1_gf_b, w1_gf_c, w1_gf_d
del w2_gf_a, w2_gf_b, w2_gf_c, w2_gf_d
del w3_gf_a, w3_gf_b, w3_gf_c, w3_gf_d

data = pd.DataFrame({'label':label, 'Goodness of fit':gf, r'$\ Channel$':channel})

del gf, channel, label

# plot
sns.set(style='whitegrid')

fig, ax = plt.subplots(figsize=(21,21))
ax.set_title(r'$\ Model\ description$', fontsize=34)
ax.tick_params(axis='both', labelsize=28)
ax.set_ylim((0, 1.))
ax = sns.boxplot(x='label', y='Goodness of fit', hue=r'$\ Channel$', data=data,
                 palette='Set3', notch=True, fliersize=12)
ax.set_xlabel(r'$\ Test\ group$', fontsize=30)
ax.set_ylabel(r'$\ Goodness\ of\ fit$', fontsize=30)
ax.legend(loc='lower right', fontsize=32)

fig.tight_layout()
plt.show()

plt.savefig(r'F:\SSVEP\figures\weisiwen\model description.png', dpi=600)

del data


#%% Fig 2: Heatmap
# load data
corr_A = io.loadmat(r'F:\SSVEP\dataset\preprocessed_data\weisiwen\34-35-36-43-44__POz\pick_chan_corr.mat')
w = corr_A['w']
sig = corr_A['sig']
# pick_chans = data['chan_info'].tolist()
del corr

pick_chans_A = ['TP8', 'P7', 'P5', 'P8', 'PO7', 'POz']

# reform data
compare = w - sig

x1 = w1_r2.flatten()  # length=300
x2 = w1_gf.flatten()

y1 = w2_r2.flatten()
y2 = w2_gf.flatten()

z1 = w3_r2.flatten()
z2 = w3_gf.flatten()

del w1_r2, w2_r2, w3_r2, w1_gf, w2_gf, w3_gf 

GF = np.zeros((1800))

GF[0:300] = x1          # w1 r2
GF[300:600] = y1        # w2 r2
GF[600:900] = z1        # w3 r2
GF[900:1200] = x2       # w1 gf
GF[1200:1500] = y2      # w2 gf
GF[1500:1800] = z2      # w3 gf

w1 = ['w1' for i in range(300)]
w2 = ['w2' for i in range(300)]
w3 = ['w3' for i in range(300)]

model = w1 + w2 + w3 + w1 + w2 + w3
method = [r'$\ MLR$' for i in range(900)] + [r'$\ IA$' for i in range(900)]

box_data = pd.DataFrame({r'$\ Model$':model, r'$\ Goodness\ of\ fit$':GF, r'$\ Method$':method})

del w1, w2, w3, x1, x2, y1, y2, z1, z2
del model, method, GF

# plot
fig = plt.figure(figsize=(21,21))
gs = GridSpec(6, 9, figure=fig)

sns.set(style='whitegrid')

#palette = sns.color_palette('yellow', 'blue')
ax1 = fig.add_subplot(gs[:, 0:6])
ax1.set_title(r'$\ Model\ description$', fontsize=30)
ax1.tick_params(axis='both', labelsize=26)
ax1.set_xlim((0.5, 1.))
ax1 = sns.boxplot(x=r'$\ Goodness\ of\ fit$', y=r'$\ Model$', hue=r'$\ Method$',
                  data=box_data, palette='Set3', notch=True, fliersize=10)

ax1.set_xlabel(r'$\ Goodness\ of\ fit$', fontsize=28)
ax1.set_ylabel(r'$\ Time\ part$', fontsize=28)
ax1.legend(loc='best', fontsize=26)

# format decimal number & remove leading zeros & hide the diagonal elements
def func(x, pos):
    return '{:.4f}'.format(x).replace('0.', '.').replace('1.0000', '').replace('.0000', '')

vmin = min(np.min(w), np.min(sig))
vmax = max(np.max(w), np.max(sig))

sns.set(style='white')

ax2 = fig.add_subplot(gs[0:2,6:])
im, _ = SPF.check_plot(data=w, row_labels=pick_chans, col_labels=pick_chans,
                       ax=ax2, cmap='Blues', vmin=vmin, vmax=vmax)
SPF.check_annotate(im, valfmt=matplotlib.ticker.FuncFormatter(func), size=16)
ax2.set_xlabel(r'$\ Background\ correlation$', fontsize=30)
ax2.tick_params(axis='both', labelsize=22)

ax3 = fig.add_subplot(gs[2:4,6:])
im, _ = SPF.check_plot(data=sig, row_labels=pick_chans, col_labels=pick_chans,
                       ax=ax3, cmap='Blues', vmin=vmin, vmax=vmax)
SPF.check_annotate(im, valfmt=matplotlib.ticker.FuncFormatter(func), size=16)
ax3.set_xlabel(r'$\ Signal\ correlation$', fontsize=30)
ax3.tick_params(axis='both', labelsize=22)

ax4 = fig.add_subplot(gs[4:6,6:])
im, _ = SPF.check_plot(data=compare, row_labels=pick_chans, col_labels=pick_chans,
                       ax=ax4, cmap='Greens', vmin=np.min(compare), vmax=np.max(compare))
SPF.check_annotate(im, valfmt=matplotlib.ticker.FuncFormatter(func), size=16)
ax4.set_xlabel(r'$\ Substraction$', fontsize=30)
ax4.tick_params(axis='both', labelsize=22)

plt.show()

# save figure
fig.subplots_adjust(top=0.949, bottom=0.05, left=0.049, right=0.990, 
                    hspace=1.000, wspace=1.000)
plt.savefig(r'F:\SSVEP\figures\weisiwen\inter-chan-corr.png', dpi=600)

# release RAM
del box_data, compare, pick_chans
del w, sig, vmin, vmax


#%% Fig 3: Barplot (Bias of estimation)
# load data
mlr = io.loadmat(r'F:\SSVEP\dataset\preprocessed_data\weisiwen\24_28__POz\MLR_model.mat')
w1w1_mlr = mlr['w1_ex_w1']
w1w2_mlr = mlr['w1_ex_w2']
w1w3_mlr = mlr['w1_ex_w3']
w2w2_mlr = mlr['w2_ex_w2']
w2w3_mlr = mlr['w2_ex_w3']
w3w3_mlr = mlr['w3_ex_w3']
del mlr

ia = io.loadmat(r'F:\SSVEP\dataset\preprocessed_data\weisiwen\24_28__POz\MLR_model.mat')
w1w1_ia = ia['w1_ex_w1']
w1w2_ia = ia['w1_ex_w2']
w1w3_ia = ia['w1_ex_w3']
w2w2_ia = ia['w2_ex_w2']
w2w3_ia = ia['w2_ex_w3']
w3w3_ia = ia['w3_ex_w3']
del ia

# data reform

# plot

# save figure
fig.subplots_adjust()
plt.savefig(r'F:\SSVEP\figures\weisiwen\bias.png', dpi=600)

# release RAM
del data

#%% Fig 4: Barplot (Cosine similarity (Normal & Tanimoto))
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