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


#%% prevent pressing F5
???

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


#%% Fig 2-1: Boxplot (MLR)
# load data
mlr_A = io.loadmat(r'F:\SSVEP\dataset\preprocessed_data\weisiwen\14_18__POZ\MLR_model.mat')
mlr_a = io.loadmat(r'F:\SSVEP\dataset\preprocessed_data\weisiwen\14_18__OZ\MLR_model.mat')
w1_gf_A = mlr_A['r2_w1'].flatten()
w2_gf_A = mlr_A['r2_w2'].flatten()
w3_gf_A = mlr_A['r2_w3'].flatten()
w1_gf_a = mlr_a['r2_w1'].flatten()
w2_gf_a = mlr_a['r2_w2'].flatten()
w3_gf_a = mlr_a['r2_w3'].flatten()
del mlr_A, mlr_a

mlr_B = io.loadmat(r'F:\SSVEP\dataset\preprocessed_data\weisiwen\19_23__POZ\MLR_model.mat')
mlr_b = io.loadmat(r'F:\SSVEP\dataset\preprocessed_data\weisiwen\19_23__OZ\MLR_model.mat')
w1_gf_B = mlr_B['r2_w1'].flatten()
w2_gf_B = mlr_B['r2_w2'].flatten()
w3_gf_B = mlr_B['r2_w3'].flatten()
w1_gf_b = mlr_b['r2_w1'].flatten()
w2_gf_b = mlr_b['r2_w2'].flatten()
w3_gf_b = mlr_b['r2_w3'].flatten()
del mlr_B, mlr_b

mlr_C = io.loadmat(r'F:\SSVEP\dataset\preprocessed_data\weisiwen\24_28__POZ\MLR_model.mat')
mlr_c = io.loadmat(r'F:\SSVEP\dataset\preprocessed_data\weisiwen\24_28__OZ\MLR_model.mat')
w1_gf_C = mlr_C['r2_w1'].flatten()
w2_gf_C = mlr_C['r2_w2'].flatten()
w3_gf_C = mlr_C['r2_w3'].flatten()
w1_gf_c = mlr_c['r2_w1'].flatten()
w2_gf_c = mlr_c['r2_w2'].flatten()
w3_gf_c = mlr_c['r2_w3'].flatten()
del mlr_C, mlr_c

mlr_D = io.loadmat(r'F:\SSVEP\dataset\preprocessed_data\weisiwen\34-35-36-43-44__POZ\MLR_model.mat')
mlr_d = io.loadmat(r'F:\SSVEP\dataset\preprocessed_data\weisiwen\34-35-36-43-44__OZ\MLR_model.mat')
w1_gf_D = mlr_D['r2_w1'].flatten()
w2_gf_D = mlr_D['r2_w2'].flatten()
w3_gf_D = mlr_D['r2_w3'].flatten()
w1_gf_d = mlr_d['r2_w1'].flatten()
w2_gf_d = mlr_d['r2_w2'].flatten()
w3_gf_d = mlr_d['r2_w3'].flatten()
del mlr_D, mlr_d

# data refrom
W1 = ['w1: POZ' for i in range(300)] + ['w1: OZ' for i in range(300)]
W2 = ['w2: POZ' for i in range(300)] + ['w2: OZ' for i in range(300)]
W3 = ['w3: POZ' for i in range(300)] + ['w3: OZ' for i in range(300)]
channel = W1 + W1 + W1 + W1 + W2 + W2 + W2 + W2 + W3 + W3 + W3 + W3
del W1, W2, W3

A = ['A' for i in range(600)]
B = ['B' for i in range(600)]
C = ['C' for i in range(600)]
D = ['D' for i in range(600)]
P = A + B + C + D
label = P + P + P
del A, B, C, D, P

gf_1 = np.hstack((w1_gf_A, w1_gf_a, w1_gf_B, w1_gf_b, w1_gf_C, w1_gf_c, w1_gf_D, w1_gf_d))
del w1_gf_A, w1_gf_a, w1_gf_B, w1_gf_b, w1_gf_C, w1_gf_c, w1_gf_D, w1_gf_d

gf_2 = np.hstack((w2_gf_A, w2_gf_a, w2_gf_B, w2_gf_b, w2_gf_C, w2_gf_c, w2_gf_D, w2_gf_d))
del w2_gf_A, w2_gf_a, w2_gf_B, w2_gf_b, w2_gf_C, w2_gf_c, w2_gf_D, w2_gf_d

gf_3 = np.hstack((w3_gf_A, w3_gf_a, w3_gf_B, w3_gf_b, w3_gf_C, w3_gf_c, w3_gf_D, w3_gf_d))
del w3_gf_A, w3_gf_a, w3_gf_B, w3_gf_b, w3_gf_C, w3_gf_c, w3_gf_D, w3_gf_d

gf = np.hstack((gf_1, gf_2, gf_3))
del gf_1, gf_2, gf_3

data = pd.DataFrame({'label':label, 'Goodness of fit':gf, 'Channel':channel})

del gf, channel, label

# plot
sns.set(style='whitegrid')

fig, ax = plt.subplots(figsize=(21,21))
ax.set_title('Model description: multi-linear regression', fontsize=34)
ax.tick_params(axis='both', labelsize=28)
ax.set_ylim((0, 1.))
ax = sns.boxplot(x='label', y='Goodness of fit', hue='Channel', data=data,
                 palette='Set3', notch=True, fliersize=12)
ax.set_xlabel('Test group', fontsize=30)
ax.set_ylabel('Goodness of fit', fontsize=30)
ax.legend(loc='lower right', fontsize=32)

fig.tight_layout()
plt.show()

plt.savefig(r'F:\SSVEP\figures\weisiwen\model description_MLR.png', dpi=600)

del data


#%% Fig 2-2: Boxplot (IA)
# load data
mlr_A = io.loadmat(r'F:\SSVEP\dataset\preprocessed_data\weisiwen\14_18__POZ\inver_array_model.mat')
mlr_a = io.loadmat(r'F:\SSVEP\dataset\preprocessed_data\weisiwen\14_18__OZ\inver_array_model.mat')
w1_gf_A = mlr_A['gf_w1'].flatten()
w2_gf_A = mlr_A['gf_w2'].flatten()
w3_gf_A = mlr_A['gf_w3'].flatten()
w1_gf_a = mlr_a['gf_w1'].flatten()
w2_gf_a = mlr_a['gf_w2'].flatten()
w3_gf_a = mlr_a['gf_w3'].flatten()
del mlr_A, mlr_a

mlr_B = io.loadmat(r'F:\SSVEP\dataset\preprocessed_data\weisiwen\19_23__POZ\inver_array_model.mat')
mlr_b = io.loadmat(r'F:\SSVEP\dataset\preprocessed_data\weisiwen\19_23__OZ\inver_array_model.mat')
w1_gf_B = mlr_B['gf_w1'].flatten()
w2_gf_B = mlr_B['gf_w2'].flatten()
w3_gf_B = mlr_B['gf_w3'].flatten()
w1_gf_b = mlr_b['gf_w1'].flatten()
w2_gf_b = mlr_b['gf_w2'].flatten()
w3_gf_b = mlr_b['gf_w3'].flatten()
del mlr_B, mlr_b

mlr_C = io.loadmat(r'F:\SSVEP\dataset\preprocessed_data\weisiwen\24_28__POZ\inver_array_model.mat')
mlr_c = io.loadmat(r'F:\SSVEP\dataset\preprocessed_data\weisiwen\24_28__OZ\inver_array_model.mat')
w1_gf_C = mlr_C['gf_w1'].flatten()
w2_gf_C = mlr_C['gf_w2'].flatten()
w3_gf_C = mlr_C['gf_w3'].flatten()
w1_gf_c = mlr_c['gf_w1'].flatten()
w2_gf_c = mlr_c['gf_w2'].flatten()
w3_gf_c = mlr_c['gf_w3'].flatten()
del mlr_C, mlr_c

mlr_D = io.loadmat(r'F:\SSVEP\dataset\preprocessed_data\weisiwen\34-35-36-43-44__POZ\inver_array_model.mat')
mlr_d = io.loadmat(r'F:\SSVEP\dataset\preprocessed_data\weisiwen\34-35-36-43-44__OZ\inver_array_model.mat')
w1_gf_D = mlr_D['gf_w1'].flatten()
w2_gf_D = mlr_D['gf_w2'].flatten()
w3_gf_D = mlr_D['gf_w3'].flatten()
w1_gf_d = mlr_d['gf_w1'].flatten()
w2_gf_d = mlr_d['gf_w2'].flatten()
w3_gf_d = mlr_d['gf_w3'].flatten()
del mlr_D, mlr_d

# data refrom
W1 = ['w1: POZ' for i in range(300)] + ['w1: OZ' for i in range(300)]
W2 = ['w2: POZ' for i in range(300)] + ['w2: OZ' for i in range(300)]
W3 = ['w3: POZ' for i in range(300)] + ['w3: OZ' for i in range(300)]
channel = W1 + W1 + W1 + W1 + W2 + W2 + W2 + W2 + W3 + W3 + W3 + W3
del W1, W2, W3

A = ['A' for i in range(600)]
B = ['B' for i in range(600)]
C = ['C' for i in range(600)]
D = ['D' for i in range(600)]
P = A + B + C + D
label = P + P + P
del A, B, C, D, P

gf_1 = np.hstack((w1_gf_A, w1_gf_a, w1_gf_B, w1_gf_b, w1_gf_C, w1_gf_c, w1_gf_D, w1_gf_d))
del w1_gf_A, w1_gf_a, w1_gf_B, w1_gf_b, w1_gf_C, w1_gf_c, w1_gf_D, w1_gf_d

gf_2 = np.hstack((w2_gf_A, w2_gf_a, w2_gf_B, w2_gf_b, w2_gf_C, w2_gf_c, w2_gf_D, w2_gf_d))
del w2_gf_A, w2_gf_a, w2_gf_B, w2_gf_b, w2_gf_C, w2_gf_c, w2_gf_D, w2_gf_d

gf_3 = np.hstack((w3_gf_A, w3_gf_a, w3_gf_B, w3_gf_b, w3_gf_C, w3_gf_c, w3_gf_D, w3_gf_d))
del w3_gf_A, w3_gf_a, w3_gf_B, w3_gf_b, w3_gf_C, w3_gf_c, w3_gf_D, w3_gf_d

gf = np.hstack((gf_1, gf_2, gf_3))
del gf_1, gf_2, gf_3

data = pd.DataFrame({'label':label, 'Goodness of fit':gf, 'Channel':channel})

del gf, channel, label

# plot
sns.set(style='whitegrid')

fig, ax = plt.subplots(figsize=(21,21))
ax.set_title('Model description: inverse array method', fontsize=34)
ax.tick_params(axis='both', labelsize=28)
ax.set_ylim((0, 1.))
ax = sns.boxplot(x='label', y='Goodness of fit', hue='Channel', data=data,
                 palette='Set3', notch=True, fliersize=12)
ax.set_xlabel('Test group', fontsize=30)
ax.set_ylabel('Goodness of fit', fontsize=30)
ax.legend(loc='lower right', fontsize=32)

fig.tight_layout()
plt.show()

plt.savefig(r'F:\SSVEP\figures\weisiwen\model description_IA.png', dpi=600)

del data


#%% Fig 3: Heatmap
# load data
corr_A = io.loadmat(r'F:\SSVEP\dataset\preprocessed_data\weisiwen\14_18__OZ\pick_chan_corr.mat')
w_A = corr_A['w']
sig_A = corr_A['sig']
# pick_chans_A = corr_A['chan_info'].tolist()
pick_chans_A = ['FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'OZ']
del corr_A

corr_B = io.loadmat(r'F:\SSVEP\dataset\preprocessed_data\weisiwen\19_23__OZ\pick_chan_corr.mat')
w_B = corr_B['w']
sig_B = corr_B['sig']
# pick_chans_B = corr_B['chan_info'].tolist()
pick_chans_B = ['FC2', 'FC4', 'T7', 'C5', 'C3', 'OZ']
del corr_B

corr_C = io.loadmat(r'F:\SSVEP\dataset\preprocessed_data\weisiwen\24_28__OZ\pick_chan_corr.mat')
w_C = corr_C['w']
sig_C = corr_C['sig']
# pick_chans_C = corr_C['chan_info'].tolist()
pick_chans_C = ['C1', 'CZ', 'C2', 'C4', 'CP5', 'OZ']
del corr_C

corr_D = io.loadmat(r'F:\SSVEP\dataset\preprocessed_data\weisiwen\34-35-36-43-44__OZ\pick_chan_corr.mat')
w_D = corr_D['w']
sig_D = corr_D['sig']
# pick_chans_D = corr_D['chan_info'].tolist()
pick_chans_D = ['TP8', 'P7', 'P5', 'P8', 'PO7', 'OZ']
del corr_D

# reform data
compare_A = w_A - sig_A
compare_B = w_B - sig_B
compare_C = w_C - sig_C
compare_D = w_D - sig_D

vmin_A = min(np.min(w_A), np.min(sig_A))
vmax_A = max(np.max(w_A), np.max(sig_A))

vmin_B = min(np.min(w_B), np.min(sig_B))
vmax_B = max(np.max(w_B), np.max(sig_B))

vmin_C = min(np.min(w_C), np.min(sig_C))
vmax_C = max(np.max(w_C), np.max(sig_C))

vmin_D = min(np.min(w_D), np.min(sig_D))
vmax_D = max(np.max(w_D), np.max(sig_D))

# plot
fig = plt.figure(figsize=(24,15))
gs = GridSpec(6, 12, figure=fig)

# format decimal number & remove leading zeros & hide the diagonal elements
def func(x, pos):
    return '{:.4f}'.format(x).replace('0.', '.').replace('1.0000', '').replace('.0000', '')

sns.set(style='white')

ax1 = fig.add_subplot(gs[0:2,0:3])
im, _ = SPF.check_plot(data=w_A, row_labels=pick_chans_A, col_labels=pick_chans_A,
                       ax=ax1, cmap='Blues', vmin=vmin_A, vmax=vmax_A)
SPF.check_annotate(im, valfmt=matplotlib.ticker.FuncFormatter(func), size=16)
ax1.set_xlabel('Background correlation (A)', fontsize=22)
ax1.tick_params(axis='both', labelsize=18)

ax2 = fig.add_subplot(gs[2:4,0:3])
im, _ = SPF.check_plot(data=sig_A, row_labels=pick_chans_A, col_labels=pick_chans_A,
                       ax=ax2, cmap='Blues', vmin=vmin_A, vmax=vmax_A)
SPF.check_annotate(im, valfmt=matplotlib.ticker.FuncFormatter(func), size=16)
ax2.set_xlabel('Signal correlation (A)', fontsize=22)
ax2.tick_params(axis='both', labelsize=18)

ax3 = fig.add_subplot(gs[4:6,0:3])
im, _ = SPF.check_plot(data=compare_A, row_labels=pick_chans_A, col_labels=pick_chans_A,
                       ax=ax3, cmap='Reds', vmin=np.min(compare_A), vmax=np.max(compare_A))
SPF.check_annotate(im, valfmt=matplotlib.ticker.FuncFormatter(func), size=16)
ax3.set_xlabel('Substraction (A)', fontsize=22)
ax3.tick_params(axis='both', labelsize=18)

del w_A, sig_A, compare_A, pick_chans_A, vmin_A, vmax_A

ax4 = fig.add_subplot(gs[0:2,3:6])
im, _ = SPF.check_plot(data=w_B, row_labels=pick_chans_B, col_labels=pick_chans_B,
                       ax=ax4, cmap='Blues', vmin=vmin_B, vmax=vmax_B)
SPF.check_annotate(im, valfmt=matplotlib.ticker.FuncFormatter(func), size=16)
ax4.set_xlabel('Background correlation (B)', fontsize=22)
ax4.tick_params(axis='both', labelsize=18)

ax5 = fig.add_subplot(gs[2:4,3:6])
im, _ = SPF.check_plot(data=sig_B, row_labels=pick_chans_B, col_labels=pick_chans_B,
                       ax=ax5, cmap='Blues', vmin=vmin_B, vmax=vmax_B)
SPF.check_annotate(im, valfmt=matplotlib.ticker.FuncFormatter(func), size=16)
ax5.set_xlabel('Signal correlation (B)', fontsize=22)
ax5.tick_params(axis='both', labelsize=18)

ax6 = fig.add_subplot(gs[4:6,3:6])
im, _ = SPF.check_plot(data=compare_B, row_labels=pick_chans_B, col_labels=pick_chans_B,
                       ax=ax6, cmap='Reds', vmin=np.min(compare_B), vmax=np.max(compare_B))
SPF.check_annotate(im, valfmt=matplotlib.ticker.FuncFormatter(func), size=16)
ax6.set_xlabel('Substraction (B)', fontsize=22)
ax6.tick_params(axis='both', labelsize=18)

del w_B, sig_B, compare_B, pick_chans_B, vmin_B, vmax_B

ax7 = fig.add_subplot(gs[0:2,6:9])
im, _ = SPF.check_plot(data=w_C, row_labels=pick_chans_C, col_labels=pick_chans_C,
                       ax=ax7, cmap='Blues', vmin=vmin_C, vmax=vmax_C)
SPF.check_annotate(im, valfmt=matplotlib.ticker.FuncFormatter(func), size=16)
ax7.set_xlabel('Background correlation (C)', fontsize=22)
ax7.tick_params(axis='both', labelsize=18)

ax8 = fig.add_subplot(gs[2:4,6:9])
im, _ = SPF.check_plot(data=sig_C, row_labels=pick_chans_C, col_labels=pick_chans_C,
                       ax=ax8, cmap='Blues', vmin=vmin_C, vmax=vmax_C)
SPF.check_annotate(im, valfmt=matplotlib.ticker.FuncFormatter(func), size=16)
ax8.set_xlabel('Signal correlation (C)', fontsize=22)
ax8.tick_params(axis='both', labelsize=18)

ax9 = fig.add_subplot(gs[4:6,6:9])
im, _ = SPF.check_plot(data=compare_C, row_labels=pick_chans_C, col_labels=pick_chans_C,
                       ax=ax9, cmap='Reds', vmin=np.min(compare_C), vmax=np.max(compare_C))
SPF.check_annotate(im, valfmt=matplotlib.ticker.FuncFormatter(func), size=16)
ax9.set_xlabel('Substraction (C)', fontsize=22)
ax9.tick_params(axis='both', labelsize=18)

del w_C, sig_C, compare_C, pick_chans_C, vmin_C, vmax_C

ax10 = fig.add_subplot(gs[0:2,9:])
im, _ = SPF.check_plot(data=w_D, row_labels=pick_chans_D, col_labels=pick_chans_D,
                       ax=ax10, cmap='Blues', vmin=vmin_D, vmax=vmax_D)
SPF.check_annotate(im, valfmt=matplotlib.ticker.FuncFormatter(func), size=16)
ax10.set_xlabel('Background correlation (D)', fontsize=22)
ax10.tick_params(axis='both', labelsize=18)

ax11 = fig.add_subplot(gs[2:4,9:])
im, _ = SPF.check_plot(data=sig_D, row_labels=pick_chans_D, col_labels=pick_chans_D,
                       ax=ax11, cmap='Blues', vmin=vmin_D, vmax=vmax_D)
SPF.check_annotate(im, valfmt=matplotlib.ticker.FuncFormatter(func), size=16)
ax11.set_xlabel('Signal correlation (D)', fontsize=22)
ax11.tick_params(axis='both', labelsize=18)

ax12 = fig.add_subplot(gs[4:6,9:])
im, _ = SPF.check_plot(data=compare_D, row_labels=pick_chans_D, col_labels=pick_chans_D,
                       ax=ax12, cmap='Reds', vmin=np.min(compare_D), vmax=np.max(compare_D))
SPF.check_annotate(im, valfmt=matplotlib.ticker.FuncFormatter(func), size=16)
ax12.set_xlabel('Substraction (D)', fontsize=22)
ax12.tick_params(axis='both', labelsize=18)

del w_D, sig_D, compare_D, pick_chans_D, vmin_D, vmax_D

plt.show()

# save figure
fig.subplots_adjust(top=0.949, bottom=0.035, left=0.049, right=0.990, 
                    hspace=1.000, wspace=1.000)
plt.savefig(r'F:\SSVEP\figures\weisiwen\inter-chan-corr_OZ.png', dpi=600)


#%% Fig 4-1: Line chart (Bias of estimation) (MLR)
# load data & reform
length = 300000

w1_w1 = ['w1-w1' for i in range(2*length)]
w1_w2 = ['w1-w2' for i in range(2*length)]
w1_w3 = ['w1-w3' for i in range(2*length)]
poz = ['POZ' for i in range(length)]
oz = ['OZ' for i in range(length)]
channel_w1 = poz + oz + poz + oz + poz + oz
type_w1 = w1_w1 + w1_w2 + w1_w3 
del w1_w1, w1_w2, w1_w3

w2_w2 = ['w2-w2' for i in range(2*length)]
w2_w3 = ['w2-w3' for i in range(2*length)]
channel_w2 = poz + oz + poz + oz
type_w2 = w2_w2 + w2_w3
del w2_w2, w2_w3

w3_w3 = ['w3-w3' for i in range(2*length)]
channel_w3 = poz + oz
type_w3 = w3_w3
del w3_w3, length

# Group A
poz = io.loadmat(r'F:\SSVEP\dataset\preprocessed_data\weisiwen\14_18__POZ\MLR_model.mat')
oz = io.loadmat(r'F:\SSVEP\dataset\preprocessed_data\weisiwen\14_18__OZ\MLR_model.mat')

w1w1_p = poz['w1_ex_w1'].flatten()
w1w1_o = oz['w1_ex_w1'].flatten()
Abias_w1 = np.hstack((w1w1_p, w1w1_o))
del w1w1_p, w1w1_o

w1w2_p = poz['w1_ex_w2'].flatten()
w1w2_o = oz['w1_ex_w2'].flatten()
Abias_w1 = np.hstack((Abias_w1, w1w2_p, w1w2_o))
del w1w2_p, w1w2_o

w1w3_p = poz['w1_ex_w3'].flatten()
w1w3_o = oz['w1_ex_w3'].flatten()
Abias_w1 = np.hstack((Abias_w1, w1w3_p, w1w3_o))
del w1w3_p, w1w3_o

w2w2_p = poz['w2_ex_w2'].flatten()
w2w2_o = oz['w2_ex_w2'].flatten()
Abias_w2 = np.hstack((w2w2_p, w2w2_o))
del w2w2_p, w2w2_o

w2w3_p = poz['w2_ex_w3'].flatten()
w2w3_o = oz['w2_ex_w3'].flatten()
Abias_w2 = np.hstack((Abias_w2, w2w3_p, w2w3_o))
del w2w3_p, w2w3_o

w3w3_p = poz['w3_ex_w3'].flatten()
w3w3_o = oz['w3_ex_w3'].flatten()
Abias_w3 = np.hstack((w3w3_p, w3w3_o))
del w3w3_p, w3w3_o
del poz, oz

bias_w1_A = pd.DataFrame({'bias':Abias_w1, 'type':type_w1, 'channel':channel_w1})
del Abias_w1
bias_w2_A = pd.DataFrame({'bias':Abias_w2, 'type':type_w2, 'channel':channel_w2})
del Abias_w2
bias_w3_A = pd.DataFrame({'bias':Abias_w3, 'type':type_w3, 'channel':channel_w3})
del Abias_w3

# Group B
poz = io.loadmat(r'F:\SSVEP\dataset\preprocessed_data\weisiwen\19_23__POZ\MLR_model.mat')
oz = io.loadmat(r'F:\SSVEP\dataset\preprocessed_data\weisiwen\19_23__OZ\MLR_model.mat')

w1w1_p = poz['w1_ex_w1'].flatten()
w1w1_o = oz['w1_ex_w1'].flatten()
Bbias_w1 = np.hstack((w1w1_p, w1w1_o))
del w1w1_p, w1w1_o

w1w2_p = poz['w1_ex_w2'].flatten()
w1w2_o = oz['w1_ex_w2'].flatten()
Bbias_w1 = np.hstack((Bbias_w1, w1w2_p, w1w2_o))
del w1w2_p, w1w2_o

w1w3_p = poz['w1_ex_w3'].flatten()
w1w3_o = oz['w1_ex_w3'].flatten()
Bbias_w1 = np.hstack((Bbias_w1, w1w3_p, w1w3_o))
del w1w3_p, w1w3_o

w2w2_p = poz['w2_ex_w2'].flatten()
w2w2_o = oz['w2_ex_w2'].flatten()
Bbias_w2 = np.hstack((w2w2_p, w2w2_o))
del w2w2_p, w2w2_o

w2w3_p = poz['w2_ex_w3'].flatten()
w2w3_o = oz['w2_ex_w3'].flatten()
Bbias_w2 = np.hstack((Bbias_w2, w2w3_p, w2w3_o))
del w2w3_p, w2w3_o

w3w3_p = poz['w3_ex_w3'].flatten()
w3w3_o = oz['w3_ex_w3'].flatten()
Bbias_w3 = np.hstack((w3w3_p, w3w3_o))
del w3w3_p, w3w3_o
del poz, oz

bias_w1_B = pd.DataFrame({'bias':Bbias_w1, 'type':type_w1, 'channel':channel_w1})
del Bbias_w1
bias_w2_B = pd.DataFrame({'bias':Bbias_w2, 'type':type_w2, 'channel':channel_w2})
del Bbias_w2
bias_w3_B = pd.DataFrame({'bias':Bbias_w3, 'type':type_w3, 'channel':channel_w3})
del Bbias_w3

# Group C
poz = io.loadmat(r'F:\SSVEP\dataset\preprocessed_data\weisiwen\24_28__POZ\MLR_model.mat')
oz = io.loadmat(r'F:\SSVEP\dataset\preprocessed_data\weisiwen\24_28__OZ\MLR_model.mat')

w1w1_p = poz['w1_ex_w1'].flatten()
w1w1_o = oz['w1_ex_w1'].flatten()
Cbias_w1 = np.hstack((w1w1_p, w1w1_o))
del w1w1_p, w1w1_o

w1w2_p = poz['w1_ex_w2'].flatten()
w1w2_o = oz['w1_ex_w2'].flatten()
Cbias_w1 = np.hstack((Cbias_w1, w1w2_p, w1w2_o))
del w1w2_p, w1w2_o

w1w3_p = poz['w1_ex_w3'].flatten()
w1w3_o = oz['w1_ex_w3'].flatten()
Cbias_w1 = np.hstack((Cbias_w1, w1w3_p, w1w3_o))
del w1w3_p, w1w3_o

w2w2_p = poz['w2_ex_w2'].flatten()
w2w2_o = oz['w2_ex_w2'].flatten()
Cbias_w2 = np.hstack((w2w2_p, w2w2_o))
del w2w2_p, w2w2_o

w2w3_p = poz['w2_ex_w3'].flatten()
w2w3_o = oz['w2_ex_w3'].flatten()
Cbias_w2 = np.hstack((Cbias_w2, w2w3_p, w2w3_o))
del w2w3_p, w2w3_o

w3w3_p = poz['w3_ex_w3'].flatten()
w3w3_o = oz['w3_ex_w3'].flatten()
Cbias_w3 = np.hstack((w3w3_p, w3w3_o))
del w3w3_p, w3w3_o
del poz, oz

bias_w1_C = pd.DataFrame({'bias':Cbias_w1, 'type':type_w1, 'channel':channel_w1})
del Cbias_w1
bias_w2_C = pd.DataFrame({'bias':Cbias_w2, 'type':type_w2, 'channel':channel_w2})
del Cbias_w2
bias_w3_C = pd.DataFrame({'bias':Cbias_w3, 'type':type_w3, 'channel':channel_w3})
del Cbias_w3

# Group D
poz = io.loadmat(r'F:\SSVEP\dataset\preprocessed_data\weisiwen\34-35-36-43-44__POZ\MLR_model.mat')
oz = io.loadmat(r'F:\SSVEP\dataset\preprocessed_data\weisiwen\34-35-36-43-44__OZ\MLR_model.mat')

w1w1_p = poz['w1_ex_w1'].flatten()
w1w1_o = oz['w1_ex_w1'].flatten()
Dbias_w1 = np.hstack((w1w1_p, w1w1_o))
del w1w1_p, w1w1_o

w1w2_p = poz['w1_ex_w2'].flatten()
w1w2_o = oz['w1_ex_w2'].flatten()
Dbias_w1 = np.hstack((Dbias_w1, w1w2_p, w1w2_o))
del w1w2_p, w1w2_o

w1w3_p = poz['w1_ex_w3'].flatten()
w1w3_o = oz['w1_ex_w3'].flatten()
Dbias_w1 = np.hstack((Dbias_w1, w1w3_p, w1w3_o))
del w1w3_p, w1w3_o

w2w2_p = poz['w2_ex_w2'].flatten()
w2w2_o = oz['w2_ex_w2'].flatten()
Dbias_w2 = np.hstack((w2w2_p, w2w2_o))
del w2w2_p, w2w2_o

w2w3_p = poz['w2_ex_w3'].flatten()
w2w3_o = oz['w2_ex_w3'].flatten()
Dbias_w2 = np.hstack((Dbias_w2, w2w3_p, w2w3_o))
del w2w3_p, w2w3_o

w3w3_p = poz['w3_ex_w3'].flatten()
w3w3_o = oz['w3_ex_w3'].flatten()
Dbias_w3 = np.hstack((w3w3_p, w3w3_o))
del w3w3_p, w3w3_o
del poz, oz

bias_w1_D = pd.DataFrame({'bias':Dbias_w1, 'type':type_w1, 'channel':channel_w1})
del Dbias_w1, type_w1, channel_w1
bias_w2_D = pd.DataFrame({'bias':Dbias_w2, 'type':type_w2, 'channel':channel_w2})
del Dbias_w2, type_w2, channel_w2
bias_w3_D = pd.DataFrame({'bias':Dbias_w3, 'type':type_w3, 'channel':channel_w3})
del Dbias_w3, type_w3, channel_w3

# plot
fig = plt.figure(figsize=(16,14))

gs = GridSpec(8,9, figure=fig)

sns.set(style='whitegrid')

ax1 = fig.add_subplot(gs[0:2, 0:3])
ax1.tick_params(axis='both', labelsize=18)
ax1 = sns.pointplot(x='type', y='bias', hue='channel', data=bias_w1_A, dodge=True,
                    markers=['o', 'x'], linestyles=['-', '--'], palette='tab10',
                    ci=95, capsize=.2)
ax1.set_xlabel('')
ax1.set_ylabel('A', fontsize=18)
ax1.legend(loc='upper left', fontsize=16)
del bias_w1_A

ax2 = fig.add_subplot(gs[0:2, 3:6])
ax2.tick_params(axis='both', labelsize=18)
ax2 = sns.pointplot(x='type', y='bias', hue='channel', data=bias_w2_A, dodge=True,
                    markers=['o', 'x'], linestyles=['-', '--'], palette='tab10',
                    ci=95, capsize=.2)
ax2.set_xlabel('')
ax2.set_ylabel('')
ax2.legend(loc='upper left', fontsize=16)
del bias_w2_A

ax3 = fig.add_subplot(gs[0:2, 6:])
ax3.tick_params(axis='both', labelsize=18)
ax3 = sns.pointplot(x='type', y='bias', hue='channel', data=bias_w3_A, dodge=True,
                    markers=['o', 'x'], linestyles=['-', '--'], palette='tab10',
                    ci=95, capsize=.2)
ax3.set_xlabel('')
ax3.set_ylabel('')
ax3.legend(loc='upper left', fontsize=16)
del bias_w3_A

ax4 = fig.add_subplot(gs[2:4, 0:3])
ax4.tick_params(axis='both', labelsize=18)
ax4 = sns.pointplot(x='type', y='bias', hue='channel', data=bias_w1_B, dodge=True,
                    markers=['o', 'x'], linestyles=['-', '--'], palette='tab10',
                    ci=95, capsize=.2)
ax4.set_xlabel('')
ax4.set_ylabel('B', fontsize=18)
ax4.legend(loc='upper left', fontsize=16)
del bias_w1_B

ax5 = fig.add_subplot(gs[2:4, 3:6])
ax5.tick_params(axis='both', labelsize=18)
ax5 = sns.pointplot(x='type', y='bias', hue='channel', data=bias_w2_B, dodge=True,
                    markers=['o', 'x'], linestyles=['-', '--'], palette='tab10',
                    ci=95, capsize=.2)
ax5.set_xlabel('')
ax5.set_ylabel('')
ax5.legend(loc='upper left', fontsize=16)
del bias_w2_B

ax6 = fig.add_subplot(gs[2:4, 6:])
ax6.tick_params(axis='both', labelsize=18)
ax6 = sns.pointplot(x='type', y='bias', hue='channel', data=bias_w3_B, dodge=True,
                    markers=['o', 'x'], linestyles=['-', '--'], palette='tab10',
                    ci=95, capsize=.2)
ax6.set_xlabel('')
ax6.set_ylabel('')
ax6.legend(loc='upper left', fontsize=16)
del bias_w3_B

ax7 = fig.add_subplot(gs[4:6, 0:3])
ax7.tick_params(axis='both', labelsize=18)
ax7 = sns.pointplot(x='type', y='bias', hue='channel', data=bias_w1_C, dodge=True,
                    markers=['o', 'x'], linestyles=['-', '--'], palette='tab10',
                    ci=95, capsize=.2)
ax7.set_xlabel('')
ax7.set_ylabel('C', fontsize=18)
ax7.legend(loc='upper left', fontsize=16)
del bias_w1_C

ax8 = fig.add_subplot(gs[4:6, 3:6])
ax8.tick_params(axis='both', labelsize=18)
ax8 = sns.pointplot(x='type', y='bias', hue='channel', data=bias_w2_C, dodge=True,
                    markers=['o', 'x'], linestyles=['-', '--'], palette='tab10',
                    ci=95, capsize=.2)
ax8.set_xlabel('')
ax8.set_ylabel('')
ax8.legend(loc='upper left', fontsize=16)
del bias_w2_C

ax9 = fig.add_subplot(gs[4:6, 6:])
ax9.tick_params(axis='both', labelsize=18)
ax9 = sns.pointplot(x='type', y='bias', hue='channel', data=bias_w3_C, dodge=True,
                    markers=['o', 'x'], linestyles=['-', '--'], palette='tab10',
                    ci=95, capsize=.2)
ax9.set_xlabel('')
ax9.set_ylabel('')
ax9.legend(loc='upper left', fontsize=16)
del bias_w3_C

ax10 = fig.add_subplot(gs[6:, 0:3])
ax10.tick_params(axis='both', labelsize=18)
ax10 = sns.pointplot(x='type', y='bias', hue='channel', data=bias_w1_D, dodge=True,
                    markers=['o', 'x'], linestyles=['-', '--'], palette='tab10',
                    ci=95, capsize=.2)
ax10.set_xlabel('')
ax10.set_ylabel('D', fontsize=18)
ax10.legend(loc='upper left', fontsize=16)
del bias_w1_D

ax11 = fig.add_subplot(gs[6:, 3:6])
ax11.tick_params(axis='both', labelsize=18)
ax11 = sns.pointplot(x='type', y='bias', hue='channel', data=bias_w2_D, dodge=True,
                    markers=['o', 'x'], linestyles=['-', '--'], palette='tab10',
                    ci=95, capsize=.2)
ax11.set_xlabel('')
ax11.set_ylabel('')
ax11.legend(loc='upper left', fontsize=16)
del bias_w2_D

ax12 = fig.add_subplot(gs[6:, 6:])
ax12.tick_params(axis='both', labelsize=18)
ax12 = sns.pointplot(x='type', y='bias', hue='channel', data=bias_w3_D, dodge=True,
                    markers=['o', 'x'], linestyles=['-', '--'], palette='tab10',
                    ci=95, capsize=.2)
ax12.set_xlabel('')
ax12.set_ylabel('')
ax12.legend(loc='upper left', fontsize=16)
del bias_w3_D

fig.tight_layout()
plt.show()

plt.savefig(r'F:\SSVEP\figures\weisiwen\bias_MLR.png', dpi=600)


#%% Fig 4-2: Line chart (Bias of estimation) (IA)
# load data & reform
length = 300000

w1_w1 = ['w1-w1' for i in range(2*length)]
w1_w2 = ['w1-w2' for i in range(2*length)]
w1_w3 = ['w1-w3' for i in range(2*length)]
poz = ['POZ' for i in range(length)]
oz = ['OZ' for i in range(length)]
channel_w1 = poz + oz + poz + oz + poz + oz
type_w1 = w1_w1 + w1_w2 + w1_w3 
del w1_w1, w1_w2, w1_w3

w2_w2 = ['w2-w2' for i in range(2*length)]
w2_w3 = ['w2-w3' for i in range(2*length)]
channel_w2 = poz + oz + poz + oz
type_w2 = w2_w2 + w2_w3
del w2_w2, w2_w3

w3_w3 = ['w3-w3' for i in range(2*length)]
channel_w3 = poz + oz
type_w3 = w3_w3
del w3_w3, length

# Group A
poz = io.loadmat(r'F:\SSVEP\dataset\preprocessed_data\weisiwen\14_18__POZ\inver_array_model.mat')
oz = io.loadmat(r'F:\SSVEP\dataset\preprocessed_data\weisiwen\14_18__OZ\inver_array_model.mat')

w1w1_p = poz['w1_ex_w1'].flatten()
w1w1_o = oz['w1_ex_w1'].flatten()
Abias_w1 = np.hstack((w1w1_p, w1w1_o))
del w1w1_p, w1w1_o

w1w2_p = poz['w1_ex_w2'].flatten()
w1w2_o = oz['w1_ex_w2'].flatten()
Abias_w1 = np.hstack((Abias_w1, w1w2_p, w1w2_o))
del w1w2_p, w1w2_o

w1w3_p = poz['w1_ex_w3'].flatten()
w1w3_o = oz['w1_ex_w3'].flatten()
Abias_w1 = np.hstack((Abias_w1, w1w3_p, w1w3_o))
del w1w3_p, w1w3_o

w2w2_p = poz['w2_ex_w2'].flatten()
w2w2_o = oz['w2_ex_w2'].flatten()
Abias_w2 = np.hstack((w2w2_p, w2w2_o))
del w2w2_p, w2w2_o

w2w3_p = poz['w2_ex_w3'].flatten()
w2w3_o = oz['w2_ex_w3'].flatten()
Abias_w2 = np.hstack((Abias_w2, w2w3_p, w2w3_o))
del w2w3_p, w2w3_o

w3w3_p = poz['w3_ex_w3'].flatten()
w3w3_o = oz['w3_ex_w3'].flatten()
Abias_w3 = np.hstack((w3w3_p, w3w3_o))
del w3w3_p, w3w3_o
del poz, oz

bias_w1_A = pd.DataFrame({'bias':Abias_w1, 'type':type_w1, 'channel':channel_w1})
del Abias_w1
bias_w2_A = pd.DataFrame({'bias':Abias_w2, 'type':type_w2, 'channel':channel_w2})
del Abias_w2
bias_w3_A = pd.DataFrame({'bias':Abias_w3, 'type':type_w3, 'channel':channel_w3})
del Abias_w3

# Group B
poz = io.loadmat(r'F:\SSVEP\dataset\preprocessed_data\weisiwen\19_23__POZ\inver_array_model.mat')
oz = io.loadmat(r'F:\SSVEP\dataset\preprocessed_data\weisiwen\19_23__OZ\inver_array_model.mat')

w1w1_p = poz['w1_ex_w1'].flatten()
w1w1_o = oz['w1_ex_w1'].flatten()
Bbias_w1 = np.hstack((w1w1_p, w1w1_o))
del w1w1_p, w1w1_o

w1w2_p = poz['w1_ex_w2'].flatten()
w1w2_o = oz['w1_ex_w2'].flatten()
Bbias_w1 = np.hstack((Bbias_w1, w1w2_p, w1w2_o))
del w1w2_p, w1w2_o

w1w3_p = poz['w1_ex_w3'].flatten()
w1w3_o = oz['w1_ex_w3'].flatten()
Bbias_w1 = np.hstack((Bbias_w1, w1w3_p, w1w3_o))
del w1w3_p, w1w3_o

w2w2_p = poz['w2_ex_w2'].flatten()
w2w2_o = oz['w2_ex_w2'].flatten()
Bbias_w2 = np.hstack((w2w2_p, w2w2_o))
del w2w2_p, w2w2_o

w2w3_p = poz['w2_ex_w3'].flatten()
w2w3_o = oz['w2_ex_w3'].flatten()
Bbias_w2 = np.hstack((Bbias_w2, w2w3_p, w2w3_o))
del w2w3_p, w2w3_o

w3w3_p = poz['w3_ex_w3'].flatten()
w3w3_o = oz['w3_ex_w3'].flatten()
Bbias_w3 = np.hstack((w3w3_p, w3w3_o))
del w3w3_p, w3w3_o
del poz, oz

bias_w1_B = pd.DataFrame({'bias':Bbias_w1, 'type':type_w1, 'channel':channel_w1})
del Bbias_w1
bias_w2_B = pd.DataFrame({'bias':Bbias_w2, 'type':type_w2, 'channel':channel_w2})
del Bbias_w2
bias_w3_B = pd.DataFrame({'bias':Bbias_w3, 'type':type_w3, 'channel':channel_w3})
del Bbias_w3

# Group C
poz = io.loadmat(r'F:\SSVEP\dataset\preprocessed_data\weisiwen\24_28__POZ\inver_array_model.mat')
oz = io.loadmat(r'F:\SSVEP\dataset\preprocessed_data\weisiwen\24_28__OZ\inver_array_model.mat')

w1w1_p = poz['w1_ex_w1'].flatten()
w1w1_o = oz['w1_ex_w1'].flatten()
Cbias_w1 = np.hstack((w1w1_p, w1w1_o))
del w1w1_p, w1w1_o

w1w2_p = poz['w1_ex_w2'].flatten()
w1w2_o = oz['w1_ex_w2'].flatten()
Cbias_w1 = np.hstack((Cbias_w1, w1w2_p, w1w2_o))
del w1w2_p, w1w2_o

w1w3_p = poz['w1_ex_w3'].flatten()
w1w3_o = oz['w1_ex_w3'].flatten()
Cbias_w1 = np.hstack((Cbias_w1, w1w3_p, w1w3_o))
del w1w3_p, w1w3_o

w2w2_p = poz['w2_ex_w2'].flatten()
w2w2_o = oz['w2_ex_w2'].flatten()
Cbias_w2 = np.hstack((w2w2_p, w2w2_o))
del w2w2_p, w2w2_o

w2w3_p = poz['w2_ex_w3'].flatten()
w2w3_o = oz['w2_ex_w3'].flatten()
Cbias_w2 = np.hstack((Cbias_w2, w2w3_p, w2w3_o))
del w2w3_p, w2w3_o

w3w3_p = poz['w3_ex_w3'].flatten()
w3w3_o = oz['w3_ex_w3'].flatten()
Cbias_w3 = np.hstack((w3w3_p, w3w3_o))
del w3w3_p, w3w3_o
del poz, oz

bias_w1_C = pd.DataFrame({'bias':Cbias_w1, 'type':type_w1, 'channel':channel_w1})
del Cbias_w1
bias_w2_C = pd.DataFrame({'bias':Cbias_w2, 'type':type_w2, 'channel':channel_w2})
del Cbias_w2
bias_w3_C = pd.DataFrame({'bias':Cbias_w3, 'type':type_w3, 'channel':channel_w3})
del Cbias_w3

# Group D
poz = io.loadmat(r'F:\SSVEP\dataset\preprocessed_data\weisiwen\34-35-36-43-44__POZ\inver_array_model.mat')
oz = io.loadmat(r'F:\SSVEP\dataset\preprocessed_data\weisiwen\34-35-36-43-44__OZ\inver_array_model.mat')

w1w1_p = poz['w1_ex_w1'].flatten()
w1w1_o = oz['w1_ex_w1'].flatten()
Dbias_w1 = np.hstack((w1w1_p, w1w1_o))
del w1w1_p, w1w1_o

w1w2_p = poz['w1_ex_w2'].flatten()
w1w2_o = oz['w1_ex_w2'].flatten()
Dbias_w1 = np.hstack((Dbias_w1, w1w2_p, w1w2_o))
del w1w2_p, w1w2_o

w1w3_p = poz['w1_ex_w3'].flatten()
w1w3_o = oz['w1_ex_w3'].flatten()
Dbias_w1 = np.hstack((Dbias_w1, w1w3_p, w1w3_o))
del w1w3_p, w1w3_o

w2w2_p = poz['w2_ex_w2'].flatten()
w2w2_o = oz['w2_ex_w2'].flatten()
Dbias_w2 = np.hstack((w2w2_p, w2w2_o))
del w2w2_p, w2w2_o

w2w3_p = poz['w2_ex_w3'].flatten()
w2w3_o = oz['w2_ex_w3'].flatten()
Dbias_w2 = np.hstack((Dbias_w2, w2w3_p, w2w3_o))
del w2w3_p, w2w3_o

w3w3_p = poz['w3_ex_w3'].flatten()
w3w3_o = oz['w3_ex_w3'].flatten()
Dbias_w3 = np.hstack((w3w3_p, w3w3_o))
del w3w3_p, w3w3_o
del poz, oz

bias_w1_D = pd.DataFrame({'bias':Dbias_w1, 'type':type_w1, 'channel':channel_w1})
del Dbias_w1, type_w1, channel_w1
bias_w2_D = pd.DataFrame({'bias':Dbias_w2, 'type':type_w2, 'channel':channel_w2})
del Dbias_w2, type_w2, channel_w2
bias_w3_D = pd.DataFrame({'bias':Dbias_w3, 'type':type_w3, 'channel':channel_w3})
del Dbias_w3, type_w3, channel_w3

# plot
fig = plt.figure(figsize=(16,14))

gs = GridSpec(8,9, figure=fig)

sns.set(style='whitegrid')

ax1 = fig.add_subplot(gs[0:2, 0:3])
ax1.tick_params(axis='both', labelsize=18)
ax1 = sns.pointplot(x='type', y='bias', hue='channel', data=bias_w1_A, dodge=True,
                    markers=['o', 'x'], linestyles=['-', '--'], palette='Accent',
                    ci=95, capsize=.2)
ax1.set_xlabel('')
ax1.set_ylabel('A', fontsize=18)
ax1.legend(loc='upper left', fontsize=16)
del bias_w1_A

ax2 = fig.add_subplot(gs[0:2, 3:6])
ax2.tick_params(axis='both', labelsize=18)
ax2 = sns.pointplot(x='type', y='bias', hue='channel', data=bias_w2_A, dodge=True,
                    markers=['o', 'x'], linestyles=['-', '--'], palette='Accent',
                    ci=95, capsize=.2)
ax2.set_xlabel('')
ax2.set_ylabel('')
ax2.legend(loc='upper left', fontsize=16)
del bias_w2_A

ax3 = fig.add_subplot(gs[0:2, 6:])
ax3.tick_params(axis='both', labelsize=18)
ax3 = sns.pointplot(x='type', y='bias', hue='channel', data=bias_w3_A, dodge=True,
                    markers=['o', 'x'], linestyles=['-', '--'], palette='Accent',
                    ci=95, capsize=.2)
ax3.set_xlabel('')
ax3.set_ylabel('')
ax3.legend(loc='upper left', fontsize=16)
del bias_w3_A

ax4 = fig.add_subplot(gs[2:4, 0:3])
ax4.tick_params(axis='both', labelsize=18)
ax4 = sns.pointplot(x='type', y='bias', hue='channel', data=bias_w1_B, dodge=True,
                    markers=['o', 'x'], linestyles=['-', '--'], palette='Accent',
                    ci=95, capsize=.2)
ax4.set_xlabel('')
ax4.set_ylabel('B', fontsize=18)
ax4.legend(loc='upper left', fontsize=16)
del bias_w1_B

ax5 = fig.add_subplot(gs[2:4, 3:6])
ax5.tick_params(axis='both', labelsize=18)
ax5 = sns.pointplot(x='type', y='bias', hue='channel', data=bias_w2_B, dodge=True,
                    markers=['o', 'x'], linestyles=['-', '--'], palette='Accent',
                    ci=95, capsize=.2)
ax5.set_xlabel('')
ax5.set_ylabel('')
ax5.legend(loc='upper left', fontsize=16)
del bias_w2_B

ax6 = fig.add_subplot(gs[2:4, 6:])
ax6.tick_params(axis='both', labelsize=18)
ax6 = sns.pointplot(x='type', y='bias', hue='channel', data=bias_w3_B, dodge=True,
                    markers=['o', 'x'], linestyles=['-', '--'], palette='Accent',
                    ci=95, capsize=.2)
ax6.set_xlabel('')
ax6.set_ylabel('')
ax6.legend(loc='upper left', fontsize=16)
del bias_w3_B

ax7 = fig.add_subplot(gs[4:6, 0:3])
ax7.tick_params(axis='both', labelsize=18)
ax7 = sns.pointplot(x='type', y='bias', hue='channel', data=bias_w1_C, dodge=True,
                    markers=['o', 'x'], linestyles=['-', '--'], palette='Accent',
                    ci=95, capsize=.2)
ax7.set_xlabel('')
ax7.set_ylabel('C', fontsize=18)
ax7.legend(loc='upper left', fontsize=16)
del bias_w1_C

ax8 = fig.add_subplot(gs[4:6, 3:6])
ax8.tick_params(axis='both', labelsize=18)
ax8 = sns.pointplot(x='type', y='bias', hue='channel', data=bias_w2_C, dodge=True,
                    markers=['o', 'x'], linestyles=['-', '--'], palette='Accent',
                    ci=95, capsize=.2)
ax8.set_xlabel('')
ax8.set_ylabel('')
ax8.legend(loc='upper left', fontsize=16)
del bias_w2_C

ax9 = fig.add_subplot(gs[4:6, 6:])
ax9.tick_params(axis='both', labelsize=18)
ax9 = sns.pointplot(x='type', y='bias', hue='channel', data=bias_w3_C, dodge=True,
                    markers=['o', 'x'], linestyles=['-', '--'], palette='Accent',
                    ci=95, capsize=.2)
ax9.set_xlabel('')
ax9.set_ylabel('')
ax9.legend(loc='upper left', fontsize=16)
del bias_w3_C

ax10 = fig.add_subplot(gs[6:, 0:3])
ax10.tick_params(axis='both', labelsize=18)
ax10 = sns.pointplot(x='type', y='bias', hue='channel', data=bias_w1_D, dodge=True,
                    markers=['o', 'x'], linestyles=['-', '--'], palette='Accent',
                    ci=95, capsize=.2)
ax10.set_xlabel('')
ax10.set_ylabel('D', fontsize=18)
ax10.legend(loc='upper left', fontsize=16)
del bias_w1_D

ax11 = fig.add_subplot(gs[6:, 3:6])
ax11.tick_params(axis='both', labelsize=18)
ax11 = sns.pointplot(x='type', y='bias', hue='channel', data=bias_w2_D, dodge=True,
                    markers=['o', 'x'], linestyles=['-', '--'], palette='Accent',
                    ci=95, capsize=.2)
ax11.set_xlabel('')
ax11.set_ylabel('')
ax11.legend(loc='upper left', fontsize=16)
del bias_w2_D

ax12 = fig.add_subplot(gs[6:, 6:])
ax12.tick_params(axis='both', labelsize=18)
ax12 = sns.pointplot(x='type', y='bias', hue='channel', data=bias_w3_D, dodge=True,
                    markers=['o', 'x'], linestyles=['-', '--'], palette='Accent',
                    ci=95, capsize=.2)
ax12.set_xlabel('')
ax12.set_ylabel('')
ax12.legend(loc='upper left', fontsize=16)
del bias_w3_D

fig.tight_layout()
plt.show()

plt.savefig(r'F:\SSVEP\figures\weisiwen\bias_IA.png', dpi=600)


#%% Fig 5: Barplot (Cosine similarity (Normal & Tanimoto))
# load data & reform
A = ['A' for i in range(1800)]
B = ['B' for i in range(1800)]
C = ['C' for i in range(1800)]
D = ['D' for i in range(1800)]
group = A + B + C + D
del A, B, C, D

w1w1 = ['w1-w1' for i in range(300)]
w1w2 = ['w1-w2' for i in range(300)]
w1w3 = ['w1-w3' for i in range(300)]
w2w2 = ['w2-w2' for i in range(300)]
w2w3 = ['w2-w3' for i in range(300)]
w3w3 = ['w3-w3' for i in range(300)]
temp = w1w1 + w1w2 + w1w3 + w2w2 + w2w3 + w3w3
del w1w1, w1w2, w1w3, w2w2, w2w3, w3w3
ttype = temp + temp + temp + temp
del temp

# Group A
cossim = io.loadmat(r'F:\SSVEP\dataset\preprocessed_data\weisiwen\14_18__OZ\cos_sim_ia.mat')
w1w1n = cossim['w1_w1_nsim'].flatten()
w1w2n = cossim['w1_w2_nsim'].flatten()
w1w3n = cossim['w1_w3_nsim'].flatten()
w2w2n = cossim['w2_w2_nsim'].flatten()
w2w3n = cossim['w2_w3_nsim'].flatten()
w3w3n = cossim['w3_w3_nsim'].flatten()
sim = np.hstack((w1w1n, w1w2n, w1w3n, w2w2n, w2w3n, w3w3n))
del w1w1n, w1w2n, w1w3n, w2w2n, w2w3n, w3w3n

w1w1t = cossim['w1_w1_tsim'].flatten()
w1w2t = cossim['w1_w2_tsim'].flatten()
w1w3t = cossim['w1_w3_tsim'].flatten()
w2w2t = cossim['w2_w2_tsim'].flatten()
w2w3t = cossim['w2_w3_tsim'].flatten()
w3w3t = cossim['w3_w3_tsim'].flatten()
tanimoto = np.hstack((w1w1t, w1w2t, w1w3t, w2w2t, w2w3t, w3w3t))
del w1w1t, w1w2t, w1w3t, w2w2t, w2w3t, w3w3t
del cossim

# Group B
cossim = io.loadmat(r'F:\SSVEP\dataset\preprocessed_data\weisiwen\19_23__OZ\cos_sim_ia.mat')
w1w1n = cossim['w1_w1_nsim'].flatten()
w1w2n = cossim['w1_w2_nsim'].flatten()
w1w3n = cossim['w1_w3_nsim'].flatten()
w2w2n = cossim['w2_w2_nsim'].flatten()
w2w3n = cossim['w2_w3_nsim'].flatten()
w3w3n = cossim['w3_w3_nsim'].flatten()
sim = np.hstack((sim, w1w1n, w1w2n, w1w3n, w2w2n, w2w3n, w3w3n))
del w1w1n, w1w2n, w1w3n, w2w2n, w2w3n, w3w3n

w1w1t = cossim['w1_w1_tsim'].flatten()
w1w2t = cossim['w1_w2_tsim'].flatten()
w1w3t = cossim['w1_w3_tsim'].flatten()
w2w2t = cossim['w2_w2_tsim'].flatten()
w2w3t = cossim['w2_w3_tsim'].flatten()
w3w3t = cossim['w3_w3_tsim'].flatten()
tanimoto = np.hstack((tanimoto, w1w1t, w1w2t, w1w3t, w2w2t, w2w3t, w3w3t))
del w1w1t, w1w2t, w1w3t, w2w2t, w2w3t, w3w3t
del cossim

# Group C
cossim = io.loadmat(r'F:\SSVEP\dataset\preprocessed_data\weisiwen\24_28__OZ\cos_sim_ia.mat')
w1w1n = cossim['w1_w1_nsim'].flatten()
w1w2n = cossim['w1_w2_nsim'].flatten()
w1w3n = cossim['w1_w3_nsim'].flatten()
w2w2n = cossim['w2_w2_nsim'].flatten()
w2w3n = cossim['w2_w3_nsim'].flatten()
w3w3n = cossim['w3_w3_nsim'].flatten()
sim = np.hstack((sim, w1w1n, w1w2n, w1w3n, w2w2n, w2w3n, w3w3n))
del w1w1n, w1w2n, w1w3n, w2w2n, w2w3n, w3w3n

w1w1t = cossim['w1_w1_tsim'].flatten()
w1w2t = cossim['w1_w2_tsim'].flatten()
w1w3t = cossim['w1_w3_tsim'].flatten()
w2w2t = cossim['w2_w2_tsim'].flatten()
w2w3t = cossim['w2_w3_tsim'].flatten()
w3w3t = cossim['w3_w3_tsim'].flatten()
tanimoto = np.hstack((tanimoto, w1w1t, w1w2t, w1w3t, w2w2t, w2w3t, w3w3t))
del w1w1t, w1w2t, w1w3t, w2w2t, w2w3t, w3w3t
del cossim

# Group D
cossim = io.loadmat(r'F:\SSVEP\dataset\preprocessed_data\weisiwen\34-35-36-43-44__OZ\cos_sim_ia.mat')
w1w1n = cossim['w1_w1_nsim'].flatten()
w1w2n = cossim['w1_w2_nsim'].flatten()
w1w3n = cossim['w1_w3_nsim'].flatten()
w2w2n = cossim['w2_w2_nsim'].flatten()
w2w3n = cossim['w2_w3_nsim'].flatten()
w3w3n = cossim['w3_w3_nsim'].flatten()
sim = np.hstack((sim, w1w1n, w1w2n, w1w3n, w2w2n, w2w3n, w3w3n))
del w1w1n, w1w2n, w1w3n, w2w2n, w2w3n, w3w3n

w1w1t = cossim['w1_w1_tsim'].flatten()
w1w2t = cossim['w1_w2_tsim'].flatten()
w1w3t = cossim['w1_w3_tsim'].flatten()
w2w2t = cossim['w2_w2_tsim'].flatten()
w2w3t = cossim['w2_w3_tsim'].flatten()
w3w3t = cossim['w3_w3_tsim'].flatten()
tanimoto = np.hstack((tanimoto, w1w1t, w1w2t, w1w3t, w2w2t, w2w3t, w3w3t))
del w1w1t, w1w2t, w1w3t, w2w2t, w2w3t, w3w3t
del cossim

cossim_n = pd.DataFrame({'sim':sim, 'type':ttype, 'group':group})
del sim
cossim_t = pd.DataFrame({'tanimoto':tanimoto, 'type':ttype, 'group':group})
del tanimoto

# plot
fig = plt.figure(figsize=(18,12))

gs = GridSpec(6,9, figure=fig)

color = ['#FB8072', '#80B1D3', '#FDB642', '#B3DE69']
brynhildr = sns.color_palette(color)
sns.set(style='whitegrid')

ax1 = fig.add_subplot(gs[0:3, :])
ax1.tick_params(axis='both', labelsize=18)
ax1 = sns.barplot(x='type', y='sim', hue='group', data=cossim_n, ci='sd',
                  palette=brynhildr, saturation=.75)
ax1.set_xlabel('')
ax1.set_ylabel('Normal cosine similarity', fontsize=18)
ax1.legend(loc='best', fontsize=16)
del cossim_n

ax2 = fig.add_subplot(gs[3:, :])
ax2.tick_params(axis='both', labelsize=18)
ax2 = sns.barplot(x='type', y='tanimoto', hue='group', data=cossim_t, ci='sd',
                  palette=brynhildr, saturation=.75)
ax2.set_xlabel('')
ax2.set_ylabel('Tanimoto coefficient', fontsize=18)
ax2.legend(loc='best', fontsize=16)
del cossim_t
del brynhildr, color, group, ttype

fig.tight_layout()
plt.show()

plt.savefig(r'F:\SSVEP\figures\weisiwen\cossim_oz_ia.png', dpi=600)


#%% Fig 6: Waveform of signal (Origin & estimate) (With zoom-in effect)
# load data
data = io.loadmat(r'F:\SSVEP\dataset\preprocessed_data\weisiwen\14_18__OZ\model_data.mat')
sig = data['sig_o']
del data

data = io.loadmat(r'F:\SSVEP\dataset\preprocessed_data\weisiwen\14_18__OZ\MLR_model.mat')
w1s = data['w1_es_s']
w1x = data['w1_ex_s']

w2s = data['w2_es_s']
w2x = data['w2_ex_s']

w3s = data['w3_es_s']
w3x = data['w3_ex_s']
del data, data_path

# plot
fig = plt.figure(figsize=(16,9))
sns.set(style='whitegrid')
gs = GridSpec(3,4, figure=fig)

ax1 = fig.add_subplot(gs[0:1, :])
ax1.tick_params(axis='both', labelsize=14)
ax1.set_title('15Hz SSVEP comparison (origin & estimation): OZ', fontsize=20)
ax1.set_xlim([-100,3900])
ax1.set_xlabel('Time/ms', fontsize=16)
ax1.set_ylabel('Amplitude/μV', fontsize=16)
ax1.plot(np.mean(sig[2,:,:], axis=0), label='Origin', color='tab:blue', linewidth=1.5)
ax1.plot(np.mean(w1s[2,:,:], axis=0), label='w1', color='tab:orange', linewidth=1)
ax1.plot(np.mean(w2s[2,:,:], axis=0), label='w2', color='tab:green', linewidth=1)
ax1.plot(np.mean(w3s[2,:,:], axis=0), label='w3', color='tab:red', linewidth=1)
ax1.legend(loc='lower right', fontsize=14)

ax2 = fig.add_subplot(gs[1:2, 0:2])
ax2.tick_params(axis='both', labelsize=14)
ax2.set_xlim(100,1300)
SPF.zoom_effect01(ax2,ax1,200,1200)
ax2.set_xlabel('Time/ms', fontsize=16)
ax2.set_ylabel('Amplitude/μV', fontsize=16)
ax2.plot(np.mean(sig[2,:,0:1300], axis=0), label='Origin', color='tab:blue', linewidth=1.5)
ax2.plot(np.mean(w1s[2,:,0:1300], axis=0), label='w1', color='tab:orange', linewidth=1)
ax2.plot(np.mean(w2s[2,:,0:1300], axis=0), label='w2', color='tab:green', linewidth=1)
ax2.plot(np.mean(w3s[2,:,0:1300], axis=0), label='w3', color='tab:red', linewidth=1)
#ax2.legend(loc='lower right', fontsize=14)
del w1s, w2s, w3s

ax3 = fig.add_subplot(gs[2:, :])
ax3.tick_params(axis='both', labelsize=14)
ax3.set_title('15Hz SSVEP comparison (origin & extraction): OZ', fontsize=20)
ax3.set_xlim([-100,3900])
ax3.set_xlabel('Time/ms', fontsize=16)
ax3.set_ylabel('Amplitude/μV', fontsize=16)
ax3.plot(np.mean(sig[2,:,:], axis=0), label='Origin', color='tab:blue', linewidth=1.5)
ax3.plot(np.mean(w1x[2,:,:], axis=0), label='w1', color='tab:orange', linewidth=1)
ax3.plot(np.mean(w2x[2,:,:], axis=0), label='w2', color='tab:green', linewidth=1)
ax3.plot(np.mean(w3x[2,:,:], axis=0), label='w3', color='tab:red', linewidth=1)
ax3.legend(loc='lower right', fontsize=14)

ax4 = fig.add_subplot(gs[1:2, 2:])
ax4.tick_params(axis='both', labelsize=14)
ax4.set_xlim(100,1300)
SPF.zoom_effect01(ax4,ax3,200,1200)
ax4.set_xlabel('Time/ms', fontsize=16)
ax4.set_ylabel('Amplitude/μV', fontsize=16)
ax4.plot(np.mean(sig[2,:,0:1300], axis=0), label='Origin', color='tab:blue', linewidth=1.5)
ax4.plot(np.mean(w1x[2,:,0:1300], axis=0), label='w1', color='tab:orange', linewidth=1)
ax4.plot(np.mean(w2x[2,:,0:1300], axis=0), label='w2', color='tab:green', linewidth=1)
ax4.plot(np.mean(w3x[2,:,0:1300], axis=0), label='w3', color='tab:red', linewidth=1)
del w1x, w2x, w3x, sig, edge

fig.subplots_adjust(top=0.950, bottom=0.070, left=0.055, right=0.990,
                    hspace=0.400, wspace=0.500)
plt.savefig(r'F:\SSVEP\figures\weisiwen\14_18__OZ_mlr_15hz.png', dpi=600)


#%% Fig 7: Power of spectrum density (Origin & extract)
# load data
data = io.loadmat(r'F:\SSVEP\dataset\preprocessed_data\weisiwen\34-35-36-43-44__OZ\psd_mlr.mat')
fs = data['fs'][1,1,:]
siga = data['sig']
w1a = data['w1']
w2a = data['w2']
w3a = data['w3']
del data

data = io.loadmat(r'F:\SSVEP\dataset\preprocessed_data\weisiwen\34-35-36-43-44__POZ\psd_mlr.mat')
sigb = data['sig']
w1b = data['w1']
w2b = data['w2']
w3b = data['w3']
del data

sfreq=1000

# plot
fig = plt.figure(figsize=(15,12))
gs = GridSpec(6,6, figure=fig)
sns.set(style='whitegrid')

ax1 = fig.add_subplot(gs[0:2, 0:3])
ax1.tick_params(axis='both', labelsize=14)
ax1.set_title('8Hz SSVEP power spectrum density (OZ)', fontsize=18)
ax1.set_xlabel('Frequency/Hz', fontsize=16)
ax1.set_ylabel(r'$μV^2\ /Hz\ (dB)$', fontsize=16)
ax1.plot(fs, np.mean(siga[0,:,:], axis=0), label='Origin', color='tab:blue', linewidth=1.5)
ax1.plot(fs, np.mean(w1a[0,:,:], axis=0), label='w1', color='tab:orange', linewidth=1)
ax1.plot(fs, np.mean(w2a[0,:,:], axis=0), label='w2', color='tab:green', linewidth=1)
ax1.plot(fs, np.mean(w3a[0,:,:], axis=0), label='w3', color='tab:red', linewidth=1)
ax1.legend(loc='upper right', fontsize=14)

ax2 = fig.add_subplot(gs[0:2, 3:])
ax2.tick_params(axis='both', labelsize=14)
ax2.set_title('8Hz SSVEP power spectrum density (POZ)', fontsize=18)
ax2.set_xlabel('Frequency/Hz', fontsize=16)
ax2.set_ylabel(r'$μV^2\ /Hz\ (dB)$', fontsize=16)
ax2.plot(fs, np.mean(sigb[0,:,:], axis=0), label='Origin', color='tab:blue', linewidth=1.5)
ax2.plot(fs, np.mean(w1b[0,:,:], axis=0), label='w1', color='tab:orange', linewidth=1)
ax2.plot(fs, np.mean(w2b[0,:,:], axis=0), label='w2', color='tab:green', linewidth=1)
ax2.plot(fs, np.mean(w3b[0,:,:], axis=0), label='w3', color='tab:red', linewidth=1)
ax2.legend(loc='upper right', fontsize=14)

ax3 = fig.add_subplot(gs[2:4, 0:3])
ax3.tick_params(axis='both', labelsize=14)
ax3.set_title('10Hz SSVEP power spectrum density (OZ)', fontsize=18)
ax3.set_xlabel('Frequency/Hz', fontsize=16)
ax3.set_ylabel(r'$μV^2\ /Hz\ (dB)$', fontsize=16)
ax3.plot(fs, np.mean(siga[1,:,:], axis=0), label='Origin', color='tab:blue', linewidth=1.5)
ax3.plot(fs, np.mean(w1a[1,:,:], axis=0), label='w1', color='tab:orange', linewidth=1)
ax3.plot(fs, np.mean(w2a[1,:,:], axis=0), label='w2', color='tab:green', linewidth=1)
ax3.plot(fs, np.mean(w3a[1,:,:], axis=0), label='w3', color='tab:red', linewidth=1)
ax3.legend(loc='upper right', fontsize=14)

ax4 = fig.add_subplot(gs[2:4, 3:])
ax4.tick_params(axis='both', labelsize=14)
ax4.set_title('10Hz SSVEP power spectrum density (POZ)', fontsize=18)
ax4.set_xlabel('Frequency/Hz', fontsize=16)
ax4.set_ylabel(r'$μV^2\ /Hz\ (dB)$', fontsize=16)
ax4.plot(fs, np.mean(sigb[1,:,:], axis=0), label='Origin', color='tab:blue', linewidth=1.5)
ax4.plot(fs, np.mean(w1b[1,:,:], axis=0), label='w1', color='tab:orange', linewidth=1)
ax4.plot(fs, np.mean(w2b[1,:,:], axis=0), label='w2', color='tab:green', linewidth=1)
ax4.plot(fs, np.mean(w3b[1,:,:], axis=0), label='w3', color='tab:red', linewidth=1)
ax4.legend(loc='upper right', fontsize=14)

ax5 = fig.add_subplot(gs[4:, 0:3])
ax5.tick_params(axis='both', labelsize=14)
ax5.set_title('15Hz SSVEP power spectrum density (OZ)', fontsize=18)
ax5.set_xlabel('Frequency/Hz', fontsize=16)
ax5.set_ylabel(r'$μV^2\ /Hz\ (dB)$', fontsize=16)
ax5.plot(fs, np.mean(siga[2,:,:], axis=0), label='Origin', color='tab:blue', linewidth=1.5)
ax5.plot(fs, np.mean(w1a[2,:,:], axis=0), label='w1', color='tab:orange', linewidth=1)
ax5.plot(fs, np.mean(w2a[2,:,:], axis=0), label='w2', color='tab:green', linewidth=1)
ax5.plot(fs, np.mean(w3a[2,:,:], axis=0), label='w3', color='tab:red', linewidth=1)
ax5.legend(loc='upper right', fontsize=14)

ax6 = fig.add_subplot(gs[4:, 3:])
ax6.tick_params(axis='both', labelsize=14)
ax6.set_title('15Hz SSVEP power spectrum density (POZ)', fontsize=18)
ax6.set_xlabel('Frequency/Hz', fontsize=16)
ax6.set_ylabel(r'$μV^2\ /Hz\ (dB)$', fontsize=16)
ax6.plot(fs, np.mean(sigb[2,:,:], axis=0), label='Origin', color='tab:blue', linewidth=1.5)
ax6.plot(fs, np.mean(w1b[2,:,:], axis=0), label='w1', color='tab:orange', linewidth=1)
ax6.plot(fs, np.mean(w2b[2,:,:], axis=0), label='w2', color='tab:green', linewidth=1)
ax6.plot(fs, np.mean(w3b[2,:,:], axis=0), label='w3', color='tab:red', linewidth=1)
ax6.legend(loc='upper right', fontsize=14)
del siga, sigb, w1a, w1b, w2a, w2b, w3a, w3b, fs, sfreq

# save fig
fig.subplots_adjust(top=0.955, bottom=0.072, left=0.063, right=0.989,
                    hspace=1.000, wspace=1.000)
plt.savefig(r'F:\SSVEP\figures\weisiwen\34-35-36-43-44_psd_mlr.png', dpi=600)


#%% Fig 9: SNR in time domain (Origin & extract) (full & strain)

def strain(X):
    strain = np.zeros((X.shape[0],int((X.shape[1]-1)/100)))
    for i in range(X.shape[0]):
        k=0
        for j in range(int(X.shape[1]/100)):
            strain[i,j] = np.mean(X[i,k:k+100])
            k += 100
    return strain

# load data
data = io.loadmat(r'F:\SSVEP\dataset\preprocessed_data\weisiwen\34-35-36-43-44__POZ\snr_t_mlr.mat')
origin = data['origin']
w1 = data['w1']
w2 = data['w2']
w3 = data['w3']
del data

origins = strain(origin)
w1s = strain(w1)
w2s = strain(w2)
w3s = strain(w3)

# plot
fig = plt.figure(figsize=(14,12))
gs = GridSpec(6,7, figure=fig)
sns.set(style='whitegrid')

ax1 = fig.add_subplot(gs[0:2, 0:4])
ax1.set_title("8Hz SSVEP's SNR", fontsize=18)
ax1.set_xlabel('Time/ms', fontsize=16)
ax1.set_ylabel('Signal-Noise-Ratio', fontsize=16)
ax1.plot(origin[0,:], label='Origin', color='tab:blue', linewidth=1.5)
ax1.plot(w1[0,:], label='w1', color='tab:orange', linewidth=1)
ax1.plot(w2[0,:], label='w2', color='tab:green', linewidth=1)
ax1.plot(w3[0,:], label='w3', color='tab:red', linewidth=1)
ax1.legend(loc='best', fontsize=14)

ax2 = fig.add_subplot(gs[0:2, 4:])
ax2.set_title("8Hz SSVEP's SNR (compressed)", fontsize=18)
ax2.set_xlabel('Time/100ms', fontsize=16)
ax2.set_ylabel('Signal-Noise-Ratio', fontsize=16)
ax2.plot(origins[0,:], label='Origin', color='tab:blue', linewidth=1.5)
ax2.plot(w1s[0,:], label='w1', color='tab:orange', linewidth=1)
ax2.plot(w2s[0,:], label='w2', color='tab:green', linewidth=1)
ax2.plot(w3s[0,:], label='w3', color='tab:red', linewidth=1)
ax2.legend(loc='best', fontsize=14)

ax3 = fig.add_subplot(gs[2:4, 0:4])
ax3.set_title("10Hz SSVEP's SNR", fontsize=18)
ax3.set_xlabel('Time/ms', fontsize=16)
ax3.set_ylabel('Signal-Noise-Ratio', fontsize=16)
ax3.plot(origin[1,:], label='Origin', color='tab:blue', linewidth=1.5)
ax3.plot(w1[1,:], label='w1', color='tab:orange', linewidth=1)
ax3.plot(w2[1,:], label='w2', color='tab:green', linewidth=1)
ax3.plot(w3[1,:], label='w3', color='tab:red', linewidth=1)
ax3.legend(loc='best', fontsize=14)

ax4 = fig.add_subplot(gs[2:4, 4:])
ax4.set_title("10Hz SSVEP's SNR (compressed)", fontsize=18)
ax4.set_xlabel('Time/100ms', fontsize=16)
ax4.set_ylabel('Signal-Noise-Ratio', fontsize=16)
ax4.plot(origins[1,:], label='Origin', color='tab:blue', linewidth=1.5)
ax4.plot(w1s[1,:], label='w1', color='tab:orange', linewidth=1)
ax4.plot(w2s[1,:], label='w2', color='tab:green', linewidth=1)
ax4.plot(w3s[1,:], label='w3', color='tab:red', linewidth=1)
ax4.legend(loc='best', fontsize=14)

ax5 = fig.add_subplot(gs[4:, 0:4])
ax5.set_title("15Hz SSVEP's SNR", fontsize=18)
ax5.set_xlabel('Time/ms', fontsize=16)
ax5.set_ylabel('Signal-Noise-Ratio', fontsize=16)
ax5.plot(origin[2,:], label='Origin', color='tab:blue', linewidth=1.5)
ax5.plot(w1[2,:], label='w1', color='tab:orange', linewidth=1)
ax5.plot(w2[2,:], label='w2', color='tab:green', linewidth=1)
ax5.plot(w3[2,:], label='w3', color='tab:red', linewidth=1)
ax5.legend(loc='best', fontsize=14)

ax6 = fig.add_subplot(gs[4:, 4:])
ax6.set_title("15Hz SSVEP's SNR (compressed)", fontsize=18)
ax6.set_xlabel('Time/100ms', fontsize=16)
ax6.set_ylabel('Signal-Noise-Ratio', fontsize=16)
ax6.plot(origins[2,:], label='Origin', color='tab:blue', linewidth=1.5)
ax6.plot(w1s[2,:], label='w1', color='tab:orange', linewidth=1)
ax6.plot(w2s[2,:], label='w2', color='tab:green', linewidth=1)
ax6.plot(w3s[2,:], label='w3', color='tab:red', linewidth=1)
ax6.legend(loc='best', fontsize=14)
del origin, origins, w1, w1s, w2, w2s, w3, w3s

fig.tight_layout()
plt.show()

# save fig
plt.savefig(r'F:\SSVEP\figures\weisiwen\34-35-36-43-44__POZ\snr_t_mlr', dpi=600)


#%% Fig 10: SNR in frequency domain (Line chart)
# compute frequency snr
from math import log

def snr_freq(X):
    '''
    X: (n_events, n_epochs, n_freqs)
    '''
    snr = np.zeros((X.shape[0], X.shape[1]))
    for i in range(X.shape[1]):
        snr[0,i] = np.sum(X[0,i,23:26]) / (np.sum(X[0,i,20:23]) + np.sum(X[0,i,26:29]))
        snr[0,i] = 10 * log(snr[0,i], 10)
        
        snr[1,i] = np.sum(X[1,i,29:32]) / (np.sum(X[1,i,26:29]) + np.sum(X[1,i,32:35]))
        snr[1,i] = 10 * log(snr[1,i], 10)
        
        snr[2,i] = np.sum(X[2,i,44:47]) / (np.sum(X[2,i,41:44]) + np.sum(X[2,i,47:50]))
        snr[2,i] = 10 * log(snr[2,i], 10)
        
    return snr

# load data & reform
w1 = ['w1' for i in range(100)]
w2 = ['w2' for i in range(100)]
w3 = ['w3' for i in range(100)]
model = w1 + w2 + w3 + w1 + w2 + w3
del w1, w2, w3

channel = ['OZ' for i in range(300)] + ['POZ' for i in range(300)]

# Group A
data = io.loadmat(r'F:\SSVEP\dataset\preprocessed_data\weisiwen\14_18__OZ\psd_mlr.mat')
fs = data['fs'][1,1,:]
sig = snr_freq(data['sig'])
w1 = snr_freq(data['w1']) - sig
w2 = snr_freq(data['w2']) - sig
w3 = snr_freq(data['w3']) - sig
del data, sig
snr1 = np.hstack((w1[0,:], w2[0,:], w3[0,:]))
snr2 = np.hstack((w1[1,:], w2[1,:], w3[1,:]))
snr3 = np.hstack((w1[2,:], w2[2,:], w3[2,:]))
del w1, w2, w3

data = io.loadmat(r'F:\SSVEP\dataset\preprocessed_data\weisiwen\14_18__POZ\psd_mlr.mat')
sig = snr_freq(data['sig'])
w1 = snr_freq(data['w1']) - sig
w2 = snr_freq(data['w2']) - sig
w3 = snr_freq(data['w3']) - sig
del data, sig
snr1 = np.hstack((snr1, w1[0,:], w2[0,:], w3[0,:]))
snr2 = np.hstack((snr2, w1[1,:], w2[1,:], w3[1,:]))
snr3 = np.hstack((snr3, w1[2,:], w2[2,:], w3[2,:]))
del w1, w2, w3

data11 = pd.DataFrame({'snr':snr1, 'model':model, 'channel':channel})
del snr1
data12 = pd.DataFrame({'snr':snr2, 'model':model, 'channel':channel})
del snr2
data13 = pd.DataFrame({'snr':snr3, 'model':model, 'channel':channel})
del snr3

# Group B
data = io.loadmat(r'F:\SSVEP\dataset\preprocessed_data\weisiwen\19_23__OZ\psd_mlr.mat')
sig = snr_freq(data['sig'])
w1 = snr_freq(data['w1']) - sig
w2 = snr_freq(data['w2']) - sig
w3 = snr_freq(data['w3']) - sig
del data, sig
snr1 = np.hstack((w1[0,:], w2[0,:], w3[0,:]))
snr2 = np.hstack((w1[1,:], w2[1,:], w3[1,:]))
snr3 = np.hstack((w1[2,:], w2[2,:], w3[2,:]))
del w1, w2, w3

data = io.loadmat(r'F:\SSVEP\dataset\preprocessed_data\weisiwen\19_23__POZ\psd_mlr.mat')
sig = snr_freq(data['sig'])
w1 = snr_freq(data['w1']) - sig
w2 = snr_freq(data['w2']) - sig
w3 = snr_freq(data['w3']) - sig
del data, sig
snr1 = np.hstack((snr1, w1[0,:], w2[0,:], w3[0,:]))
snr2 = np.hstack((snr2, w1[1,:], w2[1,:], w3[1,:]))
snr3 = np.hstack((snr3, w1[2,:], w2[2,:], w3[2,:]))
del w1, w2, w3

data21 = pd.DataFrame({'snr':snr1, 'model':model, 'channel':channel})
del snr1
data22 = pd.DataFrame({'snr':snr2, 'model':model, 'channel':channel})
del snr2
data23 = pd.DataFrame({'snr':snr3, 'model':model, 'channel':channel})
del snr3

# Group C
data = io.loadmat(r'F:\SSVEP\dataset\preprocessed_data\weisiwen\24_28__OZ\psd_mlr.mat')
sig = snr_freq(data['sig'])
w1 = snr_freq(data['w1']) - sig
w2 = snr_freq(data['w2']) - sig
w3 = snr_freq(data['w3']) - sig
del data, sig
snr1 = np.hstack((w1[0,:], w2[0,:], w3[0,:]))
snr2 = np.hstack((w1[1,:], w2[1,:], w3[1,:]))
snr3 = np.hstack((w1[2,:], w2[2,:], w3[2,:]))
del w1, w2, w3

data = io.loadmat(r'F:\SSVEP\dataset\preprocessed_data\weisiwen\24_28__POZ\psd_mlr.mat')
sig = snr_freq(data['sig'])
w1 = snr_freq(data['w1']) - sig
w2 = snr_freq(data['w2']) - sig
w3 = snr_freq(data['w3']) - sig
del data, sig
snr1 = np.hstack((snr1, w1[0,:], w2[0,:], w3[0,:]))
snr2 = np.hstack((snr2, w1[1,:], w2[1,:], w3[1,:]))
snr3 = np.hstack((snr3, w1[2,:], w2[2,:], w3[2,:]))
del w1, w2, w3

data31 = pd.DataFrame({'snr':snr1, 'model':model, 'channel':channel})
del snr1
data32 = pd.DataFrame({'snr':snr2, 'model':model, 'channel':channel})
del snr2
data33 = pd.DataFrame({'snr':snr3, 'model':model, 'channel':channel})
del snr3

# Group D
data = io.loadmat(r'F:\SSVEP\dataset\preprocessed_data\weisiwen\34-35-36-43-44__OZ\psd_mlr.mat')
sig = snr_freq(data['sig'])
w1 = snr_freq(data['w1']) - sig
w2 = snr_freq(data['w2']) - sig
w3 = snr_freq(data['w3']) - sig
del data, sig
snr1 = np.hstack((w1[0,:], w2[0,:], w3[0,:]))
snr2 = np.hstack((w1[1,:], w2[1,:], w3[1,:]))
snr3 = np.hstack((w1[2,:], w2[2,:], w3[2,:]))
del w1, w2, w3

data = io.loadmat(r'F:\SSVEP\dataset\preprocessed_data\weisiwen\34-35-36-43-44__POZ\psd_mlr.mat')
sig = snr_freq(data['sig'])
w1 = snr_freq(data['w1']) - sig
w2 = snr_freq(data['w2']) - sig
w3 = snr_freq(data['w3']) - sig
del data, sig
snr1 = np.hstack((snr1, w1[0,:], w2[0,:], w3[0,:]))
snr2 = np.hstack((snr2, w1[1,:], w2[1,:], w3[1,:]))
snr3 = np.hstack((snr3, w1[2,:], w2[2,:], w3[2,:]))
del w1, w2, w3

data41 = pd.DataFrame({'snr':snr1, 'model':model, 'channel':channel})
del snr1
data42 = pd.DataFrame({'snr':snr2, 'model':model, 'channel':channel})
del snr2
data43 = pd.DataFrame({'snr':snr3, 'model':model, 'channel':channel})
del snr3

# plot
fig = plt.figure(figsize=(16,14))
gs = GridSpec(8,9, figure=fig)
sns.set(style='whitegrid')

# Group A
ax1 = fig.add_subplot(gs[0:2, 0:3])
ax1.tick_params(axis='both', labelsize=18)
ax1 = sns.pointplot(x='model', y='snr', hue='channel', data=data11, dodge=True,
                    markers=['o', 'x'], linestyles=['-', '--'], palette='tab10',
                    ci=95, capsize=.2)
ax1.set_title("8Hz SNR improvement", fontsize=18)
ax1.set_xlabel('')
ax1.set_ylabel('A', fontsize=18)
ax1.legend(loc='best', fontsize=16)
del data11

ax2 = fig.add_subplot(gs[0:2, 3:6])
ax2.tick_params(axis='both', labelsize=18)
ax2 = sns.pointplot(x='model', y='snr', hue='channel', data=data12, dodge=True,
                    markers=['o', 'x'], linestyles=['-', '--'], palette='tab10',
                    ci=95, capsize=.2)
ax2.set_title("10Hz SNR improvement", fontsize=18)
ax2.set_xlabel('')
ax2.set_ylabel('')
ax2.legend(loc='best', fontsize=16)
del data12

ax3 = fig.add_subplot(gs[0:2, 6:])
ax3.tick_params(axis='both', labelsize=18)
ax3 = sns.pointplot(x='model', y='snr', hue='channel', data=data13, dodge=True,
                    markers=['o', 'x'], linestyles=['-', '--'], palette='tab10',
                    ci=95, capsize=.2)
ax3.set_title("15Hz SNR improvement", fontsize=18)
ax3.set_xlabel('')
ax3.set_ylabel('')
ax3.legend(loc='best', fontsize=16)
del data13

# Group B
ax4 = fig.add_subplot(gs[2:4, 0:3])
ax4.tick_params(axis='both', labelsize=18)
ax4 = sns.pointplot(x='model', y='snr', hue='channel', data=data21, dodge=True,
                    markers=['o', 'x'], linestyles=['-', '--'], palette='tab10',
                    ci=95, capsize=.2)
ax4.set_xlabel('')
ax4.set_ylabel('B', fontsize=18)
ax4.legend(loc='best', fontsize=16)
del data21

ax5 = fig.add_subplot(gs[2:4, 3:6])
ax5.tick_params(axis='both', labelsize=18)
ax5 = sns.pointplot(x='model', y='snr', hue='channel', data=data22, dodge=True,
                    markers=['o', 'x'], linestyles=['-', '--'], palette='tab10',
                    ci=95, capsize=.2)
ax5.set_xlabel('')
ax5.set_ylabel('')
ax5.legend(loc='best', fontsize=16)
del data22

ax6 = fig.add_subplot(gs[2:4, 6:])
ax6.tick_params(axis='both', labelsize=18)
ax6 = sns.pointplot(x='model', y='snr', hue='channel', data=data23, dodge=True,
                    markers=['o', 'x'], linestyles=['-', '--'], palette='tab10',
                    ci=95, capsize=.2)
ax6.set_xlabel('')
ax6.set_ylabel('')
ax6.legend(loc='best', fontsize=16)
del data23

# Group C
ax7 = fig.add_subplot(gs[4:6, 0:3])
ax7.tick_params(axis='both', labelsize=18)
ax7 = sns.pointplot(x='model', y='snr', hue='channel', data=data31, dodge=True,
                    markers=['o', 'x'], linestyles=['-', '--'], palette='tab10',
                    ci=95, capsize=.2)
ax7.set_xlabel('')
ax7.set_ylabel('C', fontsize=18)
ax7.legend(loc='best', fontsize=16)
del data31

ax8 = fig.add_subplot(gs[4:6, 3:6])
ax8.tick_params(axis='both', labelsize=18)
ax8 = sns.pointplot(x='model', y='snr', hue='channel', data=data32, dodge=True,
                    markers=['o', 'x'], linestyles=['-', '--'], palette='tab10',
                    ci=95, capsize=.2)
ax8.set_xlabel('')
ax8.set_ylabel('')
ax8.legend(loc='best', fontsize=16)
del data32

ax9 = fig.add_subplot(gs[4:6, 6:])
ax9.tick_params(axis='both', labelsize=18)
ax9 = sns.pointplot(x='model', y='snr', hue='channel', data=data33, dodge=True,
                    markers=['o', 'x'], linestyles=['-', '--'], palette='tab10',
                    ci=95, capsize=.2)
ax9.set_xlabel('')
ax9.set_ylabel('')
ax9.legend(loc='best', fontsize=16)
del data33

# Group D
ax10 = fig.add_subplot(gs[6:, 0:3])
ax10.tick_params(axis='both', labelsize=18)
ax10 = sns.pointplot(x='model', y='snr', hue='channel', data=data41, dodge=True,
                    markers=['o', 'x'], linestyles=['-', '--'], palette='tab10',
                    ci=95, capsize=.2)
ax10.set_xlabel('')
ax10.set_ylabel('D', fontsize=18)
ax10.legend(loc='best', fontsize=16)
del data41

ax11 = fig.add_subplot(gs[6:, 3:6])
ax11.tick_params(axis='both', labelsize=18)
ax11 = sns.pointplot(x='model', y='snr', hue='channel', data=data42, dodge=True,
                    markers=['o', 'x'], linestyles=['-', '--'], palette='tab10',
                    ci=95, capsize=.2)
ax11.set_xlabel('')
ax11.set_ylabel('')
ax11.legend(loc='best', fontsize=16)
del data42

ax12 = fig.add_subplot(gs[6:, 6:])
ax12.tick_params(axis='both', labelsize=18)
ax12 = sns.pointplot(x='model', y='snr', hue='channel', data=data43, dodge=True,
                    markers=['o', 'x'], linestyles=['-', '--'], palette='tab10',
                    ci=95, capsize=.2)
ax12.set_xlabel('')
ax12.set_ylabel('')
ax12.legend(loc='best', fontsize=16)
del data43
del channel, fs, model

#save data
fig.tight_layout()
plt.show()

plt.savefig(r'F:\SSVEP\figures\weisiwen\snr_freq.png', dpi=600)

