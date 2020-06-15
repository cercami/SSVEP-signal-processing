# -*- coding: utf-8 -*-
"""
Created on Sat Jan 11 15:14:37 2020

@author: Brynhildr
"""

#%% Import third part module
import numpy as np
import scipy.io as io

import srca
import copy

#%% common trca for origin data
ns = -60
acc_ori_tr = np.zeros((5,10))
for nt in range(10):
    eeg = io.loadmat(r'F:\SSVEP\dataset\preprocessed_data\wuqiaoyi\50_70_bp.mat')
    data = eeg['f_data'][:,ns:,[45,51,52,53,54,55,58,59,60],2140:int(2240+nt*100)]*1e6
    print('Data length:' + str((nt+1)*100) + 'ms')
    del eeg
    # cross-validation
    acc = []
    N = 5
    for cv in range(N):
        print('running TRCA program: train < test')
        a = int(cv * (data.shape[1]/N))
        tr_data = data[:,a:a+int(data.shape[1]/N),:,:]
        te_data = copy.deepcopy(data)
        te_data = np.delete(te_data, [a+i for i in range(tr_data.shape[1])], axis=1)
        acc_temp = srca.pure_trca(train_data=tr_data, test_data=te_data)
        acc.append(np.sum(acc_temp))
        del acc_temp
        print(str(cv+1) + 'th cv complete\n')
    acc = np.array(acc)/(te_data.shape[1]*2)
    acc_ori_tr[:,nt] = acc
    del acc

acc_ori_te = np.zeros((5,10))
for nt in range(10):
    eeg = io.loadmat(r'F:\SSVEP\dataset\preprocessed_data\wuqiaoyi\50_70_bp.mat')
    data = eeg['f_data'][:,ns:,[45,51,52,53,54,55,58,59,60],2140:int(2240+nt*100)]*1e6
    print('Data length:' + str((nt+1)*100) + 'ms')
    del eeg
    # cross-validation
    acc = []
    N = 5
    for cv in range(N):
        print('running TRCA program: train > test')
        a = int(cv * (data.shape[1]/N))
        tr_data = data[:,a:a+int(data.shape[1]/N),:,:]
        te_data = copy.deepcopy(data)
        te_data = np.delete(te_data, [a+i for i in range(tr_data.shape[1])], axis=1)
        acc_temp = srca.pure_trca(train_data=te_data, test_data=tr_data)
        acc.append(np.sum(acc_temp))
        del acc_temp
        print(str(cv+1) + 'th cv complete\n')
    acc = np.array(acc)/(tr_data.shape[1]*2)
    acc_ori_te[:,nt] = acc
    del acc

#%% common trca for SRCA data
acc_srca_tr = np.zeros((4,5,10))
for nfile in range(4):
    for nt in range(7):
        if nfile == 0:  # 55 for training
            eeg = io.loadmat(r'F:\SSVEP\dataset\preprocessed_data\real1_1\mcee_%d.mat' %(nt))
        elif nfile == 1:  # 40 for training
            eeg = io.loadmat(r'F:\SSVEP\dataset\preprocessed_data\real1_2\mcee_%d.mat' %(nt))
        elif nfile == 2:  # 30 for training
            eeg = io.loadmat(r'F:\SSVEP\dataset\preprocessed_data\real1_3\mcee_%d.mat' %(nt))
        elif nfile == 3:  # 25 for training
            eeg = io.loadmat(r'F:\SSVEP\dataset\preprocessed_data\real1_4\mcee_%d.mat' %(nt))
        data = eeg['mcee_sig'][:,ns:,:,1140:int(1240+nt*100)]
        print('Data length:' + str((nt+1)*100) + 'ms')
        del eeg
        # cross-validation
        acc = []
        N = 5
        for cv in range(N):
            print('running TRCA program...')
            a = int(cv * (data.shape[1]/N))
            tr_data = data[:,a:a+int(data.shape[1]/N),:,:]
            te_data = copy.deepcopy(data)
            te_data = np.delete(te_data, [a+i for i in range(tr_data.shape[1])], axis=1)
            acc_temp = srca.pure_trca(train_data=tr_data, test_data=te_data)
            acc.append(np.sum(acc_temp))
            del acc_temp
            print(str(cv+1) + 'th cv complete\n')
        acc = np.array(acc)/(te_data.shape[1]*2)
        acc_srca_tr[nfile,:,nt] = acc
        del acc

acc_srca_te = np.zeros((4,5,10))
for nfile in range(4):
    for nt in range(7):
        if nfile == 0:  # 55 for training
            eeg = io.loadmat(r'F:\SSVEP\dataset\preprocessed_data\real1_1\mcee_%d.mat' %(nt))
        elif nfile == 1:  # 40 for training
            eeg = io.loadmat(r'F:\SSVEP\dataset\preprocessed_data\real1_2\mcee_%d.mat' %(nt))
        elif nfile == 2:  # 30 for training
            eeg = io.loadmat(r'F:\SSVEP\dataset\preprocessed_data\real1_3\mcee_%d.mat' %(nt))
        elif nfile == 3:  # 25 for training
            eeg = io.loadmat(r'F:\SSVEP\dataset\preprocessed_data\real1_4\mcee_%d.mat' %(nt))
        data = eeg['mcee_sig'][:,ns:,:,1140:int(1240+nt*100)]
        print('Data length:' + str((nt+1)*100) + 'ms')
        del eeg
        # cross-validation
        acc = []
        N = 5
        for cv in range(N):
            print('running TRCA program...')
            a = int(cv * (data.shape[1]/N))
            tr_data = data[:,a:a+int(data.shape[1]/N),:,:]
            te_data = copy.deepcopy(data)
            te_data = np.delete(te_data, [a+i for i in range(tr_data.shape[1])], axis=1)
            acc_temp = srca.pure_trca(train_data=te_data, test_data=tr_data)
            acc.append(np.sum(acc_temp))
            del acc_temp
            print(str(cv+1) + 'th cv complete\n')
        acc = np.array(acc)/(tr_data.shape[1]*2)
        acc_srca_te[nfile,:,nt] = acc
        del acc
   
#%% DataFrame Preparation
import pandas as pd
import xlrd

# load accuracy data from excel file
excel_path = r'C:\Users\brynh\Desktop\result.xlsx'
excel_file = xlrd.open_workbook(excel_path, encoding_override='utf-8')
all_sheet = excel_file.sheets()

ori = []
for each_row in range(all_sheet[0].nrows):
    ori.append(all_sheet[0].row_values(each_row))

ols_snr = []
for each_row in range(all_sheet[1].nrows):
    ols_snr.append(all_sheet[1].row_values(each_row))

ols_corr = []
for each_row in range(all_sheet[2].nrows):
    ols_corr.append(all_sheet[2].row_values(each_row))

ri_corr = []
for each_row in range(all_sheet[3].nrows):
    ri_corr.append(all_sheet[3].row_values(each_row))
del excel_path, excel_file, all_sheet, each_row

# strings' preparation: length & group
length = []
for i in range(10):
    length += [(str(100*(i+1))+'ms') for j in range(5)]
total_length = length + length + length + length
del length
   
group_list = ['Origin', 'OLS & SNR', 'OLS & Corr', 'Ridge & SNR', 'Ridge & Corr',
              'Lasso & SNR', 'Lasso & Corr', 'ElasticNet & SNR', 'ElasticNet & Corr']
group_id = ['ori', 'ols_snr', 'ols_corr', 'ri_corr']
for i in range(len(group_id)):
    exec('g_%s=[group_list[i] for j in range(50)]' %(group_id[i]))
del group_list, i
total_group = g_ori + g_ols_snr + g_ols_corr + g_ri_corr
del g_ori, g_ols_snr, g_ols_corr, g_ri_corr


# data extraction: origin data
d_ori_re = []
d_ori_er = []
for i in range(10):  # train < test
    for j in range(5):
        d_ori_re.append(ori[j][i+2])
d_ori_re = np.array(d_ori_re)
for i in range(10):  # test < train
    for j in range(5):
        d_ori_er.append(ori[j+7][i+2])
d_ori_er = np.array(d_ori_er)
del ori

# data extraction: srca data
for i in range(len(group_id)-1):
    # SRCA training samples: 55
    exec('d_%s_re_55 = np.array([eval(group_id[i+1])[k][j+2] for j in range(10) for k in range(5)])' %(group_id[i+1]))
    exec('d_%s_er_55 = np.array([eval(group_id[i+1])[k+7][j+2] for j in range(10) for k in range(5)])' %(group_id[i+1]))
    # SRCA training samples: 40
    exec('d_%s_re_40 = np.array([eval(group_id[i+1])[k+14][j+2] for j in range(10) for k in range(5)])' %(group_id[i+1]))
    exec('d_%s_er_40 = np.array([eval(group_id[i+1])[k+21][j+2] for j in range(10) for k in range(5)])' %(group_id[i+1]))
    # SRCA training samples: 30
    exec('d_%s_re_30 = np.array([eval(group_id[i+1])[k+28][j+2] for j in range(10) for k in range(5)])' %(group_id[i+1]))
    exec('d_%s_er_30 = np.array([eval(group_id[i+1])[k+35][j+2] for j in range(10) for k in range(5)])' %(group_id[i+1]))
    # SRCA training samples: 25
    exec('d_%s_re_25 = np.array([eval(group_id[i+1])[k+42][j+2] for j in range(10) for k in range(5)])' %(group_id[i+1]))
    exec('d_%s_er_25 = np.array([eval(group_id[i+1])[k+49][j+2] for j in range(10) for k in range(5)])' %(group_id[i+1]))
del group_id, i, j, ols_corr, ols_snr, ri_corr 

# data combination: 25, 30, 40, 55 samples' training
sample_list = ['25','30','40','55']
for i in range(len(sample_list)):
    exec("d_re_%s = np.hstack((d_ori_re, eval('d_ols_snr_re_%s'), eval('d_ols_corr_re_%s'), eval('d_ri_corr_re_%s')))"
         %(sample_list[i], sample_list[i], sample_list[i], sample_list[i]))
    exec("d_er_%s = np.hstack((d_ori_er, eval('d_ols_snr_er_%s'), eval('d_ols_corr_er_%s'), eval('d_ri_corr_er_%s')))"
         %(sample_list[i], sample_list[i], sample_list[i], sample_list[i]))
for i in range(len(sample_list)):
    exec('del d_ols_snr_re_%s, d_ols_corr_re_%s, d_ri_corr_re_%s'
         %(sample_list[i], sample_list[i], sample_list[i]))
    exec('del d_ols_snr_er_%s, d_ols_corr_er_%s, d_ri_corr_er_%s'
         %(sample_list[i], sample_list[i], sample_list[i]))
del d_ori_re, d_ori_er

# complete dataframe objects
for i in range(len(sample_list)):
    exec("acc_re_%s = pd.DataFrame({'acc':d_re_%s, 'length':total_length, 'group':total_group})"
         %(sample_list[i],sample_list[i]))
    exec("acc_er_%s = pd.DataFrame({'acc':d_er_%s, 'length':total_length, 'group':total_group})"
         %(sample_list[i],sample_list[i]))
for i in range(len(sample_list)):
    exec('del d_re_%s, d_er_%s' %(sample_list[i], sample_list[i]))
del i, sample_list, total_group, total_length

#%% plotting
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns

fig = plt.figure(figsize=(18,12))
gs = GridSpec(6, 9, figure=fig)
sns.set(style='whitegrid')
ax1 = fig.add_subplot(gs[:,:])
ax1.tick_params(axis='both', labelsize=24)
ax1 = sns.barplot(x='length', y='acc', hue='group', data=acc, ci='sd',
                  palette='tab10', saturation=.75)
ax1.set_xlabel('')
ax1.set_ylabel('Accuracy/%', fontsize=28)
ax1.legend(loc='best', fontsize=28)

fig.tight_layout()
plt.show()
plt.savefig(r'C:\Users\brynh\Desktop\figure.png', dpi=600)