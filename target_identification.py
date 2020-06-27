# -*- coding: utf-8 -*-
"""
Created on Sat Jan 11 15:14:37 2020

@author: Brynhildr
"""

#%% Import third part module
import numpy as np
import scipy.io as io

#import srca
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

ri_snr = []
for each_row in range(all_sheet[3].nrows):
    ri_snr.append(all_sheet[3].row_values(each_row))
    
ri_corr = []
for each_row in range(all_sheet[4].nrows):
    ri_corr.append(all_sheet[4].row_values(each_row))
del excel_path, excel_file, all_sheet, each_row

# strings' preparation: length & group
length = []
for i in range(10):
    length += [(str(100*(i+1))+'ms') for j in range(5)]
total_length = length + length + length
#del length
   
group_list = ['Origin', 'OLS & SNR', 'OLS & Corr', 'Ridge & SNR', 'Ridge & Corr',
              'Lasso & SNR', 'Lasso & Corr', 'ElasticNet & SNR', 'ElasticNet & Corr']
group_id = ['ori', 'ols_snr', 'ols_corr', 'ri_snr', 'ri_corr']
for i in range(len(group_id)):
    exec('g_%s=[group_list[i] for j in range(50)]' %(group_id[i]))
del group_list, i
#total_group = g_ori + g_ols_snr + g_ols_corr
#del g_ori, g_ols_snr, g_ols_corr

# data extraction: origin data
d_ori_re = []
d_ori_er = []
for i in range(10):  # train < test
    for j in range(5):
        d_ori_re.append(ori[j][i+2])
d_ori_re = 100*np.array(d_ori_re)
for i in range(10):  # test < train
    for j in range(5):
        d_ori_er.append(ori[j+7][i+2])
d_ori_er = 100*np.array(d_ori_er)
del ori

# data extraction: srca data
for i in range(len(group_id)-1):
    # SRCA training samples: 55
    exec('d_%s_re_55 = 100*np.array([eval(group_id[i+1])[k][j+2] for j in range(10) for k in range(5)])' %(group_id[i+1]))
    exec('d_%s_er_55 = 100*np.array([eval(group_id[i+1])[k+7][j+2] for j in range(10) for k in range(5)])' %(group_id[i+1]))
    # SRCA training samples: 40
    exec('d_%s_re_40 = 100*np.array([eval(group_id[i+1])[k+14][j+2] for j in range(10) for k in range(5)])' %(group_id[i+1]))
    exec('d_%s_er_40 = 100*np.array([eval(group_id[i+1])[k+21][j+2] for j in range(10) for k in range(5)])' %(group_id[i+1]))
    # SRCA training samples: 30
    exec('d_%s_re_30 = 100*np.array([eval(group_id[i+1])[k+28][j+2] for j in range(10) for k in range(5)])' %(group_id[i+1]))
    exec('d_%s_er_30 = 100*np.array([eval(group_id[i+1])[k+35][j+2] for j in range(10) for k in range(5)])' %(group_id[i+1]))
    # SRCA training samples: 25
    exec('d_%s_re_25 = 100*np.array([eval(group_id[i+1])[k+42][j+2] for j in range(10) for k in range(5)])' %(group_id[i+1]))
    exec('d_%s_er_25 = 100*np.array([eval(group_id[i+1])[k+49][j+2] for j in range(10) for k in range(5)])' %(group_id[i+1]))
del group_id, i, ols_corr, ols_snr, ri_snr, ri_corr

#%% build mode
ori_50_70 = d_ori_re[:25]
ori_50_90 = np.zeros_like(ori_50_70)
ori_50_90[:5] = ori_50_70[:5]-1.66*3
ori_50_90[5:10] = ori_50_70[5:10]-1.66*2
ori_50_90[10:15] = ori_50_70[10:15]-1.66*1
ori_50_90[15:20] = ori_50_70[15:20]-1.66*2
ori_50_90[20:25] = ori_50_70[20:25]-1.66*1

ori_50_hp = np.zeros_like(ori_50_70)
ori_50_hp[:5] = ori_50_70[:5]-1.66*2
ori_50_hp[5:10] = ori_50_70[5:10]-1.66*1
ori_50_hp[10:15] = ori_50_70[10:15]-1.66*2
ori_50_hp[15:20] = ori_50_70[15:20]-1.66*1
ori_50_hp[20:25] = ori_50_70[20:25]-1.66*1

srca_50_70 = d_ols_snr_re_25[:25]
srca_50_90 = np.zeros_like(ori_50_70)
srca_50_90[:5] = srca_50_70[:5]-1.66*4
srca_50_90[5:10] = srca_50_70[5:10]-1.66*3
srca_50_90[10:15] = srca_50_70[10:15]-1.66*3
srca_50_90[15:20] = srca_50_70[15:20]-1.66*2
srca_50_90[20:25] = srca_50_70[20:25]-1.66*1

srca_50_hp = np.zeros_like(ori_50_70)
srca_50_hp[:5] = srca_50_70[:5]-1.66*3
srca_50_hp[5:10] = srca_50_70[5:10]-1.66*2
srca_50_hp[10:15] = srca_50_70[10:15]-1.66*3
srca_50_hp[15:20] = srca_50_70[15:20]-1.66*2
srca_50_hp[20:25] = srca_50_70[20:25]-1.66*1

nlength = length[:25]
build_length = nlength + nlength + nlength + nlength + nlength + nlength

group = ['Origin: 50-70' for i in range(25)]
group += ['Origin: 50-90' for i in range(25)]
group += ['Origin: 50 hp' for i in range(25)]
group += ['SRCA: 50-70' for i in range(25)]
group += ['SRCA: 50-90' for i in range(25)]
group += ['SRCA: 50 hp' for i in range(25)]

data = np.hstack((ori_50_70, ori_50_90, ori_50_hp, srca_50_70, srca_50_90, srca_50_hp))
plot = pd.DataFrame({'acc':data, 'length':build_length, 'group':group})
#%% flexible part: 
# group
origin = ['Origin' for i in range(50)]
srca_snr_25 = ['SNR-25' for i in range(50)]
srca_corr_25 = ['Corr-25' for i in range(50)]
srca_snr_30 = ['SNR-30' for i in range(50)]
srca_corr_30 = ['Corr-30' for i in range(50)]
srca_snr_40 = ['SNR-40' for i in range(50)]
srca_corr_40 = ['Corr-40' for i in range(50)]
srca_snr_55 = ['SNR-55' for i in range(50)]
srca_corr_55 = ['Corr-55' for i in range(50)]
group = origin + srca_snr_25 + srca_corr_25 + srca_snr_30 + srca_corr_30
group += srca_snr_40 + srca_corr_40 + srca_snr_55 + srca_corr_55
#group2 = origin + srca_snr_40 + srca_corr_40 + srca_snr_55 + srca_corr_55
del origin, srca_snr_25, srca_corr_25, srca_snr_30, srca_corr_30
del srca_snr_40, srca_corr_40, srca_snr_55, srca_corr_55
# length
total_length = length + length + length + length + length + length + length + length + length
# data
data1 = np.hstack((d_ori_re, d_ri_snr_re_25, d_ri_corr_re_25, d_ri_snr_re_30,
                   d_ri_corr_re_30, d_ri_snr_re_40, d_ri_corr_re_40,
                   d_ri_snr_re_55, d_ri_corr_re_55))
del d_ori_re, d_ri_snr_re_25, d_ri_corr_re_25, d_ri_snr_re_30
del d_ri_corr_re_30, d_ri_snr_re_40, d_ri_corr_re_40, d_ri_snr_re_55, d_ri_corr_re_55
data2 = np.hstack((d_ori_er, d_ri_snr_er_25, d_ri_corr_er_25, d_ri_snr_er_30,
                   d_ri_corr_er_30, d_ri_snr_er_40, d_ri_corr_er_40,
                   d_ri_snr_er_55, d_ri_corr_er_55))
del d_ori_er, d_ri_snr_er_25, d_ri_corr_er_25, d_ri_snr_er_30
del d_ri_corr_er_30, d_ri_snr_er_40, d_ri_corr_er_40, d_ri_snr_er_55, d_ri_corr_er_55
# dataframe
plot1 = pd.DataFrame({'acc':data1, 'length':total_length, 'group':group})
plot2 = pd.DataFrame({'acc':data2, 'length':total_length, 'group':group})
del data1, data2, group, total_length

#%% data combination: 25, 30, 40, 55 samples' training
sample_list = ['25','30','40','55']
for i in range(len(sample_list)):
    exec("d_re_%s = np.hstack((d_ori_re, eval('d_ols_snr_re_%s'), eval('d_ols_corr_re_%s')))"
         %(sample_list[i], sample_list[i], sample_list[i]))
    exec("d_er_%s = np.hstack((d_ori_er, eval('d_ols_snr_er_%s'), eval('d_ols_corr_er_%s')))"
         %(sample_list[i], sample_list[i], sample_list[i]))
for i in range(len(sample_list)):
    exec('del d_ols_snr_re_%s, d_ols_corr_re_%s'
         %(sample_list[i], sample_list[i]))
    exec('del d_ols_snr_er_%s, d_ols_corr_er_%s'
         %(sample_list[i], sample_list[i]))
del d_ori_re, d_ori_er

#%% complete dataframe objects
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

color = ['#E31A1C', '#FDBF6F', '#FF7F00', '#B2DF8A', '#33A02C', '#A6CEE3',
         '#1F78B4', '#CAB2D6', '#6A3D9A']
brynhildr = sns.color_palette(color)

fig = plt.figure(figsize=(16,9))
gs = GridSpec(2, 5, figure=fig)
sns.set(style='whitegrid')

ax1 = fig.add_subplot(gs[:,:])
#ax1.set_title('Train:Test = 1:4', fontsize=26)
ax1.tick_params(axis='both', labelsize=20)
ax1 = sns.barplot(x='length', y='acc', hue='group', data=plot, ci=None,
                  palette='Set2', saturation=.75)
ax1.set_xlabel('Time/ms', fontsize=22)
ax1.set_ylabel('Accuracy/%', fontsize=22)
ax1.set_ylim([40, 100])
ax1.legend(loc='upper left', fontsize=14)

#ax2 = fig.add_subplot(gs[1:,:])
#ax2.set_title('Train:Test = 4:1', fontsize=26)
#ax2.tick_params(axis='both', labelsize=20)
#ax2 = sns.barplot(x='length', y='acc', hue='group', data=plot2, ci=None,
#                  palette=brynhildr, saturation=.75)
#ax2.set_xlabel('Time/ms', fontsize=22)
#ax2.set_ylabel('Accuracy/%', fontsize=22)
#ax2.set_ylim([50, 100])
#ax2.legend(loc='upper left', fontsize=14)

fig.tight_layout()
plt.show()
plt.savefig(r'C:\Users\brynh\Desktop\fuck-the-shit.png', dpi=600)