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
import numpy as np
import scipy.io as io
#import srca
import copy

# load accuracy data from excel file
excel_path = r'C:\Users\brynh\Desktop\fs+dcpm.xlsx'
excel_file = xlrd.open_workbook(excel_path, encoding_override='utf-8')
all_sheet = excel_file.sheets()

ori = []
for each_row in range(all_sheet[0].nrows):
    ori.append(all_sheet[0].row_values(each_row))

oripi = []
for each_row in range(all_sheet[2].nrows):
    oripi.append(all_sheet[2].row_values(each_row))
    
snr = []
for each_row in range(all_sheet[3].nrows):
    snr.append(all_sheet[3].row_values(each_row))
    
snrpi = []
for each_row in range(all_sheet[4].nrows):
    snrpi.append(all_sheet[4].row_values(each_row))

#ols_snr_200 = []
#for each_row in range(all_sheet[3].nrows):
#    ols_snr_200.append(all_sheet[3].row_values(each_row))
    
#ols_corr = []
#for each_row in range(all_sheet[3].nrows):
#    ols_corr.append(all_sheet[3].row_values(each_row))
  
#ols_corr_200 = []
#for each_row in range(all_sheet[7].nrows):
#    ols_corr_200.append(all_sheet[7].row_values(each_row))
del excel_path, excel_file, all_sheet, each_row

#%% strings' preparation
length = []
for i in range(5):
    length += [(str(100*(i+1))+'ms') for j in range(10)]
#total_length = length + length + length
   
#group_list = ['Origin', 'OLS & SNR', 'OLS & Corr', 'OLS & SNR(0.4pi)', 'OLS & Corr(0.4pi)']
group_id = ['ori', 'oripi', 'snr', 'snrpi']
#for i in range(len(group_id)):
 #   exec('g_%s=[group_list[i] for j in range(25)]' %(group_id[i]))
#del group_list, i

# data extraction
d_ori_re = []
d_ori_er = []
for i in range(5):  # train < test
    for j in range(10):
        d_ori_re.append(ori[j][i+1])
d_ori_re = 100*np.array(d_ori_re)
for i in range(5):  # test < train
    for j in range(10):
        d_ori_er.append(ori[j+12][i+1])
d_ori_er = 100*np.array(d_ori_er)
del ori, i, j

d_ori_repi = []
d_ori_erpi = []
for i in range(5):  # train < test
    for j in range(10):
        d_ori_repi.append(oripi[j][i+1])
d_ori_repi = 100*np.array(d_ori_repi)
for i in range(5):  # test < train
    for j in range(10):
        d_ori_erpi.append(oripi[j+12][i+1])
d_ori_erpi = 100*np.array(d_ori_erpi)
del oripi, i, j

#%% data extraction: srca data
trainNum = [55,40,30,25]
trainNum = [55]
for tn in range(len(trainNum)):
    for i in range(len(group_id)-2):
        exec('d_%s_re_%d = 100*np.array([eval(group_id[i+2])[k+12*(tn*2)][j+2] for j in range(5) for k in range(10)])'
         %(group_id[i+2], trainNum[tn]))
        exec('d_%s_er_%d = 100*np.array([eval(group_id[i+2])[k+12*(tn*2+1)][j+2] for j in range(5) for k in range(10)])'
         %(group_id[i+2], trainNum[tn]))
del group_id, tn, i

#%% flexible part: 
# group
num = 50
origin = ['Origin' for i in range(num)]
origin_200 = ['Origin: 0.4pi' for i in range(num)]
for tn in range(len(trainNum)):
    exec("snr_%d = ['DCPM: %d' for i in range(num)]" %(trainNum[tn], trainNum[tn]))
    exec("snr_%d_200 = ['DCPM: %d(0.4pi)' for i in range(num)]" %(trainNum[tn], trainNum[tn]))
    #exec("srca_corr_%d = ['%d' for i in range(num)]" %(trainNum[tn], trainNum[tn]))
    #exec("srca_corr_%d_200 = ['%d(+60ms)' for i in range(num)]" %(trainNum[tn], trainNum[tn]))
group_snr = origin + origin_200 + snr_55 + snr_55_200

#group_corr = origin + srca_corr_25 + srca_corr_25_200 + srca_corr_30 + srca_corr_30_200
#group_corr += srca_corr_40 + srca_corr_40_200 + srca_corr_55 + srca_corr_55_200
del origin, origin_200, snr_55, snr_55_200, num, tn
#del srca_corr_25_200, srca_corr_30_200, srca_corr_40_200, srca_corr_55_200

#%% length
f_length = []
for i in range(5):
    f_length += [(str(100*(i+1))+'ms') for j in range(10)]

#length += f_length

total_length = 4 * length
del f_length

#%% data
data_re = np.hstack((d_ori_re, d_ori_repi, 
                     d_snr_re_55, d_snrpi_re_55))

data_er = np.hstack((d_ori_er, d_ori_erpi, 
                     d_snr_er_55, d_snrpi_er_55))

# dataframe
plot_re = pd.DataFrame({'acc':data_re, 'length':total_length, 'group':group_snr})
plot_er = pd.DataFrame({'acc':data_er, 'length':total_length, 'group':group_snr})
del data_re, data_er, group_snr, total_length

#%% plotting
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns

color = ['#FDBF6F', '#FF7F00', '#A6CEE3', '#1F78B4']
#, '#B2DF8A', '#33A02C', '#A6CEE3',
 #        '#1F78B4', '#CAB2D6', '#6A3D9A']
#color = ['#E31A1C', '#FDBF6F', '#FF7F00', '#B2DF8A', '#33A02C', '#A6CEE3',
#         '#1F78B4', '#CAB2D6', '#6A3D9A']
brynhildr = sns.color_palette(color)

fig = plt.figure(figsize=(15,10))
gs = GridSpec(2, 2, figure=fig)
sns.set(style='whitegrid')

ax1 = fig.add_subplot(gs[:1,:])
ax1.set_title('Train:Test = 1:4', fontsize=24)
ax1.tick_params(axis='both', labelsize=20)
ax1 = sns.barplot(x='length', y='acc', hue='group', data=plot_re, ci='sd',
                  palette=brynhildr, saturation=.75)
ax1.set_xlabel('Time/ms', fontsize=22)
ax1.set_ylabel('Accuracy/%', fontsize=22)
ax1.set_ylim([40, 105])
ax1.legend(loc='lower right', fontsize=14)

ax2 = fig.add_subplot(gs[1:,:])
ax2.set_title('Train:Test = 4:1', fontsize=24)
ax2.tick_params(axis='both', labelsize=20)
ax2 = sns.barplot(x='length', y='acc', hue='group', data=plot_er, ci='sd',
                  palette=brynhildr, saturation=.75)
ax2.set_xlabel('Time/ms', fontsize=22)
ax2.set_ylabel('Accuracy/%', fontsize=22)
ax2.set_ylim([40, 105])
ax2.legend(loc='lower right', fontsize=14)

fig.tight_layout()
plt.show()
plt.savefig(r'C:\Users\brynh\Desktop\dcpm-0.4pi.png', dpi=600)

#%%
data1 = d_ori_re
data2 = d_ori_er
x = np.zeros(10)
for i in range(5):
    x[i] = np.mean(data1[i*5:(i+1)*5])
    x[i+5] = np.mean(data2[i*5:(i+1)*5])
del data1, data2

data1 = d_ols_snr_re_25
data2 = d_ols_snr_er_25
y = np.zeros(10)
for i in range(5):
    y[i] = np.mean(data1[i*5:(i+1)*5])
    y[i+5] = np.mean(data2[i*5:(i+1)*5])
del data1, data2

#%%
chan_info = ['FP1', 'FPz',]