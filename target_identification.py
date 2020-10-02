# -*- coding: utf-8 -*-
"""
Created on Sat Jan 11 15:14:37 2020

@author: Brynhildr
"""

#%% Import third part module
import numpy as np
import scipy.io as io
import mcee
import pandas as pd
import xlrd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
import copy

#%% TRCA/eTRCA for SRCA/origin data: 60Hz/48Hz binary
tar_chans = ['PZ ','PO5','PO3','POZ','PO4','PO6','O1 ','OZ ','O2 ']
train_num = [10, 20, 30, 40]
nameList = ['pangjun', 'mengqiangfan', 'chengqian']

acc_srca_trca = np.zeros((len(nameList), len(train_num), 5, 5))
acc_srca_etrca = np.zeros((len(nameList), len(train_num), 5, 5))
acc_ori_trca = np.zeros((len(nameList), len(train_num), 5, 5))
acc_ori_etrca = np.zeros((len(nameList), len(train_num), 5, 5))

for nPeo in range(len(nameList)):
    people = nameList[nPeo]
    eeg = io.loadmat(r'F:\SSVEP\dataset\preprocessed_data\%s\40_70bp.mat' %(people))
    f_data = eeg['f_data'][:2,:,:,:]
    chans = eeg['chan_info'].tolist()
    n_events = f_data.shape[0]
    print("Now running " + people + "'s data...")
    del eeg
    for tn in range(len(train_num)):
        ns = train_num[tn]
        print('Training trials: ' + str(ns))
        for cv in range(5):  # cross-validation in training
            print('CV: %d turn...' %(cv+1))
            for nt in range(5):
                print('Data length: %d00ms' %(nt+1))
                srcaModel = io.loadmat(r'F:\SSVEP\realCV\%s\SNR\4C 40-70\train_%d\loop_%d\srca_%d.mat' 
                                   %(people, ns, cv, nt))
                # extract model info used in SRCA
                modelInfo = srcaModel['modelInfo'].flatten().tolist()
                modelChans = []
                for i in range(len(modelInfo)):
                    if i % 4 == 0 or i % 4 == 1:
                        modelChans.append(modelInfo[i].tolist()[:5])
                    else:
                        continue
                del modelInfo
                # extract trials info used in Cross Validation
                trainTrial = np.mean(srcaModel['trialInfo'], axis=0).astype(int)
                del srcaModel
                # extract origin data with correct trials & correct length
                train_data = f_data[:, trainTrial[:ns], :, :1240+nt*100]
                test_data = f_data[:, trainTrial[-80:], :, :1240+nt*100]
                del trainTrial
                # target identification main process
                accTRCA = mcee.SRCA_TRCA(train_data=train_data, test_data=test_data,
                    tar_chans=tar_chans, model_chans=modelChans, chans=chans,
                    regression='OLS', sp=1140)
                #acceTRCA = mcee.SRCA_eTRCA(train_data=train_data, test_data=test_data,
                    #tar_chans=tar_chans, model_chans=modelChans, chans=chans,
                    #regression='OLS', sp=1140)
                accOriTRCA = mcee.TRCA(train_data[:, :, [45,51,52,53,54,55,58,59,60], 1140:],
                                   test_data[:, :, [45,51,52,53,54,55,58,59,60], 1140:])
                #accOrieTRCA = mcee.eTRCA(train_data[:, :, [45,51,52,53,54,55,58,59,60], 1140:],
                                   #test_data[:, :, [45,51,52,53,54,55,58,59,60], 1140:])
                # re-arrange accuracy data
                accTRCA = np.sum(accTRCA)/(n_events*test_data.shape[1])
                #acceTRCA = np.sum(acceTRCA)/(n_events*test_data.shape[1])
                accOriTRCA = np.sum(accOriTRCA)/(n_events*test_data.shape[1])
                #accOrieTRCA = np.sum(accOrieTRCA)/(n_events*test_data.shape[1])
                # save accuracy data
                acc_srca_trca[nPeo, tn, cv, nt] = accTRCA
                #acc_srca_etrca[nPeo, tn, cv, nt] = acceTRCA
                acc_ori_trca[nPeo, tn, cv, nt] = accOriTRCA
                #acc_ori_etrca[nPeo, tn, cv, nt] = accOrieTRCA
                del accTRCA, accOriTRCA
                #del accOrieTRCA, accOrieTRCA
            print(str(cv+1) + 'th cross-validation complete!\n')
        print(str(ns) + ' training trials complete!\n')

#%%
show1 = copy.deepcopy(acc_srca_trca[1,:,:,:])
show2 = copy.deepcopy(acc_srca_trca[2,:,:,:])


#%% DCPM for SRCA/origin data (Online Mode)
tarChans = ['PZ ','PO5','PO3','POZ','PO4','PO6','O1 ','OZ ','O2 ']
trainNum = [10, 20, 30, 40]
nameList = ['pangjun', 'mengqiangfan', 'chengqian']

acc_srca_dcpm = np.zeros((len(nameList), len(trainNum), 5, 5))
acc_ori_dcpm = np.zeros((len(nameList), len(trainNum), 5, 5))

for nPeo in range(len(nameList)):
    people = nameList[nPeo]
    eeg = io.loadmat(r'F:\SSVEP\dataset\preprocessed_data\%s\40_70bp.mat' %(people))
    f_data = eeg['f_data'][[2,3],:,:,:]
    chans = eeg['chan_info'].tolist()
    del eeg
    print("Now running " + people + "'s data...")
    for tn in range(len(trainNum)):
        ns = trainNum[tn]
        print('Training trials: ' + str(ns))
        for cv in range(5):  # cross-validation in training
            print('CV: %d turn...' %(cv+1))
            for nt in range(5):
                print('Data length: %d00ms' %(nt+1))
                srcaModel = io.loadmat(r'F:\SSVEP\realCV\%s\SNR\4C 40-70\train_%d\loop_%d\srca_%d.mat' 
                                   %(people, ns, cv, nt))
                # extract model info used in SRCA
                modelInfo = srcaModel['modelInfo'].flatten().tolist()
                modelChans = []
                for i in range(len(modelInfo)):
                    if i & 4 == 2 or i % 4 == 3:
                        modelChans.append(modelInfo[i].tolist()[:5])
                    else:
                        continue
                del modelInfo
                # extract trials info used in Cross Validation
                trainTrial = np.mean(srcaModel['trialInfo'], axis=0).astype(int)
                del srcaModel
                # extract origin data with correct trials & correct length
                train_data = f_data[:, trainTrial[:ns], :, :1240+nt*100]
                test_data = f_data[:, trainTrial[-80:], :, :1240+nt*100]
                del trainTrial
                # target identification main process
                accDCPM = mcee.SRCA_DCPM(train_data=train_data, test_data=test_data,
                    tar_chans=tarChans, model_chans=modelChans, chans=chans,
                    regression='OLS', sp=1140, di=['1','2'])
                accOriDCPM = mcee.DCPM(train_data[:, :, [45,51,52,53,54,55,58,59,60], 1140:],
                    test_data[:, :, [45,51,52,53,54,55,58,59,60], 1140:], di=['1','2'])
                # re-arrange accuracy data
                accDCPM = np.sum(accDCPM)/(test_data.shape[1]*2)
                accOriDCPM = np.sum(accOriDCPM)/(test_data.shape[1]*2)
                # save accuracy data
                acc_srca_dcpm[nPeo, tn, cv, nt] = accDCPM
                acc_ori_dcpm[nPeo, tn, cv, nt] = accOriDCPM
                del accDCPM, accOriDCPM
            print(str(cv+1) + 'th cross-validation complete!\n')
        print(str(ns) + ' training trials complete!\n')

#%%
show1 = copy.deepcopy(acc_ori_dcpm[2,:,:,:])
show1[0,:,:] = show1[0,:,:]*20/160
show1[1,:,:] = show1[1,:,:]*40/160
show1[2,:,:] = show1[2,:,:]*60/160
show1[3,:,:] = show1[3,:,:]*80/160
show2 = copy.deepcopy(acc_srca_dcpm[2,:,:,:])
show2[0,:,:] = show2[0,:,:]*20/160
show2[1,:,:] = show2[1,:,:]*40/160
show2[2,:,:] = show2[2,:,:]*60/160
show2[3,:,:] = show2[3,:,:]*80/160

#%% test DCPM
tarChans = ['PZ ','PO5','PO3','POZ','PO4','PO6','O1 ','OZ ','O2 ']
#tarChans = ['PZ ','PO5','PO3','POZ','PO6','O1 ','OZ ','O2 ']
eeg = io.loadmat(r'F:\SSVEP\dataset\preprocessed_data\chengqian\60hz_50_70.mat')
f_data = eeg['f_data']
chans = eeg['chan_info'].tolist()
nTargetChan = len(tarChans)
srcaModel = io.loadmat(r'F:\SSVEP\realCV\chengqian\FS\60Hz 50-70\train_40\loop_4\srca_1.mat')
modelInfo = srcaModel['modelInfo'].flatten().tolist()
modelChans = []
for i in range(len(modelInfo)):
    modelChans.append(modelInfo[i].tolist())
#modelChans = np.delete(modelChans, 4)
trainTrial = np.mean(srcaModel['trialInfo'], axis=0).astype(int)
trainData = f_data[:, trainTrial[:40], :, :1340]
testData = f_data[:, trainTrial[-80:], :, :1340]
accDCPM = mcee.SRCA_DCPM(trainData, testData, tarChans, modelChans, chans, di=['1','2'])
accOriDCPM = mcee.DCPM(trainData[:, :, [45,51,52,53,54,55,58,59,60], 1140:],
                       testData[:, :, [45,51,52,53,54,55,58,59,60], 1140:], di=['1', '2'])
accDCPM = np.sum(accDCPM)/160
accOriDCPM = np.sum(accOriDCPM)/160
#%% [45,51,52,53,54,55,58,59,60]
# ['PZ ','PO5','PO3','POZ','PO4','PO6','O1 ','OZ ','O2 ']
srca_te = np.zeros((2,80,9,200))
data = testData
srca_te[0,:,:,:] = mcee.apply_SRCA(data[0,:,:,:], tarChans, modelChans, chans)
srca_te[1,:,:,:] = mcee.apply_SRCA(data[1,:,:,:], tarChans, modelChans, chans)

srca_tr = np.zeros((2,40,9,200))
data = trainData
srca_tr[0,:,:,:] = mcee.apply_SRCA(data[0,:,:,:], tarChans, modelChans, chans)
srca_tr[1,:,:,:] = mcee.apply_SRCA(data[1,:,:,:], tarChans, modelChans, chans)

#%%
plt.plot(np.mean(testData[0,:,51,1140:], axis=0))
plt.plot(np.mean(testData[1,:,51,1140:], axis=0))
#%%
plt.plot(np.mean(srca_te[0,:,1,:], axis=0))
plt.plot(np.mean(srca_te[1,:,1,:], axis=0))
#%%
x = mcee.fisher_score(testData[:, :, 45, 1140:])
y = mcee.fisher_score(srca_te[:, :, 0, :])
plt.plot(x, lable='origin')
plt.plot(y, label='SRCA')
plt.xlabel('Time/ms')
plt.ylabel('Fisher Score')
plt.title('PZ')
plt.show()
z = (np.mean(y) - np.mean(x))/np.mean(x)*100

#%%
def trim_axs(axs, N):
    axs = axs.flat
    for ax in axs[N:]:
        ax.remove()
    return axs[:N]

channels = ['Pz', 'PO5', 'PO3', 'POz', 'PO4', 'PO6', 'O1', 'Oz', 'O2']
figsize = (16, 9)
cols, rows = 3, 3
data = testData[:,:,[45,51,52,53,54,55,58,59,60],1140:1340]
data = srcaTe[:,:,:,:200]
sns.set(style='whitegrid')

axs = plt.figure(figsize=figsize, constrained_layout=True).subplots(rows, cols)
axs = trim_axs(axs, len(channels))
for ax, channel in zip(axs, channels):
    chan_index = channels.index(channel)
    ax.set_title(channel, fontsize=18)
    ax.tick_params(axis='both', labelsize=14)
    ax.plot(np.mean(data[0,:,chan_index,:], axis=0))
    ax.plot(np.mean(data[1,:,chan_index,:], axis=0))
    ax.set_yscale('log')
    #ax.set_yscale('log')
    #ax.legend(loc='upper right', fontsize=14)

#axs.tight_layout()
plt.show()


#%%
from mcee import snr_time, fisher_score
fig = plt.figure(figsize=(16,9))
gs = GridSpec(3,3, figure=fig)
sns.set(style='whitegrid')
data1 = testData[:,:,[45,51,52,53,54,55,58,59,60],1140:]
data2 = srcaTe

ax1 = fig.add_subplot(gs[:1,:1])
ax1.set_title('Pz', fontsize=18)
ax1.tick_params(axis='both', labelsize=14)
ax1.plot(np.mean(data2[0,:,0,:], axis=0), label='Origin')
ax1.plot(np.mean(data2[1,:,0,:], axis=0), label='SRCA')
ax1.set_xlabel('Time/ms', fontsize=16)
ax1.set_ylabel('Fisher Score', fontsize=16)
ax1.legend(loc='upper right', fontsize=14)

ax2 = fig.add_subplot(gs[:1,1:2])
ax2.set_title('PO5', fontsize=18)
ax2.tick_params(axis='both', labelsize=14)
ax2.plot(np.mean(data2[0,:,1,:], axis=0), label='Origin')
ax2.plot(np.mean(data2[1,:,1,:], axis=0), label='SRCA')
ax2.set_xlabel('Time/ms', fontsize=16)
ax2.set_ylabel('Fisher Score', fontsize=16)
ax2.legend(loc='upper right', fontsize=14)

ax3 = fig.add_subplot(gs[:1,2:])
ax3.set_title('PO3', fontsize=18)
ax3.tick_params(axis='both', labelsize=14)
ax3.plot(np.mean(data2[0,:,2,:], axis=0), label='Origin')
ax3.plot(np.mean(data2[1,:,2,:], axis=0), label='SRCA')
ax3.set_xlabel('Time/ms', fontsize=16)
ax3.set_ylabel('Fisher Score', fontsize=16)
ax3.legend(loc='upper right', fontsize=14)

ax4 = fig.add_subplot(gs[1:2,:1])
ax4.set_title('POz', fontsize=18)
ax4.tick_params(axis='both', labelsize=14)
ax4.plot(np.mean(data2[0,:,3,:], axis=0), label='Origin')
ax4.plot(np.mean(data2[1,:,3,:], axis=0), label='SRCA')
ax4.set_xlabel('Time/ms', fontsize=16)
ax4.set_ylabel('Fisher Score', fontsize=16)
ax4.legend(loc='upper right', fontsize=14)

ax5 = fig.add_subplot(gs[1:2,1:2])
ax5.set_title('PO4', fontsize=18)
ax5.tick_params(axis='both', labelsize=14)
ax5.plot(np.mean(data2[0,:,4,:], axis=0), label='Origin')
ax5.plot(np.mean(data2[1,:,4,:], axis=0), label='SRCA')
ax5.set_xlabel('Time/ms', fontsize=16)
ax5.set_ylabel('Fisher Score', fontsize=16)
ax5.legend(loc='upper right', fontsize=14)

ax6 = fig.add_subplot(gs[1:2,2:])
ax6.set_title('PO6', fontsize=18)
ax6.tick_params(axis='both', labelsize=14)
ax6.plot(np.mean(data2[0,:,5,:], axis=0), label='Origin')
ax6.plot(np.mean(data2[1,:,5,:], axis=0), label='SRCA')
ax6.set_xlabel('Time/ms', fontsize=16)
ax6.set_ylabel('Fisher Score', fontsize=16)
ax6.legend(loc='upper right', fontsize=14)

ax7 = fig.add_subplot(gs[2:,:1])
ax7.set_title('O1', fontsize=18)
ax7.tick_params(axis='both', labelsize=14)
ax7.plot(np.mean(data2[0,:,6,:], axis=0), label='Origin')
ax7.plot(np.mean(data2[1,:,6,:], axis=0), label='SRCA')
ax7.set_xlabel('Time/ms', fontsize=16)
ax7.set_ylabel('Fisher Score', fontsize=16)
ax7.legend(loc='upper right', fontsize=14)

ax8 = fig.add_subplot(gs[2:,1:2])
ax8.set_title('Oz', fontsize=18)
ax8.tick_params(axis='both', labelsize=14)
ax8.plot(np.mean(data2[0,:,7,:], axis=0), label='Origin')
ax8.plot(np.mean(data2[1,:,7,:], axis=0), label='SRCA')
ax8.set_xlabel('Time/ms', fontsize=16)
ax8.set_ylabel('Fisher Score', fontsize=16)
ax8.legend(loc='upper right', fontsize=14)

ax9 = fig.add_subplot(gs[2:,2:])
ax9.set_title('O2', fontsize=18)
ax9.tick_params(axis='both', labelsize=14)
ax9.plot(np.mean(data2[0,:,8,:], axis=0), label='Origin')
ax9.plot(np.mean(data2[1,:,8,:], axis=0), label='SRCA')
ax9.set_xlabel('Time/ms', fontsize=16)
ax9.set_ylabel('Fisher Score', fontsize=16)
ax9.legend(loc='upper right', fontsize=14)

fig.tight_layout()
plt.show()
plt.savefig(r'C:\Users\brynh\Desktop\9-5组会\wfsrca5-cq-split-te.png', dpi=600)

#%% DataFrame Preparation
# load accuracy data from excel file
excel_path = r'F:\SSVEP\results\RealCV\SNR\60Hz 40-70\防过拟合.xlsx'
excel_file = xlrd.open_workbook(excel_path, encoding_override='utf-8')
all_sheet = excel_file.sheets()

pangjun = []
for each_row in range(all_sheet[0].nrows):
    pangjun.append(all_sheet[0].row_values(each_row))

mengqiangfan = []
for each_row in range(all_sheet[1].nrows):
    mengqiangfan.append(all_sheet[1].row_values(each_row))
    
#guojiaming = []
#for each_row in range(all_sheet[2].nrows):
#    guojiaming.append(all_sheet[2].row_values(each_row))

chengqian = []
for each_row in range(all_sheet[2].nrows):
    chengqian.append(all_sheet[2].row_values(each_row))

del excel_path, excel_file, all_sheet, each_row

#% strings' preparation
length = []
for i in range(5):
    length += [(str(100*(i+1))+'ms') for j in range(5)]
del i
total_length = length*3
del length

#% group hue
group_list = ['Origin', 'SRCA (All)', 'SRCA (5)']
total_group = []
for i in range(len(group_list)):
    exec('g_%d=[group_list[i] for j in range(25)]' %(i))
    total_group += eval('g_%d' %(i))
del i
del g_0, g_1, g_2, group_list
#del g_4, g_5, g_6, g_7

#%% data extraction for 
ori_trca = []
srca_trca = []
srca_new = []

people = chengqian
k = 3

for i in range(5):      # rows
    for j in range(5):  # columns
        ori_trca.append(people[j+1+k*6][i+9])
        srca_trca.append(people[j+27+k*6][i+2])
        srca_new.append(people[j+1+k*6][i+2])
    del j
del i

#%% dataframe
data10 = np.hstack((ori_trca, srca_trca, srca_new))*1e2
train10 = pd.DataFrame({'acc':data10, 'length':total_length, 'group':total_group})
del data10
#%% dataframe
data20 = np.hstack((ori_trca, srca_trca, srca_new))*1e2
train20 = pd.DataFrame({'acc':data20, 'length':total_length, 'group':total_group})
del data20
#%% dataframe
data30 = np.hstack((ori_trca, srca_trca, srca_new))*1e2
train30 = pd.DataFrame({'acc':data30, 'length':total_length, 'group':total_group})
del data30
#%% dataframe
data40 = np.hstack((ori_trca, srca_trca, srca_new))*1e2
train40 = pd.DataFrame({'acc':data40, 'length':total_length, 'group':total_group})
del data40
#%% plotting
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns

#color = ['#E31A1C', '#FB9A99', '#FF7F00', '#FDBF6F', '#1F78B4', '#A6CEE3', '#33A02C', '#B2DF8A']
color = ['#FF7F00', '#1F78B4', '#A6CEE3']
#, '#B2DF8A', '#33A02C', '#A6CEE3',
 #        '#1F78B4', '#CAB2D6', '#6A3D9A']
#color = ['#E31A1C', '#FDBF6F', '#FF7F00', '#B2DF8A', '#33A02C', '#A6CEE3',
#         '#1F78B4', '#CAB2D6', '#6A3D9A']
brynhildr = sns.color_palette(color)

fig = plt.figure(figsize=(16,9))
gs = GridSpec(2, 2, figure=fig)
sns.set(style='whitegrid')

ax1 = fig.add_subplot(gs[:1,:1])
ax1.set_title('Training Samples: 10', fontsize=24)
ax1.tick_params(axis='both', labelsize=20)
ax1 = sns.barplot(x='length', y='acc', hue='group', data=train10, ci='sd',
                  palette=brynhildr, saturation=.75)
ax1.set_xlabel('Time/ms', fontsize=22)
ax1.set_ylabel('Accuracy/%', fontsize=22)
ax1.set_ylim([40, 105])
ax1.set_yticks(range(40,110,10))
ax1.legend(loc='lower right', fontsize=14)

ax2 = fig.add_subplot(gs[:1,1:])
ax2.set_title('Training Samples: 20', fontsize=24)
ax2.tick_params(axis='both', labelsize=20)
ax2 = sns.barplot(x='length', y='acc', hue='group', data=train20, ci='sd',
                  palette=brynhildr, saturation=.75)
ax2.set_xlabel('Time/ms', fontsize=22)
ax2.set_ylabel('Accuracy/%', fontsize=22)
ax2.set_ylim([40, 105])
ax2.set_yticks(range(40,110,10))
ax2.legend(loc='lower right', fontsize=14)

ax3 = fig.add_subplot(gs[1:,:1])
ax3.set_title('Training Samples: 30', fontsize=24)
ax3.tick_params(axis='both', labelsize=20)
ax3 = sns.barplot(x='length', y='acc', hue='group', data=train30, ci='sd',
                  palette=brynhildr, saturation=.75)
ax3.set_xlabel('Time/ms', fontsize=22)
ax3.set_ylabel('Accuracy/%', fontsize=22)
ax3.set_ylim([40, 105])
ax3.set_yticks(range(40,110,10))
ax3.legend(loc='lower right', fontsize=14)

ax4 = fig.add_subplot(gs[1:,1:])
ax4.set_title('Training Samples: 40', fontsize=24)
ax4.tick_params(axis='both', labelsize=20)
ax4 = sns.barplot(x='length', y='acc', hue='group', data=train40, ci='sd',
                  palette=brynhildr, saturation=.75)
ax4.set_xlabel('Time/ms', fontsize=22)
ax4.set_ylabel('Accuracy/%', fontsize=22)
ax4.set_ylim([40, 105])
ax4.set_yticks(range(40,110,10))
ax4.legend(loc='lower right', fontsize=14)

fig.tight_layout()
plt.show()
plt.savefig(r'C:\Users\brynh\Desktop\9-5组会\SNR-5only-40-70-cq.png', dpi=600)

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

#%% fs test
eeg = io.loadmat(r'F:\SSVEP\dataset\preprocessed_data\pangjun\60hz_50_70.mat')
f_data = eeg['f_data']
chans = eeg['chan_info'].tolist()
srca = io.loadmat(r'F:\SSVEP\realCV\pangjun\FS\train_10\loop_0\srca_4.mat')
tempChans = srca['modelInfo'].flatten().tolist()
trainTrials = np.mean(srca['trialInfo'], axis=0).astype(int)
modelChans = []
for i in range(9):
    modelChans.append(tempChans[i].tolist())
del tempChans, srca, i
# extract two dataset
trainData = f_data[:, trainTrials[:10], :, 1140:1640]
trainData = trainData[:, :, [45,51,52,53,54,55,58,59,60], :]
testData = f_data[:, trainTrials[-80:], :, 1140:1640]
testData = testData[:, :, [45,51,52,53,54,55,58,59,60], :]
del trainTrials

#%%
acc = mcee.DCPM(data, testData, di=['1','2'])
#%%
data = np.zeros((2, 40, 9, 500))
data[:,:10,:,:] = trainData
data[:,10:20,:,:] = trainData
data[:,20:30,:,:] = trainData
data[:,30:40,:,:] = trainData
#%%
data = trainData
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