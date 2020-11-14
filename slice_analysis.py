# -*- coding: utf-8 -*-
"""
Created on Fri Sep  4 12:00:29 2020

@author: Brynhildr
"""
#%% load 3rd-part module
import numpy as np
import scipy.io as io
#import srca
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
#import fp_growth as fpg
import mcee
import seaborn as sns
from math import pi

#%% Slice Analysis: SNR
tar_chans = ['PZ ','PO5','PO3','POZ','PO4','PO6','O1 ','OZ ','O2 ']
srca = io.loadmat(r'F:\SSVEP\realCV\pangjun\SNR\4C 40-70\train_40\loop_4\srca_2.mat')
tempChans = srca['modelInfo'].flatten().tolist()
trainTrials = np.mean(srca['trialInfo'], axis=0).astype(int)
f60p0Chans, f60p1Chans, f48p0Chans, f48p1Chans = [], [], [], []
num = 5
for i in range(9):
    f60p0Chans.append(tempChans[i*4].tolist())
    f60p1Chans.append(tempChans[i*4+1].tolist())
    #f48p0Chans.append(tempChans[i*4+2].tolist())
    #f48p1Chans.append(tempChans[i*4+3].tolist())
modelChans = []
for i in range(len(tempChans)):
    if i % 2 == 0 or i % 2 == 1:
        modelChans.append(tempChans[i].tolist())
del tempChans, srca, i
eeg = io.loadmat(r'F:\SSVEP\dataset\preprocessed_data\pangjun\40_70bp.mat')
f_data = eeg['f_data'][:2,:,:,:]
chans = eeg['chan_info'].tolist()
trainData = f_data[:, trainTrials[:40], :, :1440]
testData = f_data[:, trainTrials[-80:], :, :1440]
del trainTrials
del eeg, f_data

#%% (1) apply SRCA models
srcaTe = np.zeros((2,80,9,300))
srcaTe[0,:,:,:] = mcee.apply_SRCA(testData[0,:,:,:], tar_chans, f60p0Chans, chans)
srcaTe[1,:,:,:] = mcee.apply_SRCA(testData[1,:,:,:], tar_chans, f60p1Chans, chans)
srcaTr = np.zeros((2,40,9,300))
srcaTr[0,:,:,:] = mcee.apply_SRCA(trainData[0,:,:,:], tar_chans, f60p0Chans, chans)
srcaTr[1,:,:,:] = mcee.apply_SRCA(trainData[1,:,:,:], tar_chans, f60p1Chans, chans)

#%%
plt.plot(np.mean(testData[0,:,59,1140:], axis=0))
plt.plot(np.mean(testData[1,:,59,1140:], axis=0))
#%%
plt.plot(np.mean(srcaTe[0,:,7,:], axis=0))
plt.plot(np.mean(srcaTe[1,:,7,:], axis=0))
#%%

#%% (2) check waveform: origin data
ori_chan = [45,51,52,53,54,55,58,59,60]
i = -2
data = trainData
plt.plot(np.mean(data[0,:,ori_chan[i],1140:], axis=0))
plt.plot(np.mean(data[1,:,ori_chan[i],1140:], axis=0))

#%% (3) check waveform: SRCA data
ori_chan = [45,51,52,53,54,55,58,59,60]
i = -2
data = srcaTr
plt.plot(np.mean(data[0,:,i,:], axis=0))
plt.plot(np.mean(data[1,:,i,:], axis=0))

#%% (4) check acc
realAcc, realAcc2, realAcc3, realAcc4 = 0, 0, 0, 0
for i in range(20):
    randPick = np.arange(80)
    np.random.shuffle(randPick)
    acc1 = mcee.SRCA_TRCA(train_data=testData[:,randPick[:20],:,:],
                     test_data=testData[:,randPick[20:],:,:], tar_chans=tar_chans,
                     model_chans=modelChans, chans=chans, regression='OLS', sp=1140)
    acc1 = np.sum(acc1)/(2*60)
    realAcc += acc1
    acc2 = mcee.SRCA_TRCA(train_data=trainData, test_data=testData[:,randPick[20:],:,:], tar_chans=tar_chans,
                     model_chans=modelChans, chans=chans, regression='OLS', sp=1140)
    acc2 = np.sum(acc2)/(2*60)
    realAcc2 += acc2
    z = testData[:,:,[45,51,52,53,54,55,58,59,60],1140:]
    x = z[:,randPick[20:],:,:]
    y = z[:,randPick[:20],:,:]
    acc3 = mcee.TRCA(trainData[:,:,[45,51,52,53,54,55,58,59,60],1140:], x)
    acc3 = np.sum(acc3)/(2*60)
    realAcc3 += acc3
    acc4 = mcee.TRCA(y,x)
    acc4 = np.sum(acc4)/(2*60)
    realAcc4 += acc4
realAcc /= 20
realAcc2 /= 20
realAcc3 /= 20
realAcc4 /= 20

#%%
realAcc5 = 0
acc = mcee.SRCA_TRCA(train_data=trainData, test_data=trainData, tar_chans=tar_chans,
                     model_chans=modelChans, chans=chans, regression='OLS', sp=1140)
acc = np.sum(acc)/(2*20)
realAcc5 += acc

realAcc6 = 0
acc = mcee.TRCA(train_data=trainData[:,:,[45,51,52,53,54,55,58,59,60],1140:],
                test_data=trainData[:,:,[45,51,52,53,54,55,58,59,60],1140:])
acc = np.sum(acc)/(2*20)
realAcc6 += acc

#%%
realAcc4 = 0
for i in range(10):
    randPick = np.arange(80)
    np.random.shuffle(randPick)
    acc = mcee.SRCA_TRCA(train_data=trainData, test_data=testData[:,randPick[:40],:,:], tar_chans=tar_chans,
                     model_chans=modelChans, chans=chans, regression='OLS', sp=1140)
    acc = np.sum(acc)/(2*40)
    realAcc4 += acc
realAcc4 /= 10

#%% (5) check SNR alteration
k = 0
ori_chan = [45,51,52,53,54,55,58,59,60]
i = -3
x = mcee.snr_time(trainData[k, :, ori_chan[i], 1140:])
y = mcee.snr_time(srcaTr[k, :, i, :])
plt.plot(x.T)
plt.plot(y.T)
plt.show()
z = np.mean(y) - np.mean(x)

#%% (6) check SNR alteration in figure (training dataset)
eeg = io.loadmat(r'F:\SSVEP\realCV\chengqian\SNR\4C 40-70\train_20\loop_1\srca_1.mat')
para = eeg['parameter'].flatten().tolist()
model = eeg['modelInfo'].flatten().tolist()
tar_chans = ['PZ ','PO5','PO3','POZ','PO4','PO6','O1 ','OZ ','O2 ']

fig = plt.figure(figsize=(15,10))
gs = GridSpec(2,2,figure=fig)
sns.set(style='whitegrid')

ax1 = fig.add_subplot(gs[:,:])
ax1.set_title('SNR Alteration', fontsize=24)
ax1.tick_params(axis='both', labelsize=20)
for i in range(len(tar_chans)):
    exec('ax1.plot(para[i].T, label=tar_chans[i], linewidth=1.5)')
ax1.set_xlabel('Number of channels', fontsize=22)
ax1.set_ylabel('SNR', fontsize=22)
ax1.legend(loc='upper right', fontsize=20)

fig.tight_layout()
plt.show()

#%% (7) check target identification algorithms step by step
model_sig = srcaTr
n_events = srcaTr.shape[0]
# apply different srca models on one trial's data
n_tests = testData.shape[1]
target_sig = np.zeros((n_events, n_events, n_tests, 9, 300))
for ne_tr in range(n_events):
    temp_model = modelChans[ne_tr::n_events]
    for ne_re in range(n_events):
        target_sig[ne_tr, ne_re, :,:,:] = mcee.apply_SRCA(testData[ne_re,:,:,:],
                        tar_chans, temp_model, chans, regression='OLS', sp=1140)
    del ne_re
del ne_tr
# template data: (n_events, n_chans, n_times)
template = np.mean(model_sig, axis=1)
# Matrix Q: (n_events, n_chans, n_chans) | inter-channel covarianve
q = mcee.matrix_Q(model_sig)
# Matrix S: (n_events, n_chans, n_chans) | inter-channels' inter-trial covariance
s = mcee.matrix_S(model_sig)
# Spatial filter W: (n_events, n_chans)
w = mcee.spatial_W(q, s)
# Test data operating & Compute accuracy
r = np.zeros((n_events, n_tests, n_events))  # (n_events srca, n_tests, n_events test)
for nes in range(n_events):
    for nte in range(n_tests):
        for ner in range(n_events):
            temp_test = np.mat(w[nes, :]) * np.mat(target_sig[nes, ner, nte, :, :])
            temp_template = np.mat(w[nes, :]) * np.mat(template[nes, :, :])
            r[nes, nte, ner] = np.sum(np.tril(np.corrcoef(temp_test, temp_template),-1))
        del ner
    del nte
del nes
# compute accuracy
accuracy = []
for nes in range(n_events):
    for nte in range(n_tests):
        if np.max(np.where(r[nes, nte, :] == np.max(r[nes, nte, :]))) == nes:
            accuracy.append(1)


#%%********************************************************************************%%#
#%% Slice Analysis: FS
tar_chans = ['PZ ','PO5','PO3','POZ','PO4','PO6','O1 ','OZ ','O2 ']
srca = io.loadmat(r'F:\SSVEP\realCV\chengqian\FS\Cross identification\Cross 01\train_40\loop_3\srca_1.mat')
tempChans = srca['modelInfo'].flatten().tolist()
trainTrials = np.mean(srca['trialInfo'], axis=0).astype(int)
modelChans = []
for i in range(9):
    modelChans.append(tempChans[i].tolist())
del tempChans, srca, i
eeg = io.loadmat(r'F:\SSVEP\dataset\preprocessed_data\chengqian\40_70bp.mat')
f_data = eeg['f_data']
chans = eeg['chan_info'].tolist()
trainData = f_data[:, trainTrials[:40], :, :1340]
testData = f_data[:, trainTrials[-80:], :, :1340]
del trainTrials, eeg

#%% (1) apply SRCA models
srcaTe = np.zeros((2,80,9,200))
srcaTe[0,:,:,:] = mcee.apply_SRCA(testData[0,:,:,:1340], tar_chans, modelChans, chans)
srcaTe[1,:,:,:] = mcee.apply_SRCA(testData[1,:,:,:1340], tar_chans, modelChans, chans)

srcaTr = np.zeros((2,40,9,200))
srcaTr[0,:,:,:] = mcee.apply_SRCA(trainData[0,:,:,:1340], tar_chans, modelChans, chans)
srcaTr[1,:,:,:] = mcee.apply_SRCA(trainData[1,:,:,:1340], tar_chans, modelChans, chans)

#%% phase analysis
from mcee import template_corr
import pandas as pd

x = testData[0,:,59,1140:]
# x = srcaTr[0,:,7,:]
corr = np.zeros((100, x.shape[0]))
phase = [2*x/100 for x in range(100)]

for i in range(len(phase)):
    corr[i,:] = template_corr(x, 1000, 60, phase[i])

bestPhase = np.zeros((x.shape[0]))
for i in range(80):
    bestPhase[i] = 2*np.max(np.where(corr[:,i] == np.max(corr[:,i])))/100

dataPlot = pd.DataFrame({'phase':bestPhase})
ax = sns.boxplot(x=dataPlot['phase'])
print(np.mean(bestPhase))

#%% (2) check waveform: origin data
ori_chan = [45,51,52,53,54,55,58,59,60]
i = 0
data = testData
plt.plot(np.mean(data[0,:,ori_chan[i],1140], axis=0))
plt.plot(np.mean(data[1,:,ori_chan[i],1140], axis=0))

#%% (3) check waveform: SRCA data
ori_chan = [45,51,52,53,54,55,58,59,60]
i = 0
data = srcaTe
plt.plot(np.mean(data[0,:,i,:], axis=0))
plt.plot(np.mean(data[1,:,i,:], axis=0))

#%% (4) check acc SRCA
train_data = np.zeros((2,40,62,1340))
train_data[0,:,:,:] = trainData[0,:,:,:1340]
train_data[1,:,:,:] = trainData[2,:,:,15:1355]
test_data = np.zeros((2,80,62,1340))
test_data[0,:,:,:] = testData[0,:,:,:1340]
test_data[1,:,:,:] = testData[2,:,:,15:1355]

acc = mcee.SRCA_DCPM(train_data=train_data, test_data=train_data, tar_chans=tar_chans,
        model_chans=modelChans, chans=chans, regression='OLS', sp=1140, di=['1','2'])
acc = np.sum(acc)/(2*train_data.shape[1])

#%% (5) check acc Ori
train_data = np.zeros((2,9,40,200))
train_data[0,:,:,:] = trainData[0,:,[45,51,52,53,54,55,58,59,60],1140:1340]
train_data[1,:,:,:] = trainData[2,:,[45,51,52,53,54,55,58,59,60],1155:1355]
train_data = np.swapaxes(train_data, 1,2)
test_data = np.zeros((2,9,80,200))
test_data[0,:,:,:] = testData[0,:,[45,51,52,53,54,55,58,59,60],1140:1340]
test_data[1,:,:,:] = testData[2,:,[45,51,52,53,54,55,58,59,60],1155:1355]
test_data = np.swapaxes(test_data, 1,2)

acc = mcee.DCPM(train_data=train_data, test_data=train_data, di=['1','2'])
acc = np.sum(acc)/(2*train_data.shape[1])

#%% (5) check fisher score alteration
ori_chan = [45,51,52,53,54,55,58,59,60]
i = 0
x = mcee.fisher_score(testData[:, :, ori_chan[i], 1140:])
y = mcee.fisher_score(srcaTe[:, :, i, :])
plt.plot(x)
plt.plot(y)
plt.show()
z = (y.mean() - x.mean())/x.mean()*100

#%% (6) check fisher score alteration
eeg = io.loadmat(r'F:\SSVEP\realCV\mengqiangfan\FS\60Hz 50-70\train_40\loop_4\srca_4.mat')
para = eeg['parameter'].flatten().tolist()
model = eeg['modelInfo'].flatten().tolist()
tar_chans = ['PZ ','PO5','PO3','POZ','PO4','PO6','O1 ','OZ ','O2 ']

fig = plt.figure(figsize=(15,10))
gs = GridSpec(2,2,figure=fig)
sns.set(style='whitegrid')

ax1 = fig.add_subplot(gs[:,:])
ax1.set_title('Fisher Score Alteration', fontsize=24)
ax1.tick_params(axis='both', labelsize=20)
for i in range(len(tar_chans)):
    exec('ax1.plot(para[i].T, label=tar_chans[i], linewidth=1.5)')
ax1.set_xlabel('Number of iterations', fontsize=22)
ax1.set_ylabel('Fisher Score', fontsize=22)
ax1.legend(loc='upper right', fontsize=20)

fig.tight_layout()
plt.show()

#%% (7) check target identification algorithms step by step
data_path = r'F:\SSVEP\realCV\chengqian\FS\Cross identification\Cross 01\dcpm_result.mat'
result = io.loadmat(data_path)
ori = result['ori']
srca = result['srca']

#%% phase analysis
time = 0.1
sfreq = 1000
n_points = int(time*sfreq)
x = np.linspace(0, 2-2/100, 100)

f60 = lambda x:np.sin(2*np.pi*60*np.linspace(0, (n_points-1)/sfreq, n_points) + x*np.pi)
f48 = lambda x:np.sin(2*np.pi*48*np.linspace(0, (n_points-1)/sfreq, n_points) + x*np.pi)

corr = np.zeros((100))
for i in range(100):
    corr[i] = np.corrcoef(f60(0), f48(x[i]))[0,1]

plt.plot(corr)
index = np.max(np.where(corr == np.min(corr)))

print('best phase for 48Hz: ' + str(x[index]))
print('points need to shift: ' + str((x[index]/2) * (1000/48)))

#%%
eeg = io.loadmat(r'F:\SSVEP\dataset\20-30\a.mat')
mdata = np.swapaxes(eeg['a'], 0,2)
mdata = np.swapaxes(mdata, 1,3)
eeg = io.loadmat(r'F:\SSVEP\dataset\20-30\b.mat')
rdata = np.swapaxes(eeg['b'], 0,2)
rdata = np.swapaxes(rdata, 1,3)
del eeg

f_data = np.concatenate((rdata, mdata), axis=-1)[:,:,[48,54,55,56,57,58,61,62,63],:]

#%%
f_data = eeg['f_data'][:,:,:,:]
f60p0 = f_data[0,:,:,1200:1700]
f60p1 = f_data[1,:,:,1200:1700]
f48p0 = f_data[2,:,:,1140:1640]
f48p1 = f_data[3,:,:,1140:1640]
del eeg, f_data


corr1 = mcee.template_corr(f60p0[:,59,:], 1000, 60, 0)
corr2 = mcee.template_corr(f60p0[:,59,:], 1000, 60, 1)

message = '60 Hz pi & 60 Hz 0: ' + str(corr1.mean()) + '\n'
message += '60 Hz pi & 60 Hz pi: ' + str(corr2.mean()) + '\n'
print(message)

#%% new TRCA test
data_path = r'D:\SSVEP\dataset\preprocessed_data\60&80\zhaowei\fir_50_90.mat'
eeg = io.loadmat(data_path)
chans = eeg['chan_info'].tolist()
tar_data = eeg['f_data'][:2,:, [45,51,52,53,54,55,58,59,60], 1140:1540]
del eeg, data_path


from numpy import corrcoef as Corr
import time
from mcee import TRCA_off, TRCA_compute

test_data = tar_data[:, -60:, ...]
train_data = tar_data[:, :80, ...]
template = train_data.mean(axis=1)
n_events = tar_data.shape[0]

start1 = time.perf_counter()
w = mcee.TRCA_compute(tar_data[:, :80, ...])
r = np.zeros((n_events,60,n_events))
for nete in range(n_events):
    for nte in range(60):
        for netr in range(n_events):
            tp_test = np.dot(w[netr,:], test_data[nete,nte,...]).T
            tp_template = np.dot(w[netr,:], template[netr,...]).T
            r[nete, nte, netr] = np.sum(np.tril(Corr(tp_test.T, tp_template.T),-1))
acc1 = []
for ne in range(n_events):
    for nt in range(60):
        if np.max(np.where(r[ne, nt, :] == np.max(r[ne, nt, :]))) == ne:
            acc1.append(1)
acc1 = np.sum(acc1)/(n_events*60)*100
end1 = time.perf_counter()
print('New TRCA Running Time: ' + str(end1-start1) + 's')

start2 = time.perf_counter()
acc2 = np.sum(mcee.TRCA_off(train_data, test_data))/(n_events*60)*100
end2 = time.perf_counter()
print('Old TRCA Running Time: ' + str(end2-start2) + 's')


