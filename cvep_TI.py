# -*- coding: utf-8 -*-
"""
target identification module for code-VEP

author: Brynhildr W

"""
# %% Import third part module
import numpy as np
from numpy import newaxis as NA
import scipy.io as io
import mcee
import pandas as pd
import xlrd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
import copy

# %% TRCA/eTRCA for SRCA/origin data
tar_chans = ['PZ ','PO5','PO3','POZ','PO4','PO6','O1 ','OZ ','O2 ']
tar_chan_index = [45,51,52,53,54,55,58,59,60]
nameList = ['yangman', 'gaorunyuan', 'wuqiaoyi']
nameList = ['wuqiaoyi', 'gaorunyuan']

n_cv = 6

ori_trca = np.zeros((len(nameList), n_cv))
ori_strca = np.zeros((len(nameList), n_cv))
ori_etrca = np.zeros((len(nameList), n_cv))
ori_setrca = np.zeros((len(nameList), n_cv))

srca_trca = np.zeros((len(nameList), n_cv))
srca_etrca = np.zeros((len(nameList), n_cv))
srca_strca = np.zeros((len(nameList), n_cv))
srca_setrca = np.zeros((len(nameList), n_cv))

for n_peo in range(len(nameList)):
    people = nameList[n_peo]
    eeg = io.loadmat(r'D:\SSVEP\dataset\preprocessed_data\cvep_8\%s\fir_50_70.mat' %(people))
    f_data = eeg['f_data'][...,:2940]
    chans = eeg['chan_info'].tolist()
    n_events = f_data.shape[0]
    del eeg

    print("Now running " + people + "'s data...")

    srca_model = io.loadmat(r'D:\SSVEP\realCV\code_VEP\%s\e_pCORR_0.mat' %(people))
    model_info = srca_model['modelInfo'].flatten().tolist()
    model_chans = []
    for i in range(len(model_info)):
        model_chans.append(model_info[i].tolist())
    del model_info, i

    for cv in range(n_cv):
        print('CV: %d turn...' %(cv+1))
        test_list = [i+cv*10 for i in range(10)]
        # divide test/training dataset
        train_data = np.delete(f_data, test_list, axis=1)
        test_data = f_data[:,test_list,...]

        srca_train_data = np.zeros((8,50,9,1800))
        srca_test_data = np.zeros((8,10,9,1800))
        for i in range(n_events):
            srca_train_data[i,...] = mcee.apply_SRCA(train_data[i,...], tar_chans, model_chans, chans)
            srca_test_data[i,...] = mcee.apply_SRCA(test_data[i,...], tar_chans, model_chans, chans)
        
        train_data = train_data[...,tar_chan_index,1140:]
        test_data = test_data[...,tar_chan_index,1140:]
        # target identification
        _, ostrca = mcee.split_TRCA(600, train_data, test_data)
        otrca = mcee.TRCA(train_data, test_data)
        _, osetrca = mcee.split_eTRCA(600, train_data, test_data)
        oetrca = mcee.eTRCA(train_data, test_data)
        _, sstrca = mcee.split_TRCA(600, srca_train_data, srca_test_data)
        strca = mcee.TRCA(srca_train_data, srca_test_data)
        _, ssetrca = mcee.split_eTRCA(600, srca_train_data, srca_test_data)
        setrca = mcee.eTRCA(srca_train_data, srca_test_data)

        # save acc data
        ori_trca[n_peo, cv] = otrca
        ori_etrca[n_peo, cv] = oetrca
        ori_strca[n_peo, cv] = ostrca
        ori_setrca[n_peo, cv] = osetrca
        srca_trca[n_peo, cv] = strca
        srca_etrca[n_peo, cv] = setrca
        srca_strca[n_peo, cv] = sstrca
        srca_setrca[n_peo, cv] = ssetrca

        print(str(cv+1) + 'th cross-validation complete!\n')





# %%
