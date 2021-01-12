
# -*- coding: utf-8 -*-


# %%
import numpy as np
from numpy import newaxis as NA
from numpy import (sin, cos, pi)
# from sCCA import sCCA
from scipy import io
import matplotlib.pyplot as plt
import mcee

# %%
# data_dit = io.loadmat('zhaowei_predata_all_500')  #字典类型
data_dit = io.loadmat('zhaowei_predata_data_all.mat')  #字典类型
data = data_dit['matrix']
# delete？
n_events = data.shape[0]
n_trials = data.shape[1]
n_chans = data.shape[2]
n_times = data.shape[-1]

acc_ori_CCA = np.zeros((5, 5))

ns = 80
print("Training trials: " + str(ns))
for cv in range(5):       # 交叉验证
    print('CV: %d turn...' %(cv+1))
    # randomly pick channels for identification
   # randpick = np.arange(data.shape[1])
   # np.random.shuffle(randpick)
    for n_t in range(5):
        # -60 - cv * 15: -cv * 15 - 1
        print('Data length: %d00ms'%(n_t+1))
        train_data = data[:, cv*15:cv*15+80, :, 0:100+n_t*100]  # 把前N个试次作为训练集，剩下的作为测试集
        orin_index = np.arange(144)
        mask_index = np.arange(cv * 15, cv * 15 + 80)
        mask = np.in1d(orin_index, mask_index)
        mask_array = orin_index[~mask]  # 生成掩码式矩阵
        test_data = data[:, mask_array, :, 0:100 + n_t * 100]  # test data:(n_events, n_test_trails, n_chans, n_times)
        n_events = train_data.shape[0]
        template = train_data.mean(axis=1)  # template data: (n_events, n_chans, n_times)
        n_tests = test_data.shape[1]
        r = np.zeros((n_events, n_tests, n_events))
        # target identification
        for ne in range(n_events):                       # test data 中按类别取出数据
            for nte in range(n_tests):                 # test data 中每一类别有多个trials
                for netr in range(n_events):             # n_events in training dataset  一个test trial 和所有类别的W计算相关系数
                    temp_test = test_data[ne, nte, ...]  # (N_chans, N_points)
                    temp_template = template[netr, ...]  # (N_chans, N_points)
                    r[ne, nte, netr] = sCCA(temp_test, temp_template)  # 所有trails都会得出一个corrcoef, 存入netr中
        accuracy = []
        # compute accuracys
        for ne in range(n_events):
            for nt in range(n_tests):
                if np.argmax(r[ne, nt, :]) == ne: # 所有trials得出的最大corrcoef的索引(预测的类别)是否等于原标签
                    accuracy.append(1)
        accuracy = np.sum(accuracy) / (n_events * n_tests)
        acc_ori_CCA[cv, n_t] = accuracy
        print(str(cv+1)+'th cross-validation complete!\n')
    print(str(ns)+'training trials complete!\n')
    print('acc_ori_CCA:', acc_ori_CCA.mean(axis=0))       # 按列求平均
    print('acc_ori_CCA:', np.std(acc_ori_CCA,axis=0))

def sCCA(X,Y):
    """

    Parameters
    -----------
    X: (N_chans, N_points)
        input data array (default z-scored after bandpass filtering)
    Y: (2*N_harmos, N_points)
        input data array (default z-scored after bandpass filtering)

    Returns
    ---------
    corrcoef:float
    """

    # 转置
    st_X = X.T                          # Np x Nc
    st_Y = Y.T                          # Np x 2Nh

    # 计算协方差
    C_XX = np.dot(st_X.T,st_X)          # NcxNp,NpxNc = NcxNc
    C_YY = np.dot(st_Y.T,st_Y)          # 2NhxNp,Npx2Nh = 2Nhx2Nh
    C_XY = np.dot(st_X.T,st_Y)          # NcxNp,Npx2Nh = Ncx2Nh
    C_YX = np.dot(st_Y.T,st_X)          # 2NhxNp,NpxNc = 2NhxNc

    # 求C_XX、C_YY的逆矩阵
    C_XX_Reverse = np.linalg.inv(C_XX)  # NcxNc
    C_YY_Reverse = np.linalg.inv(C_YY)  # NhxNh

    # 求X的系数特征值的方程
    X_Feature_mat = np.dot(np.dot(C_XX_Reverse, C_XY), np.dot(C_YY_Reverse, C_YX))  # 特征方程 NcxNcxNcx2Nhx2Nhx2Nhx2NhxNc
    eigvals_X, eigvectors_X = np.linalg.eig(X_Feature_mat)
    X_coff = eigvectors_X[:, np.argmax(eigvals_X)]                                   # 计算最大特征值对应的特征向量，列向量

    # 求Y的系数特征值的方程
    Y_Feature_mat = np.dot(np.dot(C_YY_Reverse, C_YX), np.dot(C_XX_Reverse, C_XY))  # 特征方程 NcxNcxNcx2Nhx2Nhx2Nhx2NhxNc
    eigvals_Y, eigvectors_Y = np.linalg.eig(Y_Feature_mat)
    Y_coff = eigvectors_Y[:, np.argmax(eigvals_Y)]                                    # 计算最大特征值对应的特征向量，列向量

    # 计算相关系数  np.corrcoef
    Trans_X = np.dot(X_coff.T, st_X.T)              # 1xNcxNcxNp
    Trans_Y = np.dot(Y_coff.T, st_Y.T)              # 1xNcxNcxNp
    corrcoef_XY = np.corrcoef(Trans_X.T, Trans_Y.T)
    corrcoef_XY = corrcoef_XY[0, 1]                 # 计算出的相关系数是一个矩阵

    return corrcoef_XY


# %%
eeg = io.loadmat(r'D:\SSVEP\dataset\preprocessed_data\60&80\zhaowei\fir_50_90.mat')
f_data = eeg['f_data'][[1,3],...]

tar_list = [45,51,52,53,54,55,58,59,60]

test = f_data[:,:,tar_list,1140:1640]
acc1 = mcee.sCCA(test, [60,80], 1)
acc2 = mcee.itCCA(test[:,:80,...], test[:,-60:,...])

# %%
