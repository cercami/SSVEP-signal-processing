# -*- coding: utf-8 -*-
"""
Created on Sun Jan 19 20:10:16 2020

TRCA method without filter band

@author: Brynhildr
"""

#%% Import third part module
import numpy as np
import scipy.io as io
from scipy import signal

import copy

import mcee

import matplotlib.pyplot as plt
import seaborn as sns

#%% load mcee data
eeg = io.loadmat(r'I:\SSVEP\dataset\preprocessed_data\wuqiaoyi\60_b_140ms\mcee_0.mat')
tar_chans = eeg['chan_info'].tolist()
data = eeg['mcee_sig'][:,:,:,1140:1240]
del eeg

