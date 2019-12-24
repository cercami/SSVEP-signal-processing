# -*- coding: utf-8 -*-
"""
Created on Tue Dec 24 15:46:50 2019

Task-Related Component Analysis

@author: Brynhildr
"""

#%% Import third part module
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
import mcee

import copy

#%% Load data
eeg = io.loadmat(r'F:\SSVEP\dataset\preprocessed_data\weisiwen\f_data.mat')
f_data = eeg['f_data']
chans = eeg['chan_info'].tolist()
f_data *= 1e6  

del eeg

sfreq = 1000

#w = f_data[:, :, :, 2000:3000]
signal_data = f_data[:, :, :, 3000:3500]   

del f_data


#%% Matrix Q

#%% Matrix S

#%% Square Q^-1*S

#%% Eigen vector w

#%% 
