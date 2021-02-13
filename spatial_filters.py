# -*- coding: utf-8 -*-
"""
@ author: Brynhildr Wu
@ email: brynhildrwu@gmail.com

A toolbox for constructing commonly used spatial filters based on Unified Framework,
    and other spatial filters with special design.
Main Refer: 
    [1] Wong C M, et al. Spatial Filtering in SSVEP-based BCIs: Unified Framework and New Improvements[J].
        IEEE Transactions on Biomedical Engineering, 2020, PP(99):1-1.
    [2] 

Requirements:
1. data format: 
    (1) balanced sample
        4-D data: (n_events, n_trials, n_chans, n_times)
    (2) unbalanced sample
        3-D data: (n_trials, n_chans, n_times)
        1-D label: (labels,)

Prefunctions:
1. zero_mean: (Non-sensitive to sample balance, i.e. 'non-sensitive' in the following)
    zero-mean normalization (if necessary). 
        Generally speaking, the data preprocessed by bandpass filtering from MNE has been
        zero-averaged already. Well, who knows how you guys preprocess the EEG data?
2. corr_coef: (non-sensitive)
    compute Pearson Correlation Coefficient
3. sinw:
    make a piece of sinusoidal wave
4. real_phase:
    compute best initial phase for training dataset
5. time_shift:
    cyclic rotate time sequence to destroy the time correlation of noise
6. pearson_corr2: 
    compute 2-D Pearson coefficient
7. Imn:
    concatenate multiple unit matrices vertically
8. diag_splice:
    the operator for diagonal concatenation

Special design:
1. TRCA_compute:
    special design for TRCA algorithm

The Unified Framework:
1. spatial_filter (W): class
    framework 1: (Z.T)*D*P*(P.T)*(D.T)*Z*W = Z.T*D*D.T*Z*W*Lamda
    framework 2: (Z.T)*D*P*(P.T)*(D.T)*Z*W = W*Lamda

Target identification functions
1. CCA series (Canonical Correlation Analysis):
    sCCA, itCCA, ttCCA, extended-CCA, fbCCA, MsetCCA, msCCA
2. TRCA series (Task-Related Component Analysis):
    origin: TRCA, eTRCA
    (add sin-cos reference): TRCA-R, eTRCA-R
    (split in time-domain): split-...
    (add filter banks): fb-...

(Not included in unified framework)
3. DCPM series:
    DSP, DCPM, PCA-DCPM
4. LDA series:
    LDA, stepwise-LDA, SKLDA, BLDA, STDA
5. Latest algorithms:(unstable)

update: 2021/2/8

"""

# %% basic modules
import numpy as np
from numpy import newaxis as NA
from numpy import linalg as LA
from numpy import (sin, cos, pi, sqrt, diagonal)
from sympy import diag
from sklearn import linear_model

from copy import deepcopy
from time import perf_counter
import scipy.io as io
import matplotlib.pyplot as plt

# %% Prefunctions
def zero_mean(data):
    """
    Parameters
    ----------
    data : ndarray, (n_trials, n_channels, n_times)
        input data array.

    Returns
    -------
    data : ndarray, (n_trials, n_channels, n_times)
        data after zero-mean normalization.
    """
    data -= data.mean(axis=-1, keepdims=True)

    return data

def corr_coef(X, y):
    """
    Parameters
    ----------
    X : ndarray, (..., n_points)
        input 2-D data array. (could be sequence)
    y : ndarray, (1, n_points)
        input data vector/sequence

    Returns
    -------
    corrcoef : float
        Pearson's Correlation Coefficient.
    """
    X, y = zero_mean(X), zero_mean(y)
    cov_yx = y @ X.T
    # 'int' object has no attribute 'ndim', but the dimension of 'float' object is 0
    if cov_yx.ndim == 0:
        var_xx = sqrt(X @ X.T)
    else:
        var_xx = sqrt(diagonal(X @ X.T))
    var_yy = sqrt(float(y @ y.T))
    corrcoef = cov_yx / (var_xx*var_yy)

    return corrcoef.mean()

def sinw(freq, time, phase, sfreq=1000):
    """
    Parameters
    ----------
    freq : float
        frequency / Hz.
    time : float
        time length / s.
    phase : float
        0-2.
    sfreq : float/int, optional
        sampling frequency. The default is 1000.

    Returns
    -------
    wave : ndarray, (time*sfreq,)
        sinusoidal sequence.
    """
    n_point = int(time*sfreq)
    time_point = np.linspace(0, (n_point-1)/sfreq, n_point)
    wave = sin(2*pi*freq*time_point + pi*phase)

    return wave

def real_phase(data, freq, step=100, sfreq=1000):
    """
    Parameters
    ----------
    data : ndarray, (n_trials, n_points)
    freq : float
        frequency of template.
    step : int, optional
        stepwidth = 1 / steps. The default is 100.
    sfreq : float, optional
        sampling frequency of signal. The default is 1000.

    Returns
    -------
    phase : float
        0-2.
    """
    time = data.shape[-1] / sfreq
    step = [2*x/step for x in range(step)]
    corr = np.zeros((step))

    signal_mean = data.mean(axis=0, keepdims=True)  # (1, n_points)
    for i in range(step):
        template = sinw(freq, time, step[i], sfreq)
        corr[i] = corr_coef(signal_mean, template[NA,:])
    phase = np.argmax(corr)*2/100

    return phase

def time_shift(data, step, axis=None):
    """
    Parameters
    ----------
    data : ndarray, (n_channels, n_points)
        Input data array.
    step : int/float
        The length of the scroll. 
    axis : int or tuple of ints
        Dimension of scrolling, 0 - vertical, 1 - horizontal.
        By default(None) , the array will be flattened before being shifted,
            and then restored to its original shape.

    Returns
    -------
    tf_data : ndarray, (n_channels, n_points)
        Data containing background EEG with corrupted temporal and spatial correlation.
    """
    n_chans = data.shape[0]
    tf_data = np.zeros_like(data)
    tf_data[0, :] = data[0, :]
    for i in range(n_chans-1):
        tf_data[i+1, :] = np.roll(data[i+1, :], shift=round(step*(i+1)), axis=axis)

    return tf_data

def pearson_corr2(data_A, data_B):
    """

    Parameters
    ----------
    data_A : ndarray, (n_chans, n_times)
    data_B : ndarray, (n_chans, n_times)

    Returns
    -------
    corr2 : float
        2-D correlation coefficient.
    """
    mean_A = data_A.mean()
    mean_B = data_B.mean()
    numerator = np.sum((data_A-mean_A) * (data_B-mean_B))
    denominator_A = np.sum((data_A-mean_A)**2)
    denominator_B = np.sum((data_B-mean_B)**2)
    corr2 = numerator / sqrt(denominator_A*denominator_B)

    return corr2

def Imn(m,n):
    """

    Parameters
    ----------
    m : int
        Total number of identity matrix.
    n : int
        Dimensions of the identity matrix.
            
    Returns
    -------
    target : ndarray, (m*n, n)

    """
    target, unit = np.eye(n), np.eye(n)
    i = 0
    for i in range(m-1):
        target = np.vstack((target, unit))
        i += 1
        
    return target

def diag_splice(*arg):
    """

    Parameters
    ----------
    arg : 2-D ndarray, (dimension_x, n_times)
        Various data matrices, total number(assumed M) is unlimited.
        n_times should be equal.

    Returns
    -------
    target : ndarray, (sum(dimension_x), M*n_times)
        Diagonal mosaic matrix.

    """
    target = 0
    for data in arg:
        target = diag(target, data)
    target.col_del(0).row_del(0)
    target = np.array(target, dtype=float)

    return target   

    
# %% The Unified Framework
class spatial_filter:
    """
    The Unified Framework for constructing spatial filters used in SSVEP signal processing:
        Type I : (Z.T)*D*P*(P.T)*(D.T)*Z*W = (Z.T)*D*(D.T)*Z*W*Lambda
        Type II : (Z.T)*D*P*(P.T)*(D.T)*Z*W = W*Lambda
    The data matrix has been preprocessed by time-domain filtering by default, so D = E, i.e.
        Type I : (Z.T)*P*P.T*Z*W = (Z.T)*Z*W*Lambda
        Type II : (Z.T)*P*P.T*Z*W = W*Lambda

    """
    def __init__(self, train_data, subject=None):
        """

        Parameters
        ----------
        train_data : ndarray, (n_events, n_trials, n_chans, n_times)
            input data array (default z-scored after bandpass filtering).
        subject : dict, {'name':[num1,num2], ...}
            Dictation to describe the trial subscript of each participant.
            If None, i.e. single subject's data. The default is None.
        """
        # basic information
        self.ori_data = train_data

        self.n_events = train_data.shape[0]
        self.n_trains = train_data.shape[1]
        self.n_chans = train_data.shape[2]
        self.n_times = train_data.shape[-1]

        if subject is not None:
            for sub in subject:
                start_point, end_point = subject[sub][0], subject[sub][1]
                sub_data = train_data[..., start_point:end_point]
                exec("self.train_data_%s = sub_data" %sub)

        # check if some attribute exist
        # if hasattr(self, 'attribute'):


    def framework(self, filter_core=None, filter_type=None, **kwargs):
        """
        Choose different frameworks for different spatial filter schemes.
        Determine the form of the data matrix and projection matrix.
        Parameters
        ----------
        filter_core : str
            Filter series. Now supported: TRCA, CCA, LDA, DCPM
        filter_type : list of str
            Specific filter model. Now supported:
            TRCA : origin, ensemble | reference | split, filter bank
            CCA : origin, individual template, transform template | extended, multi-set |
                    split, filter bank, 
            LDA : origin, stepwise, bayesian, shrinkage, spatial-time
            DCPM : origin, PCA, 
        *args : list of str
            Additional parameters according to specific needs

        """
        # initialise
        self.train_data = train_data
        self.matrix_A = np.zeros((self.n_events, self.n_chans, self.n_chans))
        self.matrix_B = np.zeros_like(self.matrix_A)
        self.template = np.zeros((self.n_events, self.n_chans, self.n_times))
        self.frame_type = '1'

        if filter_core not in self.core_list:
            raise Exception('Not supported filter core: ' + filter_core)
        elif filter_type not in self.filter_list:
            raise Exception('Not supported spatial filters: ' + filter_type)

        if filter_core == 'TRCA':
            # detection order: 
            # (1) whether time domain segmentation is required
            # (2) whether to include a filter bank
            # (3) which type: origin, ensemble or reference
            self.special_design = True
            if 'reference' in filter_type:
                pass

            if 'origin' or 'ensemble' in filter_type:
                # same in filter constructing, different in target identification
                for ne in range(self.n_events):
                    temp = self.train_data[ne, ...].swapaxes(0,1).reshape((self.n_chans,-1), order='C')
                    self.matrix_A[ne, ...] = (temp@temp.T) / (self.n_trains*self.n_times)
                    for i in range(self.n_trains):
                        for j in range(self.n_trains):
                            if i != j:
                                data_i = self.train_data[ne, i, ...]
                                data_j = self.train_data[ne, j, ...]
                                self.matrix_B[ne, ...] += (data_i@data_j.T) / self.n_times
                self.template = self.train_data.mean(axis=1)
                if 'ensemble' in filter_type:
                    self.ensemble = True


        elif filter_series == 'CCA':
            # detection order: 
            # (1) whether time domain segmentation is required
            # (2) whether to include a filter bank
            # (3) which specific type

            pass


        elif filter_series == None:
            pass
        
        # deal with unknown options or errors
        if self.matrix_A == self.matrix_B:  # nothing happened or something wrong
            raise Exception('Unknown Error while constructing framework, maybe not supported combination?')


    def build_filter(self):
        """
        Solve Generalized Eigenvalue Problems(GEPS): A*W = B*W*Lambda
        Parameters
        ----------
        data : ndarray, basic units: (n_chans, n_times)
            matrix Z, 
        projection : ndarray, basic units: (n_times, n_times)
            matrix P, orthogonal projector, Hermite matrix.
        frame_type : int
            1 or 2. For details, please refer to the corresponding literature.

        """
        # initialization
        self.w = np.zeros((self.n_events, self.n_chans))
        # unified framework
        if self.special_design:
            matrix_A = self.matrix_A
            matrix_B = self.matrix_B
        else:
            if self.split == True:
            for ne in range(self.n_events):
                data = self.combine_data[ne, ...]
                projection = self.projection[ne, ...]
                matrix_A[ne, ...] = data.T @ projection @ projection.T @ data
                if self.frame_type == '1':
                    matrix_B[ne, ...] = data.T @ data
                elif self.frame_type == '2':  # B = E
                    matrix_B[ne, ...] = np.eye(matrix_A.shape[0])
        # solve Generalized Eigenvalue Problems(GEPS)
        for ne in range(self.n_events):
            e_va, e_vec = LA.eig(LA.inv(matrix_A[ne, ...]) @ matrix_B[ne, ...])
            w_index = np.argmax(e_va)
            self.w[ne, :] = e_vec[:, w_index].T


# %% DCPM Series
