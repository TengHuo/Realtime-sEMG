# -*- coding: UTF-8 -*-
"""
*** The CSL-HDEMG dataset ***

** Overview **

The dataset contained in this archive is described in detail in the paper:

Advancing Muscle-Computer Interfaces with High-Density Electromyography
Christoph Amma, Thomas Krings, Jonas Böer, Tanja Schultz
CHI 2015

Please cite this paper in case you publish any work using this dataset.

The data set contains EMG data from five different subjects. Each subject
contributed five sessions recorded on different days. The data set consists of
27 different gestures and every gesture was repeated 10 times per session
(excluding the idle gesture which was repeated 30 times).

The dataset is organized as follows: There is a folder for each subject
containing a folder for each session containing a .mat file for each gesture
performed. The .mat file contains a cell array named 'gestures' with an entry
for each repetition of the given gesture

The data for every trial is saved as an 192xL matrix. L is the length of the
repetition (3 seconds sampled with 2048 Hz). It may vary by a few milliseconds.
Each row represents one channel. Every eighth channel does not contain
meaningful data due to the differential amplification in bipolar recordings and
should be ignored.

The first channel is the differential signal of the electrodes 1 and 2, the
second is the differential signal of the electrodes 2 and 3 and so on.
Electrodes 1, 9, 17, ..., 185 etc are located on the proximal end, and
electrodes 8, 16, 24, ..., 192 are located on the distal end.  Further
description can be obtained from the the paper.


** Accessing the data **

In the "src" folder, you find two example scripts that read and plot data. We
provide Matlab and Python code. The scripts are named example.py and example.m

The example code loads gesture 11 (bending of the ring finger) from the third
session of subject 5. It computes the average root-mean-square for each channel
over the whole 3 second duration and plots it as a heat map. Additionally it
plots the raw data of an exemplary single channel for the 3 second duration.
"""
# example code to load data from the csl-hdemg dataset

import matplotlib.pyplot as plt
import scipy.io as sio
import numpy as np
from math import sqrt

# load the data
mat = sio.loadmat('../subject5/session3/gest11.mat')
gestures = mat['gestures']

#compute RMS
rms = np.zeros(168)
for i in range(0,10):
 trial = gestures[i,0]
 #deleting edge channels
 trial = np.delete(trial,np.s_[7:192:8],0)
 for c in range(0,trial.shape[0]):
     #computing mean rms over all repetitions
     rms[c] += np.linalg.norm(trial[c,:]) / sqrt(len(trial[c,:])) / 10

#reshaping to the correct shape
rms = np.reshape(rms,(24,7))
rms = np.flipud(np.transpose(rms))

#plot data
plt.subplot(211)
plt.imshow(rms, cmap='hot_r', interpolation='nearest', vmin=0, vmax=0.0035)
plt.axis('off')
plt.title('Average RMS over all repetitions')
plt.subplot(212)
plt.plot(gestures[9,0][164,:])
plt.title('Exemplary plot of a single channel (channel 165)')
plt.show()
