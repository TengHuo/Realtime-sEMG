# -*- coding: UTF-8 -*-
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
