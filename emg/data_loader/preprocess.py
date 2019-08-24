# -*- coding: UTF-8 -*-
# preprocess.py
# @Time     : 27/Mar/2019
# @Author   : TENG HUO
# @Email    : teng_huo@outlook.com
# @Version  : 1.0.0
# @License  : MIT
#
#

import numpy as np

from scipy.ndimage.filters import median_filter
from scipy.signal import butter, lfilter


def _butter_bandpass_filter(emg_data, lowcut, highcut, fs, order):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq

    b, a = butter(order, [low, high], btype='bandpass')
    y = lfilter(b, a, emg_data)
    return y


def _continuous_segments(label):
    label = np.asarray(label)

    if not len(label):
        return

    breaks = list(np.where(label[:-1] != label[1:])[0] + 1)
    for begin, end in zip([0] + breaks, breaks + [len(label)]):
        assert begin < end
        yield begin, end


def _csl_cut(emg_data, framerate):
    window = int(np.round(150 * framerate / 2048))
    emg_data = emg_data[:len(emg_data) // window * window].reshape(-1, 150, emg_data.shape[1])
    rms = np.sqrt(np.mean(np.square(emg_data), axis=1))
    rms = [median_filter(image, 3).ravel() for image in rms.reshape(-1, 24, 7)]
    rms = np.mean(rms, axis=1)
    threshold = np.mean(rms)
    mask = rms > threshold
    for i in range(1, len(mask) - 1):
        if not mask[i] and mask[i - 1] and mask[i + 1]:
            mask[i] = True
    begin, end = max(_continuous_segments(mask),
                     key=lambda s: (mask[s[0]], s[1] - s[0]))
    return begin * window, end * window


def csl_preprocess(trial):
    trial = np.delete(trial, np.s_[7:192:8], 0)
    trial = trial.T

    # bandpass
    trial = np.transpose([_butter_bandpass_filter(ch, 20, 400, 2048, 4) for ch in trial.T])
    # cut
    begin, end = _csl_cut(trial, 2048)
    # print(begin, end)
    trial = trial[begin:end]
    # median filter
    trial = np.array([median_filter(image, 3).ravel() for image in trial.reshape(-1, 24, 7)])
    return trial


if __name__ == "__main__":
    pass
