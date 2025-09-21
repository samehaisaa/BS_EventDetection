import numpy as np
from scipy.signal import butter, filtfilt, hilbert

def bandpass(x, sr, low, high, order=4):
    ny = 0.5*sr
    b, a = butter(order, [low/ny, high/ny], btype="band")
    return filtfilt(b, a, x)

def envelope(x, sr, smooth_ms=10):
    a = np.abs(hilbert(x))
    n = max(1, int(smooth_ms*1e-3*sr))
    k = np.ones(n)/n
    return np.convolve(a, k, mode="same")
