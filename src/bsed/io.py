import numpy as np
import librosa
from scipy.signal import butter, sosfiltfilt

def load_audio(path, sr=None):
    y, sr = librosa.load(path, sr=sr, mono=True)
    return y, sr

def highpass_biquad(y, sr, cutoff_hz=60.0):
    sos = butter(N=2, Wn=cutoff_hz, btype="highpass", fs=sr, output="sos")
    return sosfiltfilt(sos, y)

def split_nonoverlapping_segments(y, sr, seg_dur_s=10.0):
    seg_len = int(round(seg_dur_s * sr))
    n_full = len(y) // seg_len
    if n_full == 0:
        return []
    y = y[: n_full * seg_len]
    return [y[i*seg_len:(i+1)*seg_len] for i in range(n_full)]
