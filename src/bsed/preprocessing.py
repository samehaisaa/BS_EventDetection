from __future__ import annotations
from typing import Optional, Tuple
import numpy as np
import librosa
from scipy.signal import butter, sosfiltfilt

def pre_emphasis(y: np.ndarray, coeff: float) -> np.ndarray:
    if coeff <= 0:
        return y
    out = np.empty_like(y)
    out[0] = y[0]
    out[1:] = y[1:] - coeff * y[:-1]
    return out

def highpass(y: np.ndarray, sr: int, cutoff_hz: float) -> np.ndarray:
    if cutoff_hz <= 0:
        return y
    sos = butter(2, cutoff_hz, btype="highpass", fs=sr, output="sos")
    return sosfiltfilt(sos, y)

def logmel(y: np.ndarray, sr: int, n_mels: int, n_fft: int, hop_length: int,
           fmin: float, fmax: Optional[float], ref_level_db: float, top_db: float,
           normalize: str = "instance") -> Tuple[np.ndarray, int]:
    S = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels,
        fmin=fmin, fmax=fmax or sr//2, power=2.0
    )
    S_db = librosa.power_to_db(S, ref=ref_level_db, top_db=top_db)
    if normalize == "instance":
        m = np.mean(S_db, axis=None)
        s = np.std(S_db, axis=None) + 1e-8
        S_db = (S_db - m) / s
    return S_db.astype(np.float32), hop_length
