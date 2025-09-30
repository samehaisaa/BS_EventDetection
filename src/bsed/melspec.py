import numpy as np, librosa

def logmel_segment(y_seg, sr, n_mels=128, win_ms=25.0, hop_ms=10.0):
    win_len = int(round((win_ms/1000.0) * sr))
    hop_len = int(round((hop_ms/1000.0) * sr))
    n_fft   = win_len

    S = librosa.feature.melspectrogram(
        y=y_seg, sr=sr, n_fft=n_fft, hop_length=hop_len,
        win_length=win_len, window="hann", center=False,
        power=2.0, n_mels=n_mels
    )
    S_db = librosa.power_to_db(S, ref=np.max)
    mu, sigma = S_db.mean(), (S_db.std() if S_db.std() > 0 else 1.0)
    Z = (S_db - mu) / sigma

    expected_frames = 1 + int(np.floor(((len(y_seg) - win_len) / hop_len)))
    if Z.shape[1] != expected_frames:
        print(f"[WARN] frames {Z.shape[1]} != expected {expected_frames}")
    return Z
