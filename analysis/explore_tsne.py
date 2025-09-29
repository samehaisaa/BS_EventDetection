#!/usr/bin/env python3
import numpy as np
import pandas as pd
import librosa
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def _safe_delta(X, max_width=9):
    if X.ndim != 2: raise ValueError
    T = X.shape[1]
    if T < 3: return np.zeros_like(X)
    width = min(max_width, T if (T % 2 == 1) else T - 1)
    if width < 3: return np.zeros_like(X)
    return librosa.feature.delta(X, width=width, mode='interp')

def load_labels(p):
    out = []
    with open(p, "r") as f:
        for ln in f:
            a = ln.strip().split()
            if len(a) >= 3:
                out.append({"onset": float(a[0]), "offset": float(a[1]), "label": a[2]})
    return out

def segment_feats(y, sr, n_fft=400, hop=160, n_mfcc=13):
    if len(y) < n_fft + 8*hop:
        pad = n_fft + 8*hop - len(y)
        y = np.pad(y, (0, pad), mode="reflect" if len(y) > 1 else "constant")
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=n_fft, hop_length=hop)[0].mean() if len(y) else 0.0
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop)) + 1e-20
    sc = librosa.feature.spectral_centroid(S=S, sr=sr, hop_length=hop)[0].mean()
    sf = librosa.feature.spectral_flatness(S=S, hop_length=hop)[0].mean()
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop)
    d1 = _safe_delta(mfcc)
    d2 = _safe_delta(d1)
    v = {
        "duration_ms": 1000.0 * len(y) / sr,
        "zcr": float(zcr),
        "spec_centroid": float(sc),
        "spec_flatness": float(sf),
    }
    for i in range(n_mfcc):
        v[f"mfcc_{i}_mean"] = float(mfcc[i].mean())
        v[f"dmfcc_{i}_mean"] = float(d1[i].mean())
        v[f"ddmfcc_{i}_mean"] = float(d2[i].mean())
    return v

def process_pair(wav_path, txt_path, sr=16000, n_fft=400, hop=160):
    y, _ = librosa.load(str(wav_path), sr=sr)
    labs = load_labels(str(txt_path))
    rows = []
    for l in labs:
        a = int(l["onset"] * sr); b = int(l["offset"] * sr)
        seg = y[a:b]
        if seg.size == 0: continue
        f = segment_feats(seg, sr, n_fft=n_fft, hop=hop)
        f["file"] = wav_path.stem
        f["label"] = l["label"]
        rows.append(f)
    return rows

def main():
    audio_dir = Path("./dige")
    label_dir = Path("./dige")
    feats = []
    for wav in audio_dir.glob("*.wav"):
        txt = label_dir / f"{wav.stem}.txt"
        if txt.exists():
            feats.extend(process_pair(wav, txt))
    if not feats: return
    df = pd.DataFrame(feats)
    feature_cols = [c for c in df.columns if c not in ["label", "file"]]
    X = df[feature_cols].values
    y = df["label"].values
    X = StandardScaler().fit_transform(X)
    Xp = PCA(n_components=min(50, X.shape[1]), random_state=42).fit_transform(X)
    Xt = TSNE(n_components=2, perplexity=30, random_state=42, n_iter=1000).fit_transform(Xp)
    classes = sorted(pd.unique(y))
    plt.figure(figsize=(7,6))
    for cls in classes:
        m = y == cls
        plt.scatter(Xt[m,0], Xt[m,1], s=24, alpha=0.75, label=str(cls), edgecolors="k", linewidths=0.3)
    plt.xlabel("t-SNE 1"); plt.ylabel("t-SNE 2"); plt.title("t-SNE of Bowel Sound Features"); plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
