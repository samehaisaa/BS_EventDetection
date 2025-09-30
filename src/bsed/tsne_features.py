from pathlib import Path
import numpy as np, pandas as pd, librosa, scipy.stats
from scipy import signal
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt, seaborn as sns
from tqdm.auto import tqdm

def _safe_delta(X: np.ndarray, max_width: int = 9) -> np.ndarray:
    if X.ndim != 2: raise ValueError("Expected 2D array")
    T = X.shape[1]
    if T < 3: return np.zeros_like(X)
    width = min(max_width, T if (T % 2 == 1) else T - 1)
    if width < 3: return np.zeros_like(X)
    return librosa.feature.delta(X, width=width, mode='interp')

def _normalize_label(raw: str) -> str:
    s = (raw or "").strip().lower()
    if s == "b": return "sb"
    if s in {"sb","mb","h"}: return s
    return s

class BowelSoundFeatureExtractor:
    def __init__(self, sr=16000, fft_size=400, hop_size=160):
        self.sr = sr; self.fft_size = fft_size; self.hop_size = hop_size; self.n_mfcc = 13
        self.freq_bands = [(0,100),(100,200),(200,400),(400,800),(800,1500),(1500,3000),(3000,8000)]

    def extract_temporal_features(self, audio_segment: np.ndarray) -> dict:
        f = {}
        f['duration_ms'] = len(audio_segment)/self.sr*1000.0
        f['rms'] = float(np.sqrt(np.mean(audio_segment**2) + 1e-20))
        f['peak_amplitude'] = float(np.max(np.abs(audio_segment)) if audio_segment.size else 0.0)
        f['crest_factor'] = float(f['peak_amplitude']/(f['rms']+1e-20))
        zcr = librosa.feature.zero_crossing_rate(audio_segment, frame_length=self.fft_size, hop_length=self.hop_size)[0]
        f['zcr'] = float(np.mean(zcr)) if zcr.size else 0.0
        envelope = np.abs(signal.hilbert(audio_segment)); time_axis = np.arange(len(envelope))/self.sr
        denom = float(np.sum(envelope)+1e-20); f['temporal_centroid'] = float(np.sum(time_axis*envelope)/denom)
        env = envelope
        if len(env) > 1:
            win = min(51, (len(env)//4)*2+1); env_smooth = signal.savgol_filter(env, win, 3) if win >= 3 else env
        else:
            env_smooth = env
        max_env = float(np.max(env_smooth) if env_smooth.size else 0.0)
        if max_env > 0:
            rise_10 = np.where(env_smooth >= 0.1*max_env)[0]; rise_90 = np.where(env_smooth >= 0.9*max_env)[0]
            f['attack_time_ms'] = ((rise_90[0]-rise_10[0])/self.sr*1000.0) if (rise_10.size and rise_90.size and rise_90[0]>=rise_10[0]) else 0.0
        else:
            f['attack_time_ms'] = 0.0
        gmean = float(scipy.stats.gmean(env + 1e-20)) if env.size else 0.0
        mean_env = float(np.mean(env)+1e-20); f['temporal_flatness'] = float(gmean/mean_env)
        prom = 0.3*max_env if max_env>0 else 0.0
        peaks,_ = signal.find_peaks(env_smooth, prominence=prom, distance=int(0.01*self.sr))
        f['num_peaks'] = int(len(peaks))
        return f

    def extract_spectral_features(self, audio_segment: np.ndarray) -> dict:
        f = {}
        D = librosa.stft(audio_segment, n_fft=self.fft_size, hop_length=self.hop_size); S = np.abs(D)+1e-20
        freqs = librosa.fft_frequencies(sr=self.sr, n_fft=self.fft_size)
        sc = librosa.feature.spectral_centroid(S=S, sr=self.sr, hop_length=self.hop_size)[0]
        f['spectral_centroid_mean'] = float(np.mean(sc)); f['spectral_centroid_std'] = float(np.std(sc))
        sb = librosa.feature.spectral_bandwidth(S=S, sr=self.sr, hop_length=self.hop_size)[0]
        f['spectral_bandwidth_mean'] = float(np.mean(sb)); f['spectral_bandwidth_std'] = float(np.std(sb))
        ro = librosa.feature.spectral_rolloff(S=S, sr=self.sr, hop_length=self.hop_size)[0]
        f['spectral_rolloff_mean'] = float(np.mean(ro)); f['spectral_rolloff_std'] = float(np.std(ro))
        sf = librosa.feature.spectral_flatness(S=S, hop_length=self.hop_size)[0]
        f['spectral_flatness_mean'] = float(np.mean(sf)); f['spectral_flatness_std'] = float(np.std(sf))
        if S.shape[1] >= 2:
            flux = np.sum(np.diff(S, axis=1)**2, axis=0); f['spectral_flux_mean'] = float(np.mean(flux)); f['spectral_flux_std'] = float(np.std(flux))
        else:
            f['spectral_flux_mean'] = 0.0; f['spectral_flux_std'] = 0.0
        S_norm = S / (np.sum(S, axis=0, keepdims=True)+1e-20)
        sent = -np.sum(S_norm * np.log2(S_norm+1e-20), axis=0)
        f['spectral_entropy_mean'] = float(np.mean(sent)); f['spectral_entropy_std'] = float(np.std(sent))
        total_energy = float(np.sum(S**2)+1e-20)
        for (lo,hi) in self.freq_bands:
            mask = (freqs>=lo)&(freqs<hi)
            band_e = float(np.sum(S[mask,:]**2)) if np.any(mask) else 0.0
            f[f'band_energy_ratio_{lo}_{hi}Hz'] = float(band_e/total_energy)
        mean_spec = np.mean(S, axis=1); dom_idx = int(np.argmax(mean_spec))
        f['dominant_frequency'] = float(freqs[dom_idx]); f['dominant_freq_ratio'] = float(mean_spec[dom_idx]/(np.sum(mean_spec)+1e-20))
        if audio_segment.size >= 2:
            ac = np.correlate(audio_segment, audio_segment, mode='full'); ac = ac[len(ac)//2:]
            min_lag = int(0.002*self.sr); max_lag = min(int(0.02*self.sr), len(ac)-1)
            if max_lag > min_lag and (max_lag-min_lag)>0:
                seg = ac[min_lag:max_lag]; f['hnr_approximation'] = float((np.max(seg) if seg.size else 0.0)/(ac[0]+1e-20))
            else:
                f['hnr_approximation'] = 0.0
        else:
            f['hnr_approximation'] = 0.0
        return f

    def extract_mfcc_features(self, audio_segment: np.ndarray) -> dict:
        f = {}
        mfccs = librosa.feature.mfcc(y=audio_segment, sr=self.sr, n_mfcc=self.n_mfcc, n_fft=self.fft_size, hop_length=self.hop_size)
        for i in range(self.n_mfcc):
            xi = mfccs[i]; f[f'mfcc_{i}_mean'] = float(np.mean(xi)); f[f'mfcc_{i}_std'] = float(np.std(xi))
            f[f'mfcc_{i}_max'] = float(np.max(xi)); f[f'mfcc_{i}_min'] = float(np.min(xi))
        d = _safe_delta(mfccs, max_width=9)
        for i in range(self.n_mfcc):
            di = d[i]; f[f'delta_mfcc_{i}_mean'] = float(np.mean(di)); f[f'delta_mfcc_{i}_std'] = float(np.std(di))
        a = _safe_delta(d, max_width=9)
        for i in range(self.n_mfcc): f[f'delta2_mfcc_{i}_mean'] = float(np.mean(a[i]))
        return f

    def extract_all_features(self, audio_segment: np.ndarray, label: str) -> dict:
        out = {'label': label}
        min_frames = 9; min_len = self.fft_size + (min_frames-1)*self.hop_size
        if len(audio_segment) < min_len:
            pad = min_len - len(audio_segment); mode = 'reflect' if len(audio_segment)>1 else 'constant'
            audio_segment = np.pad(audio_segment, (0,pad), mode=mode)
        out.update(self.extract_temporal_features(audio_segment))
        out.update(self.extract_spectral_features(audio_segment))
        out.update(self.extract_mfcc_features(audio_segment))
        return out

def _load_labels(label_file: Path):
    labs = []
    with open(label_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3:
                onset, offset, cls = float(parts[0]), float(parts[1]), _normalize_label(parts[2])
                labs.append({"onset": onset, "offset": offset, "class": cls})
    return labs

def _process_audio_file(audio_path: Path, label_path: Path, extractor: BowelSoundFeatureExtractor, max_segments=1000):
    audio, sr = librosa.load(str(audio_path), sr=extractor.sr)
    labels = _load_labels(label_path)[:max_segments]
    feats_list = []
    for lab in tqdm(labels, desc=f"Segments in {audio_path.name}", leave=False):
        start = int(lab["onset"]*sr); end = int(lab["offset"]*sr)
        seg = audio[start:end]
        if seg.size:
            feats = extractor.extract_all_features(seg, lab["class"])
            feats.update({"file": audio_path.stem, "onset": lab["onset"], "offset": lab["offset"]})
            feats_list.append(feats)
    return feats_list

def _visualize_tsne(df: pd.DataFrame, feature_cols: list, out_png: Path):
    X = df[feature_cols].values; y = df["label"].values
    Xs = StandardScaler().fit_transform(X)
    n = len(df); perpl = int(np.clip((n-1)//3, 5, 50)) if n>10 else max(5, n-1)
    tsne = TSNE(n_components=2, perplexity=perpl, random_state=42, n_iter=1000, init='random', learning_rate='auto')
    X2 = tsne.fit_transform(Xs)
    classes = np.unique(y); colors = sns.color_palette('husl', n_colors=len(classes))
    fig = plt.figure(figsize=(14,5))
    ax = plt.subplot(1,2,1)
    for i,cls in enumerate(classes):
        m = (y==cls)
        ax.scatter(X2[m,0], X2[m,1], c=[colors[i]], label=f"{cls}", alpha=0.7, s=50, edgecolors="black", linewidth=0.4)
    ax.set_title("t-SNE of Bowel Sound Features"); ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2"); ax.legend(); ax.grid(True, alpha=0.3)
    ax = plt.subplot(1,2,2); vc = df["label"].value_counts(); ax.bar(vc.index.astype(str), vc.values)
    ax.set_title("Class Distribution"); ax.set_xlabel("Class"); ax.set_ylabel("Count"); ax.grid(True, alpha=0.3)
    plt.tight_layout(); fig.savefig(out_png, dpi=150, bbox_inches="tight"); plt.show()
    return out_png

def run_tsne_pipeline(base_dir: Path, out_png: Path):
    base = Path(base_dir); audio_dir = base; label_dir = base
    extractor = BowelSoundFeatureExtractor(sr=16000, fft_size=400, hop_size=160)
    all_feats = []
    wavs = sorted(audio_dir.glob("*.wav"))
    if not wavs: raise FileNotFoundError(f"No WAV in {audio_dir}")
    for w in tqdm(wavs, desc="Processing files"):
        lab = label_dir / f"{w.stem}.txt"
        if lab.exists(): all_feats.extend(_process_audio_file(w, lab, extractor))
    if not all_feats: raise RuntimeError("No features extracted.")
    df = pd.DataFrame(all_feats)
    df = df[df["label"].isin({"h","mb","sb"})].reset_index(drop=True)
    feature_cols = [c for c in df.columns if c not in ["label","file","onset","offset"]]
    out_png = Path(out_png)
    return _visualize_tsne(df, feature_cols, out_png)
