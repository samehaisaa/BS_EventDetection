#!/usr/bin/env python3
"""
Bowel Sound Feature Extraction and t-SNE Visualization
Extracts event-level descriptors from labeled bowel sound segments and visualizes using PCA → t-SNE
"""

import numpy as np
import pandas as pd
import librosa
import scipy.stats
from scipy import signal
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')


# ---------- helpers ----------

def _safe_delta(X: np.ndarray, max_width: int = 9) -> np.ndarray:
    """Delta that never crashes on short sequences."""
    # X shape = (n_features, n_frames)
    if X.ndim != 2:
        raise ValueError("Expected 2D array (features, frames) for delta computation.")
    T = X.shape[1]
    if T < 3:
        return np.zeros_like(X)  # nothing to difference reliably
    # width must be odd and <= T
    width = min(max_width, T if (T % 2 == 1) else T - 1)
    if width < 3:  # still too tiny: degrade gracefully
        return np.zeros_like(X)
    return librosa.feature.delta(X, width=width, mode='interp')


# ---------- feature extractor ----------

class BowelSoundFeatureExtractor:
    def __init__(self, sr=16000, fft_size=400, hop_size=160):
        """
        Initialize feature extractor for bowel sounds

        Args:
            sr: Sample rate (16000 Hz)
            fft_size: FFT window size (400 samples = 25ms at 16kHz)
            hop_size: Hop size (160 samples = 10ms at 16kHz)
        """
        self.sr = sr
        self.fft_size = fft_size
        self.hop_size = hop_size
        self.n_mfcc = 13

        # Define frequency bands for bowel sounds (Hz)
        # Based on literature, bowel sounds typically range from ~100–1500 Hz
        self.freq_bands = [
            (0, 100),      # Sub-bass (noise floor)
            (100, 200),    # Low frequency bowel sounds
            (200, 400),    # Mid-low frequency (typical bursts)
            (400, 800),    # Mid frequency (harmonics)
            (800, 1500),   # High frequency bowel sounds
            (1500, 3000),  # Very high (may contain harmonics)
            (3000, 8000)   # Ultra-high (mostly noise)
        ]

    def extract_temporal_features(self, audio_segment: np.ndarray) -> dict:
        """Extract temporal-domain features optimal for short bursts."""
        features = {}

        # Basic statistics
        features['duration_ms'] = len(audio_segment) / self.sr * 1000.0
        features['rms'] = float(np.sqrt(np.mean(audio_segment**2) + 1e-20))
        features['peak_amplitude'] = float(np.max(np.abs(audio_segment)) if audio_segment.size else 0.0)

        # Crest factor (peak/rms ratio) - good for burst detection
        features['crest_factor'] = float(features['peak_amplitude'] / (features['rms'] + 1e-20))

        # Zero crossing rate - useful for distinguishing harmonics/noise
        zcr = librosa.feature.zero_crossing_rate(
            audio_segment, frame_length=self.fft_size, hop_length=self.hop_size
        )[0]
        features['zcr'] = float(np.mean(zcr)) if zcr.size else 0.0

        # Temporal centroid - "center of mass" in time
        envelope = np.abs(signal.hilbert(audio_segment))
        time_axis = np.arange(len(envelope)) / self.sr
        denom = float(np.sum(envelope) + 1e-20)
        features['temporal_centroid'] = float(np.sum(time_axis * envelope) / denom)

        # Attack time (10–90% of max amplitude)
        if len(envelope) > 1:
            win = min(51, (len(envelope) // 4) * 2 + 1)  # odd, <= len
            env_smooth = signal.savgol_filter(envelope, win, 3) if win >= 3 else envelope
        else:
            env_smooth = envelope
        max_env = float(np.max(env_smooth) if env_smooth.size else 0.0)

        if max_env > 0:
            rise_10 = np.where(env_smooth >= 0.1 * max_env)[0]
            rise_90 = np.where(env_smooth >= 0.9 * max_env)[0]
            if rise_10.size > 0 and rise_90.size > 0 and rise_90[0] >= rise_10[0]:
                features['attack_time_ms'] = (rise_90[0] - rise_10[0]) / self.sr * 1000.0
            else:
                features['attack_time_ms'] = 0.0
        else:
            features['attack_time_ms'] = 0.0

        # Temporal flatness (how "bursty" vs sustained)
        gmean = float(scipy.stats.gmean(envelope + 1e-20)) if envelope.size else 0.0
        mean_env = float(np.mean(envelope) + 1e-20)
        features['temporal_flatness'] = float(gmean / mean_env)

        # Count prominent peaks (distinguish single vs multiple bursts)
        prominence_threshold = 0.3 * max_env if max_env > 0 else 0.0
        peaks, _ = signal.find_peaks(env_smooth, prominence=prominence_threshold,
                                     distance=int(0.01 * self.sr))  # >=10ms apart
        features['num_peaks'] = int(len(peaks))

        return features

    def extract_spectral_features(self, audio_segment: np.ndarray) -> dict:
        """Extract spectral features optimized for bowel sounds."""
        features = {}

        # Compute STFT
        D = librosa.stft(audio_segment, n_fft=self.fft_size, hop_length=self.hop_size)
        S = np.abs(D) + 1e-20  # avoid zeros
        freqs = librosa.fft_frequencies(sr=self.sr, n_fft=self.fft_size)

        # Spectral centroid, bandwidth, rolloff
        spec_centroid = librosa.feature.spectral_centroid(S=S, sr=self.sr, hop_length=self.hop_size)[0]
        features['spectral_centroid_mean'] = float(np.mean(spec_centroid))
        features['spectral_centroid_std'] = float(np.std(spec_centroid))

        spec_bandwidth = librosa.feature.spectral_bandwidth(S=S, sr=self.sr, hop_length=self.hop_size)[0]
        features['spectral_bandwidth_mean'] = float(np.mean(spec_bandwidth))
        features['spectral_bandwidth_std'] = float(np.std(spec_bandwidth))

        spec_rolloff = librosa.feature.spectral_rolloff(S=S, sr=self.sr, hop_length=self.hop_size)[0]
        features['spectral_rolloff_mean'] = float(np.mean(spec_rolloff))
        features['spectral_rolloff_std'] = float(np.std(spec_rolloff))

        # Spectral flatness
        spec_flatness = librosa.feature.spectral_flatness(S=S, hop_length=self.hop_size)[0]
        features['spectral_flatness_mean'] = float(np.mean(spec_flatness))
        features['spectral_flatness_std'] = float(np.std(spec_flatness))

        # Spectral flux (simple squared diff across frames)
        if S.shape[1] >= 2:
            spec_flux = np.sum(np.diff(S, axis=1) ** 2, axis=0)
            features['spectral_flux_mean'] = float(np.mean(spec_flux))
            features['spectral_flux_std'] = float(np.std(spec_flux))
        else:
            features['spectral_flux_mean'] = 0.0
            features['spectral_flux_std'] = 0.0

        # Spectral entropy
        S_norm = S / (np.sum(S, axis=0, keepdims=True) + 1e-20)
        spectral_entropy = -np.sum(S_norm * np.log2(S_norm + 1e-20), axis=0)
        features['spectral_entropy_mean'] = float(np.mean(spectral_entropy))
        features['spectral_entropy_std'] = float(np.std(spectral_entropy))

        # Band energy ratios
        total_energy = float(np.sum(S ** 2) + 1e-20)
        for (low, high) in self.freq_bands:
            band_mask = (freqs >= low) & (freqs < high)
            band_energy = float(np.sum(S[band_mask, :] ** 2)) if np.any(band_mask) else 0.0
            features[f'band_energy_ratio_{low}_{high}Hz'] = float(band_energy / total_energy)

        # Dominant frequency (from mean spectrum)
        mean_spectrum = np.mean(S, axis=1)
        dom_idx = int(np.argmax(mean_spectrum))
        features['dominant_frequency'] = float(freqs[dom_idx])
        features['dominant_freq_ratio'] = float(mean_spectrum[dom_idx] / (np.sum(mean_spectrum) + 1e-20))

        # HNR approximation (autocorr peak / zero-lag)
        if audio_segment.size >= 2:
            autocorr = np.correlate(audio_segment, audio_segment, mode='full')
            autocorr = autocorr[len(autocorr) // 2:]
            min_lag = int(0.002 * self.sr)  # 2 ms
            max_lag = min(int(0.02 * self.sr), len(autocorr) - 1)  # 20 ms
            if max_lag > min_lag:
                seg = autocorr[min_lag:max_lag]
                if seg.size > 0:
                    max_autoc = float(np.max(seg))
                    features['hnr_approximation'] = float(max_autoc / (autocorr[0] + 1e-20))
                else:
                    features['hnr_approximation'] = 0.0
            else:
                features['hnr_approximation'] = 0.0
        else:
            features['hnr_approximation'] = 0.0

        return features

    def extract_mfcc_features(self, audio_segment: np.ndarray) -> dict:
        """Extract MFCCs (+ deltas, delta-deltas) safely on short segments."""
        features = {}

        mfccs = librosa.feature.mfcc(
            y=audio_segment, sr=self.sr, n_mfcc=self.n_mfcc,
            n_fft=self.fft_size, hop_length=self.hop_size
        )

        # Stats for each MFCC coefficient
        for i in range(self.n_mfcc):
            xi = mfccs[i]
            features[f'mfcc_{i}_mean'] = float(np.mean(xi))
            features[f'mfcc_{i}_std'] = float(np.std(xi))
            features[f'mfcc_{i}_max'] = float(np.max(xi))
            features[f'mfcc_{i}_min'] = float(np.min(xi))

        # Safe deltas with adaptive width
        delta_mfccs = _safe_delta(mfccs, max_width=9)
        for i in range(self.n_mfcc):
            di = delta_mfccs[i]
            features[f'delta_mfcc_{i}_mean'] = float(np.mean(di))
            features[f'delta_mfcc_{i}_std'] = float(np.std(di))

        # Acceleration (delta of delta), still safe
        accel = _safe_delta(delta_mfccs, max_width=9)
        for i in range(self.n_mfcc):
            features[f'delta2_mfcc_{i}_mean'] = float(np.mean(accel[i]))

        return features

    def extract_all_features(self, audio_segment: np.ndarray, label: str) -> dict:
        """Extract all features for a single audio segment."""
        all_features = {'label': label}

        # Guarantee enough frames for robust deltas
        # With hop=160 (10 ms), want ~≥9 frames → min_len ≈ 400 + 8*160 = 1680 samples (~105 ms)
        min_frames = 9
        min_len = self.fft_size + (min_frames - 1) * self.hop_size
        if len(audio_segment) < min_len:
            pad = min_len - len(audio_segment)
            mode = 'reflect' if len(audio_segment) > 1 else 'constant'
            audio_segment = np.pad(audio_segment, (0, pad), mode=mode)

        # Extract feature groups
        all_features.update(self.extract_temporal_features(audio_segment))
        all_features.update(self.extract_spectral_features(audio_segment))
        all_features.update(self.extract_mfcc_features(audio_segment))

        return all_features


# ---------- IO & analysis ----------

def load_labels(label_file: str) -> list:
    """Load labels from text file (onset, offset, class)."""
    labels = []
    with open(label_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3:
                onset = float(parts[0])
                offset = float(parts[1])
                class_label = parts[2]
                labels.append({'onset': onset, 'offset': offset, 'class': class_label})
    return labels


def process_audio_file(audio_path: Path, label_path: Path, extractor: BowelSoundFeatureExtractor) -> list:
    """Process a single audio file with its labels."""
    # Load audio
    audio, sr = librosa.load(str(audio_path), sr=extractor.sr)

    # Load labels
    labels = load_labels(str(label_path))

    # Extract features for each labeled segment
    features_list = []
    for lab in labels:
        start_sample = int(lab['onset'] * sr)
        end_sample = int(lab['offset'] * sr)
        segment = audio[start_sample:end_sample]

        if segment.size > 0:
            feats = extractor.extract_all_features(segment, lab['class'])
            feats['file'] = audio_path.stem
            feats['onset'] = lab['onset']
            feats['offset'] = lab['offset']
            features_list.append(feats)

    return features_list


def visualize_tsne(df_features: pd.DataFrame, feature_cols: list, output_dir: str = './'):
    X = df_features[feature_cols].values
    y = df_features['label'].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    tsne = TSNE(n_components=2, perplexity=30, random_state=42, n_iter=1000, init='random', learning_rate='auto')
    X_tsne = tsne.fit_transform(X_scaled)

    classes = np.unique(y)
    colors = sns.color_palette('husl', n_colors=len(classes))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1) Main t-SNE plot (no PCA)
    ax = axes[0]
    for i, cls in enumerate(classes):
        m = (y == cls)
        ax.scatter(X_tsne[m, 0], X_tsne[m, 1], c=[colors[i]], label=f'{cls}',
                   alpha=0.7, s=50, edgecolors='black', linewidth=0.4)
    ax.set_title('t-SNE of Bowel Sound Features (raw)', fontsize=13, fontweight='bold')
    ax.set_xlabel('t-SNE 1'); ax.set_ylabel('t-SNE 2'); ax.legend(); ax.grid(True, alpha=0.3)

    # 2) Class distribution
    ax = axes[1]
    vc = df_features['label'].value_counts()
    ax.bar(vc.index.astype(str), vc.values, color=colors[:len(vc)])
    ax.set_title('Class Distribution', fontsize=13, fontweight='bold')
    ax.set_xlabel('Class'); ax.set_ylabel('Count'); ax.grid(True, alpha=0.3)

    # 3) Spectral centroid vs number of peaks
    ax = axes[2]
    for i, cls in enumerate(classes):
        m = (df_features['label'] == cls)
        ax.scatter(df_features.loc[m, 'spectral_centroid_mean'],
                   df_features.loc[m, 'num_peaks'],
                   c=[colors[i]], label=f'{cls}', alpha=0.7, s=35)
    ax.set_title('Spectral Centroid vs Num Peaks', fontsize=13, fontweight='bold')
    ax.set_xlabel('Mean Spectral Centroid (Hz)'); ax.set_ylabel('Num Peaks')
    ax.legend(); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    plt.savefig(f'{output_dir}/bowel_sound_tsne_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()

    return X_tsne, None


def analyze_feature_discriminability(df_features: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    """Analyze which features best discriminate between classes via one-way ANOVA with eta-squared."""
    from scipy.stats import f_oneway

    results = []
    classes = df_features['label'].unique()

    for feature in feature_cols:
        groups = [df_features[df_features['label'] == cls][feature].values for cls in classes]
        # Skip features that are constant or missing
        if any(len(g) == 0 for g in groups):
            continue
        try:
            f_stat, p_value = f_oneway(*groups)
        except Exception:
            # If ANOVA fails (e.g., identical values), skip gracefully
            continue

        mean_all = float(df_features[feature].mean())
        ss_between = sum(len(g) * (float(np.mean(g)) - mean_all) ** 2 for g in groups)
        ss_total = float(np.sum((df_features[feature].values - mean_all) ** 2) + 1e-20)
        eta_squared = float(ss_between / ss_total)

        results.append({
            'feature': feature,
            'f_statistic': float(f_stat),
            'p_value': float(p_value),
            'eta_squared': eta_squared
        })

    if not results:
        return pd.DataFrame(columns=['feature', 'f_statistic', 'p_value', 'eta_squared'])

    results_df = pd.DataFrame(results).sort_values('eta_squared', ascending=False)

    print("\n" + "=" * 60)
    print("TOP 20 MOST DISCRIMINATIVE FEATURES (by Effect Size)")
    print("=" * 60)
    print(results_df.head(20).to_string(index=False))

    return results_df


# ---------- main ----------

def main():
    audio_dir = Path('./dige')
    label_dir = Path('./dige')
    output_dir = Path('./'); output_dir.mkdir(exist_ok=True)

    extractor = BowelSoundFeatureExtractor(sr=16000, fft_size=400, hop_size=160)

    all_features = []
    audio_files = list(audio_dir.glob('*.wav'))
    for audio_path in audio_files:
        label_path = label_dir / f"{audio_path.stem}.txt"
        if label_path.exists():
            print(f"Processing {audio_path.name}...")
            feats = process_audio_file(audio_path, label_path, extractor)
            all_features.extend(feats)

    if not all_features:
        print("No features extracted. Check your audio/label paths and formats.")
        return None, None, None, None

    df = pd.DataFrame(all_features)

    # ★ Keep only the target classes
    target_classes = {'h', 'mb', 'sb'}
    df = df[df['label'].isin(target_classes)].reset_index(drop=True)
    if df.empty:
        print("No samples for classes {'h','mb','sb'} after filtering.")
        return None, None, None, None

    df.to_csv(output_dir / 'bowel_sound_features.csv', index=False)
    print(f"\nSaved {len(df)} samples to bowel_sound_features.csv")

    metadata_cols = ['label', 'file', 'onset', 'offset']
    feature_cols = [c for c in df.columns if c not in metadata_cols]

    print("\n" + "=" * 40)
    print("CLASS DISTRIBUTION (h, mb, sb):")
    print("=" * 40)
    print(df['label'].value_counts())

    discriminability_df = analyze_feature_discriminability(df, feature_cols)

    # Raw t-SNE (no PCA)
    X_tsne, _ = visualize_tsne(df, feature_cols, str(output_dir))

    print("\n" + "=" * 60)
    print("KEY INSIGHTS FOR BOWEL SOUND CLASSIFICATION (h / mb / sb):")
    print("=" * 60)

    print("\n1. MOST INFORMATIVE FEATURES FOR SHORT BURSTS:")
    print("-" * 50)
    if discriminability_df is not None and not discriminability_df.empty:
        for i, feat in enumerate(discriminability_df.head(10)['feature'].tolist(), 1):
            if 'band_energy' in feat:
                note = "Frequency content split across ~100–800 Hz is key"
            elif 'num_peaks' in feat:
                note = "Separates single vs multiple bursts"
            elif 'hnr' in feat:
                note = "Harmonic vs noisy content"
            elif 'temporal' in feat:
                note = "Burst timing/shape"
            elif 'mfcc' in feat:
                note = "Spectral envelope"
            else:
                note = ""
            print(f"   {i}. {feat}{' - ' + note if note else ''}")
    else:
        print("   (Could not compute discriminability rankings.)")

    print("\n2. RECOMMENDED FEATURE SUBSETS:")
    print("-" * 50)
    print("   Minimal: band_energy_ratio_100_200/200_400/400_800, num_peaks, crest_factor, temporal_flatness,")
    print("            spectral_centroid_mean, spectral_flatness_mean, MFCC 0–4 means")
    print("   Full:    all band energies, all temporal stats, spectral stats (mean+std), MFCCs+Δ, HNR approx")

    print("\n3. CLASSIFICATION STRATEGY:")
    print("-" * 50)
    print("   Hierarchical: (h) vs (mb/sb) → then (mb) vs (sb). Tree/XGBoost/LightGBM work well here.")

    return df, X_tsne, None, discriminability_df


if __name__ == "__main__":
    df, X_tsne, pca, discriminability_df = main()
