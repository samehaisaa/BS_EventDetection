import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import get_window
import librosa

def spectrogram_for_event(
    class_name: str,
    occurrence_one_based: int = 1,
    n_fft: int | None = None,
    hop_length: int | None = None,
    window: str = "hann",
    mode: str = "db",
    fmin_hz: float = 0.0,
    fmax_hz: float | None = None,
    view: tuple[float, float] | None = None,
    show_windows: bool = False,
    dynamic_range_db: float | None = 80.0,
    n_mels: int = 64,
):
    if class_name.lower() == "all":
        seg = audio.astype(float)
        on_t, off_t = 0.0, len(audio) / float(sr)
        title_prefix = "Full signal"
        k = None
    else:
        idxs = np.where(labels == class_name)[0]
        if len(idxs) == 0:
            raise ValueError(f"No events for class '{class_name}'.")
        idxs = idxs[np.argsort(onsets_s[idxs])]
        k = max(1, min(occurrence_one_based, len(idxs)))
        ev = idxs[k - 1]
        s0, s1 = int(onsets_s[ev] * sr), int(offsets_s[ev] * sr)
        seg = audio[s0:s1].astype(float)
        if seg.size < 2:
            raise ValueError("Segment too short.")
        on_t, off_t = onsets_s[ev], offsets_s[ev]
        title_prefix = f"class='{class_name}', occ={k}"
    if n_fft is None:
        target_ms = 0.128
        n_fft = int(round(sr * target_ms))
        n_fft = max(64, n_fft + (n_fft % 2))
    if hop_length is None:
        hop_length = max(1, n_fft // 4)
    if seg.size < n_fft:
        n_fft = max(16, int(2 ** np.floor(np.log2(seg.size))))
        hop_length = max(1, n_fft // 4)
    _ = get_window(window, n_fft, fftbins=True)
    T = seg.size / float(sr)
    df = sr / float(n_fft)
    Tw = n_fft / float(sr)
    Th = hop_length / float(sr)
    nyq = sr / 2.0
    fmin = max(0.0, float(fmin_hz))
    fmax = nyq if fmax_hz is None else float(np.clip(fmax_hz, 0, nyq))
    S = librosa.feature.melspectrogram(y=seg, sr=sr, n_fft=n_fft, hop_length=hop_length, window=window, center=False, power=2.0, fmin=fmin, fmax=fmax, n_mels=n_mels)
    if mode == "magnitude":
        S_rep = np.sqrt(np.maximum(S, 0.0))
        vlabel = "Mel magnitude"
    elif mode == "power":
        S_rep = S
        vlabel = "Mel power"
    elif mode == "db":
        S_rep = librosa.power_to_db(S, ref=np.max if dynamic_range_db is not None else 1.0, top_db=dynamic_range_db if dynamic_range_db is not None else None)
        vlabel = "Mel power (dB)"
    else:
        raise ValueError("mode must be one of {'magnitude','power','db'}")
    n_frames = S.shape[1]
    frame_idx = np.arange(n_frames)
    t_local = frame_idx * hop_length / float(sr) + Tw / 2.0
    t_global = t_local + on_t
    window_starts = t_global - Tw / 2.0
    window_ends = t_global + Tw / 2.0
    mel_freqs = librosa.mel_frequencies(n_mels=n_mels, fmin=fmin, fmax=fmax)
    if view is not None:
        t0, t1 = float(view[0]), float(view[1])
        tmask = (t_global >= t0) & (t_global <= t1)
        if not np.any(tmask):
            raise ValueError("No frames fall inside the requested 'view' range.")
        t_view = t_global[tmask]
        S_plot = S_rep[:, tmask]
        window_starts_v = window_starts[tmask]
        window_ends_v = window_ends[tmask]
        xmin, xmax = t_view[0], t_view[-1]
    else:
        t_view = t_global
        S_plot = S_rep
        window_starts_v = window_starts
        window_ends_v = window_ends
        xmin, xmax = t_view[0], t_view[-1]
    if mode != "db" and dynamic_range_db is not None:
        vmax = np.nanmax(S_plot)
        vmin = vmax - float(dynamic_range_db)
    else:
        vmin = None
        vmax = None
    plt.figure(figsize=(12, 4.6))
    extent = [xmin, xmax, mel_freqs[0], mel_freqs[-1]]
    im = plt.imshow(S_plot, aspect="auto", origin="lower", extent=extent, vmin=vmin, vmax=vmax)
    cbar = plt.colorbar(im)
    cbar.set_label(vlabel)
    plt.xlabel("Time (s)")
    plt.ylabel("Mel frequency (Hz)")
    if show_windows:
        ax = plt.gca()
        for ws, we in zip(window_starts_v, window_ends_v):
            ax.axvspan(ws, we, ymin=0.0, ymax=0.06, alpha=0.12, color="k")
        for tc in t_view:
            ax.axvline(tc, ymin=0.0, ymax=0.06, alpha=0.25, linewidth=0.6, color="k")
    if k is None:
        title_suffix = f"(T={T:.3f}s, Δf={df:.3f} Hz, window={Tw*1e3:.0f} ms, hop={Th*1e3:.0f} ms)"
    else:
        title_suffix = f"{on_t:.3f}–{off_t:.3f}s (T={T:.3f}s, Δf={df:.3f} Hz, window={Tw*1e3:.0f} ms, hop={Th*1e3:.0f} ms)"
    plt.title(f"Log-mel spectrogram — {title_prefix} | {title_suffix}")
    plt.tight_layout()
    plt.show()
    print(f"Segment duration T = {T:.6f} s")
    print(f"Sampling rate sr   = {sr} Hz")
    print(f"n_fft (window)     = {n_fft} samples  (~{Tw*1e3:.1f} ms)")
    print(f"hop_length         = {hop_length} samples  (~{Th*1e3:.1f} ms)")
    print(f"Δf (bin width)     = {df:.6f} Hz")
    print(f"Nyquist            = {nyq:.2f} Hz")
    if fmax_hz is not None:
        print(f"Displayed up to    = {fmax:.2f} Hz")
    if view is not None:
        print(f"View window        = [{xmin:.3f}, {xmax:.3f}] s")
        print(f"Frames in view     = {t_view.size}")
        if t_view.size:
            print(f"First window       = [{window_starts_v[0]:.3f}, {window_ends_v[0]:.3f}] s")
            print(f"Last window        = [{window_starts_v[-1]:.3f}, {window_ends_v[-1]:.3f}] s")
    return mel_freqs, t_local, None, S_rep

spectrogram_for_event(
    "all",
    n_fft=1024,
    hop_length=256,
    mode="db",
    fmin_hz=0,
    fmax_hz=4000,
    view=(0.0, 60.0),
    show_windows=False,
    dynamic_range_db=80.0,
    n_mels=64,
)
