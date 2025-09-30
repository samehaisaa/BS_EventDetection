import numpy as np, matplotlib.pyplot as plt, librosa.display
from .frames import _frame_geometry
from .labels import labels_for_segment

def show_segment_wave_and_mel(DATA, key="wav1", idx=0, mel_vmin=-3, mel_vmax=3, win_ms=25.0, hop_ms=10.0):
    item = DATA[key]; sr = item["sr"]
    y = item["segments_hp"][idx]; Z = item["specs_z"][idx]
    win_len, hop_len = _frame_geometry(sr, win_ms, hop_ms)
    t = np.arange(len(y))/sr
    fig = plt.figure(figsize=(14,6))
    ax1 = plt.subplot(1,2,1)
    librosa.display.waveshow(y, sr=sr, ax=ax1)
    ax1.set_title(f"{item['name']} | seg #{idx} – time domain (HP 60 Hz)")
    ax1.set_xlabel("Time (s)"); ax1.set_ylabel("Amplitude")
    ax2 = plt.subplot(1,2,2)
    img = librosa.display.specshow(Z, sr=sr, hop_length=hop_len, x_axis="time", y_axis="mel", ax=ax2)
    img.set_clim(mel_vmin, mel_vmax)
    plt.colorbar(img, ax=ax2, format="%.1f").set_label("Standardized log-mel")
    ax2.set_title(f"{item['name']} | seg #{idx} – Mel")
    plt.tight_layout(); plt.show()

def preview_grid(DATA, key="wav1", n=4, mel_vmin=-3, mel_vmax=3, win_ms=25.0, hop_ms=10.0):
    item = DATA[key]; sr = item["sr"]
    win_len, hop_len = _frame_geometry(sr, win_ms, hop_ms)
    n = min(n, len(item["specs_z"]))
    cols = 2; rows = int(np.ceil(n/cols))
    plt.figure(figsize=(12, 3*rows))
    for i in range(n):
        ax = plt.subplot(rows, cols, i+1)
        Z = item["specs_z"][i]
        img = librosa.display.specshow(Z, sr=sr, hop_length=hop_len, x_axis="time", y_axis="mel", ax=ax)
        img.set_clim(mel_vmin, mel_vmax)
        ax.set_title(f"{item['name']} seg #{i}")
    plt.tight_layout(); plt.show()

def show_segment_wave_mel_with_labels(DATA, key="wav1", idx=0, mel_vmin=-3, mel_vmax=3, win_ms=25.0, hop_ms=10.0):
    from matplotlib.gridspec import GridSpec
    item = DATA[key]; sr = item["sr"]
    y = item["segments_hp"][idx]; Z = item["specs_z"][idx]
    L = labels_for_segment(item.get("labels"), seg_idx=idx, seg_len_s=10.0)
    win_len, hop_len = _frame_geometry(sr, win_ms, hop_ms)
    t_wave = np.arange(len(y))/sr
    frame_idx = np.arange(Z.shape[1]); t_frames = frame_idx * (hop_len / sr)

    fig = plt.figure(figsize=(14,7))
    gs = GridSpec(3,2, width_ratios=[1.0,1.2], height_ratios=[2.0,0.6,0.0], figure=fig)

    ax_wave = fig.add_subplot(gs[0,0]); ax_wave.plot(t_wave, y, lw=0.8); ax_wave.set_xlim(0,10)
    ax_wave.set_title(f"{item['name']} | seg #{idx} – time domain (HP 60 Hz)")
    ax_wave.set_xlabel("Time (s)"); ax_wave.set_ylabel("Amp")

    ax_gantt = fig.add_subplot(gs[1,0], sharex=ax_wave)
    if L is not None and not L.empty:
        lanes = {"b":20, "mb":10, "h":0}; height = 8
        colors = {"b":"#1f77b4","mb":"#2ca02c","h":"#d62728"}
        for _, r in L.iterrows():
            x0 = float(r["local_start"]); w = float(r["local_end"] - r["local_start"])
            y0 = lanes.get(r["label"],0)
            ax_gantt.broken_barh([(x0,w)], (y0,height), facecolors=colors.get(r["label"],"gray"))
        ax_gantt.set_yticks([4,14,24]); ax_gantt.set_yticklabels(["h","mb","b"])
    else:
        ax_gantt.text(0.5,0.5,"No labels", ha="center", va="center", transform=ax_gantt.transAxes, alpha=0.7)
        ax_gantt.set_yticks([])
    ax_gantt.set_xlim(0,10); ax_gantt.set_xlabel("Time (s)"); ax_gantt.set_title("Labels (Gantt)")

    ax_mel = fig.add_subplot(gs[:,1])
    img = librosa.display.specshow(Z, x_coords=t_frames, y_axis="mel", sr=sr, ax=ax_mel)
    img.set_clim(mel_vmin, mel_vmax)
    cb = plt.colorbar(img, ax=ax_mel, format="%.1f"); cb.set_label("Standardized log-mel")
    ax_mel.set_xlim(0,10); ax_mel.set_title(f"{item['name']} | seg #{idx} – Mel (128 × {Z.shape[1]})")
    plt.tight_layout(); plt.show()

def show_binary_gantt_for_segment(DATA, key="wav1", idx=0, win_ms=25.0):
    item = DATA[key]
    if "frame_targets" not in item:
        raise RuntimeError("Run build-binary first.")
    mask = item["frame_targets"][idx]
    t = item["frame_times_s"][idx]
    win_s = win_ms/1000.0

    diffs = np.diff(np.concatenate([[0], mask, [0]]))
    starts_idx = np.where(diffs == 1)[0]
    ends_idx   = np.where(diffs == -1)[0]

    fig, ax = plt.subplots(figsize=(12,2.8))
    bars = []
    for s_idx, e_idx in zip(starts_idx, ends_idx):
        x0 = t[s_idx]; x1 = t[e_idx-1] + win_s
        bars.append((x0, x1 - x0))
    if bars:
        ax.broken_barh(bars, (5,8), facecolors="#2ca02c")
    ax.set_xlim(0,10); ax.set_ylim(0,20)
    ax.set_yticks([9]); ax.set_yticklabels(["BS (1)"])
    ax.set_xlabel("Time (s)"); ax.set_title(f"{item['name']} | seg #{idx} – Binary (50% rule)")
    ax.hlines(9,0,10,colors="lightgray", linestyles="--", lw=0.8)
    plt.tight_layout(); plt.show()

def plot_prob_track(DATA, probs_by_key, key="wav1", idx=0, thr=0.5):
    import matplotlib.pyplot as plt
    sr = DATA[key]["sr"]; hop = int(0.010*sr)
    p = probs_by_key[key][idx]
    t = np.arange(len(p))*(hop/sr)
    plt.figure(figsize=(12,3))
    plt.plot(t, p); plt.axhline(thr, ls="--", label=f"thr={thr}")
    plt.xlabel("Time (s)"); plt.ylabel("p(BS)")
    plt.title(f"{key}:{idx} probability track"); plt.legend(); plt.show()

def show_gantt_gt_vs_pred(DATA, key, idx, probs, thr=0.5, gap_merge_s=0.100, min_dur_s=0.018):
    import matplotlib.pyplot as plt
    from .infer import probs_to_events_simple
    # build GT intervals (local)
    from .labels import labels_for_segment
    gt_df = labels_for_segment(DATA[key]["labels"], idx, seg_len_s=10.0)
    gt = [(float(r.local_start), float(r.local_end)) for _, r in gt_df.iterrows()] if gt_df is not None else []
    pred = probs_to_events_simple(probs, thr=thr, gap_merge_s=gap_merge_s, min_dur_s=min_dur_s)
    fig, ax = plt.subplots(figsize=(12,2))
    for st,en in gt:
        ax.broken_barh([(st, en-st)], (10,6), facecolors="green", alpha=0.6, label="GT")
    for st,en in pred:
        ax.broken_barh([(st, en-st)], (0,6), facecolors="red", alpha=0.6, label="Pred")
    ax.set_xlim(0,10); ax.set_ylim(-2,20)
    ax.set_xlabel("Time (s)"); ax.set_yticks([3,13]); ax.set_yticklabels(["Pred","GT"])
    ax.set_title(f"{key} seg {idx} – GT vs Pred")
    h,l = ax.get_legend_handles_labels(); ax.legend(dict(zip(l,h)).values(), dict(zip(l,h)).keys())
    plt.show()
