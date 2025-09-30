import os
from pathlib import Path
import librosa
from .io import load_audio, highpass_biquad, split_nonoverlapping_segments
from .melspec import logmel_segment
from .labels import load_labels_for_wav

def process_wav_collect(path, sr, seg_dur_s, hp_cut_hz, n_mels, win_ms, hop_ms):
    y, sr0 = load_audio(str(path), sr=None)
    assert sr0 is not None
    y_hp = highpass_biquad(y, sr0, cutoff_hz=hp_cut_hz)
    segs = split_nonoverlapping_segments(y_hp, sr0, seg_dur_s=seg_dur_s)
    specs = [logmel_segment(seg, sr0, n_mels=n_mels, win_ms=win_ms, hop_ms=hop_ms) for seg in segs]
    return {"sr": sr0, "duration_s": len(y)/sr0, "segments_hp": segs, "specs_z": specs, "path": str(path), "name": os.path.basename(path)}

def build_DATA(base_dir, sr, seg_dur_s, hp_cut_hz, n_mels, win_ms, hop_ms, wanted):
    base = Path(base_dir)
    wavs = sorted(base.glob("*.wav"))
    if not wavs:
        raise FileNotFoundError(f"No WAV files in {base_dir}")

    DATA = {}
    for i, w in enumerate(wavs, 1):
        item = process_wav_collect(w, sr, seg_dur_s, hp_cut_hz, n_mels, win_ms, hop_ms)
        labels = load_labels_for_wav(str(w), force_seconds=True, wanted=wanted)
        item["labels"] = labels
        key = f"wav{i}"
        DATA[key] = item
        print(f"[{item['name']}] sr={item['sr']} Hz | 10s segments={len(item['segments_hp'])}")
        if item["specs_z"]:
            bad = [k for k, Z in enumerate(item["specs_z"]) if Z.shape[0]!=n_mels]
            if bad:
                print(f"[WARN] {len(bad)} segments not {n_mels} mels")
    return DATA
