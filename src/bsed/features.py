from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

from .tsne_features import BowelSoundFeatureExtractor  # reuse your extractor

def _clip(a, lo, hi):
    return max(lo, min(hi, a))

def extract_features_for_events(DATA: dict, events_df: pd.DataFrame, sr_override: int | None = None) -> pd.DataFrame:
    """
    For each row in events_df, cut the audio from DATA[key]["segments_hp"][seg_idx]
    using onset/offset (local seconds), extract descriptors, and return a DF with:
    [event_id, key, seg_idx, onset/offset local/global, label(if present), <features...>]
    """
    rows = []
    for key, item in DATA.items():
        sr = sr_override or int(item["sr"])
        segments = item["segments_hp"]
        # group events by (key)
        Ev = events_df[events_df["key"] == key]
        if Ev.empty:
            continue
        for _, r in Ev.iterrows():
            seg_idx = int(r["seg_idx"])
            if seg_idx < 0 or seg_idx >= len(segments):
                continue
            y = segments[seg_idx]
            on = float(r["onset_s_local"]); off = float(r["offset_s_local"])
            i0 = _clip(int(round(on * sr)), 0, len(y))
            i1 = _clip(int(round(off * sr)), i0+1, len(y))
            seg = y[i0:i1]
            # extract
            extr = BowelSoundFeatureExtractor(sr=sr, fft_size=400, hop_size=160)
            feats = extr.extract_all_features(seg, label=str(r.get("label", "")) if "label" in r else "")
            feats.update({
                "event_id": r["event_id"],
                "key": key,
                "seg_idx": seg_idx,
                "onset_s_local": on,
                "offset_s_local": off,
                "onset_s_global": float(r.get("onset_s_global", seg_idx*10.0 + on)),
                "offset_s_global": float(r.get("offset_s_global", seg_idx*10.0 + off)),
            })
            rows.append(feats)
    return pd.DataFrame(rows)
