from __future__ import annotations
from typing import List, Dict, Optional, Sequence
import numpy as np
from .io import Event

def _median_smooth(p: np.ndarray, win: int) -> np.ndarray:
    if win <= 1: 
        return p
    pad = win // 2
    z = np.pad(p, (pad, pad), mode="edge")
    out = np.empty_like(p)
    for i in range(len(p)):
        out[i] = np.median(z[i:i+win])
    return out

def binarize_and_merge(
    frame_probs: np.ndarray,          # (K, T)
    sr: int,
    hop_length: int,
    classes: Sequence[str],
    threshold: float | Sequence[float] = 0.5,
    median_window: int = 0,
    min_dur_s: float = 0.08,
    min_gap_s: float = 0.05
) -> List[Event]:
    K, T = frame_probs.shape
    thr = np.array(threshold if isinstance(threshold, (list, tuple, np.ndarray)) else [threshold]*K, dtype=float)
    events: List[Event] = []
    for k, cls in enumerate(classes):
        p = frame_probs[k].copy()
        if median_window > 1:
            p = _median_smooth(p, median_window)
        mask = p >= thr[k]
        # remove tiny events (min_dur) and fill tiny gaps 
        min_dur = int(round(min_dur_s * sr / hop_length))
        min_gap = int(round(min_gap_s * sr / hop_length))
        # close small gaps: dilate then erode trick
        if min_gap > 0:
            # dilate
            dil = mask.copy()
            i = 0
            while i < T:
                if not dil[i]:
                    j = i
                    while j < T and not dil[j]:
                        j += 1
                    gap = j - i
                    if gap > 0 and gap <= min_gap:
                        dil[i:j] = True
                    i = j
                else:
                    i += 1
            mask = dil
        # remove small islands
        if min_dur > 0:
            pruned = mask.copy()
            i = 0
            while i < T:
                if pruned[i]:
                    j = i
                    while j < T and pruned[j]:
                        j += 1
                    dur = j - i
                    if dur < min_dur:
                        pruned[i:j] = False
                    i = j
                else:
                    i += 1
            mask = pruned
        # to events
        i = 0
        while i < T:
            if mask[i]:
                j = i
                while j < T and mask[j]:
                    j += 1
                onset = i * hop_length / sr
                offset = j * hop_length / sr
                events.append(Event(onset, offset, cls))
                i = j
            else:
                i += 1
    # now we sort by onset
    events.sort(key=lambda e: e.onset)
    return events
