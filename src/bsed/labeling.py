from __future__ import annotations
from typing import Dict, List, Tuple, Optional
import numpy as np
from .io import Event

def build_class_map(classes: List[str]) -> Dict[str, int]:
    return {c:i for i,c in enumerate(classes)}

def events_to_frame_targets(events: List[Event], n_frames: int, hop_length: int, sr: int,
                            class_map: Dict[str,int]) -> np.ndarray:
    K = len(class_map)
    y = np.zeros((K, n_frames), dtype=np.float32)
    for ev in events:
        if ev.label not in class_map:
            continue
        k = class_map[ev.label]
        i0 = int(np.floor(ev.onset * sr / hop_length))
        i1 = int(np.ceil(ev.offset * sr / hop_length))
        i0 = max(i0, 0); i1 = min(i1, n_frames)
        if i1 > i0:
            y[k, i0:i1] = 1.0
    return y
