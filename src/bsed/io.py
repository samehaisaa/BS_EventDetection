from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import json
import pandas as pd
import numpy as np
import soundfile as sf
import librosa

@dataclass
class Event:
    onset: float
    offset: float
    label: str

def read_wav(path: str, sr: Optional[int] = None) -> Tuple[np.ndarray, int]:
    if sr is None:
        y, file_sr = sf.read(path, always_2d=False)
        if y.ndim > 1:
            y = np.mean(y, axis=1)
        return y.astype(np.float32), file_sr
    y, _ = librosa.load(path, sr=sr, mono=True)
    return y.astype(np.float32), sr

def read_labels_txt(path: str, ignore_labels: Optional[List[str]] = None) -> List[Event]:
    ignore = set(ignore_labels or [])
    rows: List[Event] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"): 
                continue
            parts = [p.strip() for p in line.replace(",", " ").split() if p.strip()]
            if len(parts) < 3:
                continue
            try:
                onset_ms = float(parts[0])
                offset_ms = float(parts[1])
                label = parts[2]
            except Exception:
                # try CSV
                parts = [p.strip() for p in line.split(",")]
                onset_ms = float(parts[0]); offset_ms = float(parts[1]); label = parts[2]
            if label in ignore:
                continue
            rows.append(Event(onset_ms/1000.0, offset_ms/1000.0, label))
    return rows

def load_manifest(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    req = {"audio_path","label_path","split"}
    missing = req - set(df.columns)
    if missing:
        raise ValueError(f"Manifest missing columns: {missing}")
    return df

def save_events_json(path: str, events: List[Event], meta: Optional[Dict]=None) -> None:
    obj = {
        "events": [ {"onset": e.onset, "offset": e.offset, "label": e.label} for e in events ],
        "meta": meta or {},
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
