# src/bsed/splits.py
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

def make_segment_split(DATA: dict, train=0.7, val=0.2, test=0.1, seed=1337) -> pd.DataFrame:
    """
    Per-file segment split. For each key (wavN), randomly assign segment indices to
    train/val/test in the given proportions (rounded per file).
    """
    assert abs(train + val + test - 1.0) < 1e-6, "ratios must sum to 1"
    rng = np.random.default_rng(int(seed))
    rows = []
    for key, item in DATA.items():
        n = len(item.get("segments_hp", []))
        idxs = np.arange(n)
        rng.shuffle(idxs)
        n_train = int(round(train * n))
        n_val   = int(round(val   * n))
        n_test  = max(0, n - n_train - n_val)
        split_arr = (["train"] * n_train) + (["val"] * n_val) + (["test"] * n_test)
        for seg_idx, split in zip(idxs, split_arr):
            rows.append({"key": key, "seg_idx": int(seg_idx), "split": split})
    df = pd.DataFrame(rows).sort_values(["key","seg_idx"]).reset_index(drop=True)
    return df

def save_split(df: pd.DataFrame, path: Path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)

def load_split(path: Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    return df[["key","seg_idx","split"]].copy()

def index_map_for(DATA: dict, split_df: pd.DataFrame, subset=("train",)) -> list[tuple[str,int]]:
    subs = set(subset) if isinstance(subset, (list, tuple, set)) else {subset}
    ok = split_df[split_df["split"].isin(subs)][["key","seg_idx"]]
    return [(r.key, int(r.seg_idx)) for r in ok.itertuples(index=False)]

def filter_events_by_split(events_df: pd.DataFrame, split_df: pd.DataFrame, subset=("test",)) -> pd.DataFrame:
    subs = set(subset) if isinstance(subset, (list, tuple, set)) else {subset}
    m = split_df[split_df["split"].isin(subs)][["key","seg_idx"]]
    out = events_df.merge(m, on=["key","seg_idx"], how="inner")
    return out
