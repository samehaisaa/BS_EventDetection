import numpy as np

def _frame_geometry(sr: int, win_ms: float = 25.0, hop_ms: float = 10.0):
    win_len = int(round((win_ms/1000.0) * sr))
    hop_len = int(round((hop_ms/1000.0) * sr))
    return win_len, hop_len

def _frame_times_for_T(T: int, sr: int, hop_ms: float = 10.0, win_ms: float = 25.0):
    hop_len = int(round((hop_ms/1000.0) * sr))
    win_len = int(round((win_ms/1000.0) * sr))
    starts = np.arange(T) * (hop_len / sr)
    ends   = starts + (win_len / sr)
    return starts, ends

def build_binary_targets_inplace(DATA: dict, *, win_ms=25.0, hop_ms=10.0,
                                 seg_len_s=10.0, overlap_threshold=0.5,
                                 wanted_labels=("b","mb","h")):
    for key, item in DATA.items():
        sr = int(item["sr"])
        specs_list = item["specs_z"]
        labels_df = item.get("labels", None)
        if labels_df is not None and len(labels_df):
            labels_df = labels_df[labels_df["label"].isin(wanted_labels)].reset_index(drop=True)

        frame_targets, frame_times = [], []
        for seg_idx, Z in enumerate(specs_list):
            T = Z.shape[1]
            f_starts, f_ends = _frame_times_for_T(T, sr, hop_ms=hop_ms, win_ms=win_ms)
            win_s = (win_ms/1000.0); thresh = overlap_threshold * win_s
            # slice labels to this segment (local coords)
            if labels_df is not None and len(labels_df):
                t0, t1 = seg_idx*seg_len_s, (seg_idx+1)*seg_len_s
                L = labels_df[(labels_df["end"]>t0)&(labels_df["start"]<t1)].copy()
                if not L.empty:
                    L["local_start"] = np.maximum(L["start"].values, t0) - t0
                    L["local_end"]   = np.minimum(L["end"].values,   t1) - t0
                else:
                    L = None
            else:
                L = None

            mask = np.zeros(T, dtype=np.uint8)
            if L is not None and len(L):
                for _, row in L.iterrows():
                    s, e = float(row["local_start"]), float(row["local_end"])
                    overlap = np.minimum(f_ends, e) - np.maximum(f_starts, s)
                    overlap = np.maximum(overlap, 0.0)
                    mask |= (overlap >= thresh).astype(np.uint8)
            frame_targets.append(mask)
            frame_times.append(f_starts)
        item["frame_targets"] = frame_targets
        item["frame_times_s"] = frame_times
