import os, pandas as pd, numpy as np

def load_labels_for_wav(wav_path, *, force_seconds=False, wanted=("b","mb","h")):
    txt_path = os.path.splitext(wav_path)[0] + ".txt"
    if not os.path.exists(txt_path):
        return pd.DataFrame(columns=["start","end","label"])
    df = pd.read_csv(txt_path, sep=r"[,\s]+", engine="python", header=None, comment="#")
    df = df.iloc[:, :3]; df.columns = ["start","end","label"]
    df["label"] = df["label"].astype(str).str.strip()
    df["start"] = pd.to_numeric(df["start"], errors="coerce")
    df["end"]   = pd.to_numeric(df["end"],   errors="coerce")
    df = df.dropna(subset=["start","end"]).reset_index(drop=True)
    df = df[df["label"].isin(wanted)].reset_index(drop=True)
    if not force_seconds and len(df):
        dur = (df["end"] - df["start"]).abs()
        looks_ms = (dur.median() > 5.0) or (max(df["start"].max(), df["end"].max()) > 300.0)
        if looks_ms:
            df["start"] /= 1000.0; df["end"] /= 1000.0
    return df

def labels_for_segment(labels_df, seg_idx, seg_len_s=10.0):
    t0, t1 = seg_idx*seg_len_s, seg_idx*seg_len_s + seg_len_s
    df = labels_df[(labels_df["end"] > t0) & (labels_df["start"] < t1)].copy()
    if df.empty: return df.assign(local_start=[], local_end=[])
    df["local_start"] = np.maximum(df["start"].values, t0) - t0
    df["local_end"]   = np.minimum(df["end"].values,   t1) - t0
    df = df[df["local_end"] > df["local_start"]].reset_index(drop=True)
    return df
