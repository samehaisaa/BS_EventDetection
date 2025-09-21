import numpy as np
import pandas as pd
from scipy.io import wavfile

def read_wav(path):
    sr, x = wavfile.read(path)
    if x.ndim > 1:
        x = x[:,0]
    return sr, x

def wav_duration(path):
    sr, x = read_wav(path)
    return float(len(x)) / float(sr)

def load_labels(path):
    df = pd.read_csv(path, sep=r"\s+|\t|,", engine="python", header=None).iloc[:,:3]
    df.columns = ["start","end","label"]
    df["start"] = pd.to_numeric(df["start"], errors="coerce")
    df["end"] = pd.to_numeric(df["end"], errors="coerce")
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    df = df.dropna(subset=["start","end"]).reset_index(drop=True)
    df.loc[df["label"]=="sbs","label"]="sb"
    swap = df["end"] < df["start"]
    if swap.any():
        df.loc[swap,["start","end"]] = df.loc[swap,["end","start"]].values
    df["duration"] = df["end"] - df["start"]
    return df.sort_values("start").reset_index(drop=True)
