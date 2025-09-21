import numpy as np
import pandas as pd

def slice_labels(df, t0, t1):
    m = (df["end"]>t0) & (df["start"]<t1)
    sub = df.loc[m].copy()
    sub["start"] = sub["start"].clip(lower=t0, upper=t1)
    sub["end"] = sub["end"].clip(lower=t0, upper=t1)
    return sub.reset_index(drop=True)

def frame_counts(df, hop_s, t_end):
    n = int(np.floor(t_end/hop_s))+1
    labs = sorted(df["label"].unique().tolist())
    c = {k:0 for k in labs}
    for k in labs:
        g = df[df["label"]==k][["start","end"]].to_numpy()
        for s,e in g:
            i = max(0, int(np.floor(s/hop_s)))
            j = min(n-1, int(np.ceil(e/hop_s)))
            if j>=i:
                c[k] += (j-i+1)
    c["_total_frames"] = n
    return c
