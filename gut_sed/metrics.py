import numpy as np
import pandas as pd

def interval_overlap(a, b):
    s1,e1 = a; s2,e2 = b
    return max(0.0, min(e1,e2) - max(s1,s2))

def pairwise_overlap_seconds(df):
    labs = sorted(df["label"].unique().tolist())
    m = pd.DataFrame(0.0, index=labs, columns=labs)
    by = {k: df[df["label"]==k][["start","end"]].to_numpy() for k in labs}
    for i,li in enumerate(labs):
        A = by[li]
        for lj in labs[i:]:
            B = by[lj]
            tot = 0.0
            for s1,e1 in A:
                for s2,e2 in B:
                    if e1<=s2 or e2<=s1:
                        continue
                    tot += interval_overlap((s1,e1),(s2,e2))
            m.loc[li,lj]=tot
            m.loc[lj,li]=tot
    return m

def union_annotated_seconds(df):
    if df.empty:
        return 0.0
    segs = df[["start","end"]].sort_values("start").to_numpy().tolist()
    out = []
    s,e = segs[0]
    for u,v in segs[1:]:
        if u<=e:
            e = max(e,v)
        else:
            out.append((s,e))
            s,e = u,v
    out.append((s,e))
    return float(sum(v-u for u,v in out))
