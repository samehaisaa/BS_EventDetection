import os, math
import numpy as np
import pandas as pd
from .io import load_labels, wav_duration
from .metrics import pairwise_overlap_seconds, union_annotated_seconds
from .windowing import frame_counts

def percentiles(x, qs=(0,25,50,75,100)):
    if x.size==0:
        return {f"p{q}":0.0 for q in qs}
    return {f"p{q}":float(np.percentile(x,q)) for q in qs}

def per_recording_stats(files, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    rec_rows=[]; lab_rows=[]; ov_paths=[]; frame_rows=[]
    for rid, f in files.items():
        if not (os.path.exists(f["wav"]) and os.path.exists(f["txt"])):
            continue
        df = load_labels(f["txt"])
        dur = wav_duration(f["wav"])
        labs = sorted(df["label"].unique().tolist())
        union = union_annotated_seconds(df)
        cov = union/dur if dur>0 else 0.0
        for lab, sub in df.groupby("label"):
            d = sub["duration"].to_numpy()
            q = percentiles(d)
            lab_rows.append({
                "recording_id": rid,
                "label": lab,
                "count": int(len(sub)),
                "total_dur_s": float(d.sum()),
                "min_dur_s": float(d.min()) if d.size else 0.0,
                "p25_dur_s": q["p25"],
                "median_dur_s": q["p50"],
                "p75_dur_s": q["p75"],
                "max_dur_s": float(d.max()) if d.size else 0.0,
                "mean_dur_s": float(d.mean()) if d.size else 0.0,
            })
        rec_rows.append({
            "recording_id": rid,
            "audio_dur_s": dur,
            "audio_dur_min": dur/60.0,
            "n_events_total": int(len(df)),
            "labels_present": ",".join(labs),
            "annot_union_s": union,
            "annot_coverage_pct": 100.0*cov,
        })
        ov = pairwise_overlap_seconds(df)
        p = os.path.join(out_dir, f"{rid}_pairwise_overlap_seconds.csv")
        ov.to_csv(p)
        ov_paths.append({"recording_id": rid, "overlap_csv": p})
        hop = 0.01
        t_end = float(df["end"].max()) if len(df) else dur
        cnt = frame_counts(df, hop_s=hop, t_end=t_end)
        cnt["recording_id"]=rid; cnt["hop_s"]=hop
        frame_rows.append(cnt)
    rec_df = pd.DataFrame(rec_rows).sort_values("recording_id")
    lab_df = pd.DataFrame(lab_rows).sort_values(["recording_id","label"])
    frm_df = pd.DataFrame(frame_rows).sort_values("recording_id")
    rec_df.to_csv(os.path.join(out_dir, "dataset_per_recording_stats.csv"), index=False)
    lab_df.to_csv(os.path.join(out_dir, "dataset_per_label_stats.csv"), index=False)
    frm_df.to_csv(os.path.join(out_dir, "dataset_frame_counts_10ms.csv"), index=False)
    return rec_df, lab_df, frm_df, ov_paths
