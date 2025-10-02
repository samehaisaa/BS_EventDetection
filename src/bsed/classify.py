from __future__ import annotations
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

ALLOWED_CLASSES = {"sb", "b", "mb", "h"}  # we'll normalize 'b'→'sb' at training time, but keep robust

def _normalize_label(lbl: str) -> str:
    s = (lbl or "").strip().lower()
    return "sb" if s == "b" else s

def _feature_columns(df: pd.DataFrame) -> list[str]:
    exclude = {"event_id", "key", "seg_idx", "onset_s_local", "offset_s_local",
               "onset_s_global", "offset_s_global", "label", "file", "onset", "offset"}
    cols = [c for c in df.columns if c not in exclude and np.issubdtype(df[c].dtype, np.number)]
    return cols

def train_knn_on_gt(gt_feat_df: pd.DataFrame, k: int = 5, out_path: str | Path = "models/bs_clf.joblib") -> Path:
    df = gt_feat_df.copy()
    if "label" not in df.columns:
        raise ValueError("GT features must contain a 'label' column.")
    df["label"] = df["label"].astype(str).map(_normalize_label)
    df = df[df["label"].isin({"sb","mb","h"})].reset_index(drop=True)
    X_cols = _feature_columns(df)
    X = df[X_cols].values
    y = df["label"].values
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier(n_neighbors=k, weights="distance")),
    ])
    pipe.fit(X, y)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": pipe, "features": X_cols, "classes": sorted(set(y))}, out_path)
    return out_path

def predict_knn(features_df: pd.DataFrame, model_path: str | Path) -> pd.DataFrame:
    bundle = joblib.load(model_path)
    pipe: Pipeline = bundle["pipeline"]
    X_cols: list[str] = bundle["features"]
    classes: list[str] = bundle["classes"]
    # prepare
    X = features_df[X_cols].values
    proba = pipe.predict_proba(X) if hasattr(pipe.named_steps["knn"], "predict_proba") else None
    pred = pipe.predict(X)
    out = features_df[["event_id","key","seg_idx","onset_s_local","offset_s_local","onset_s_global","offset_s_global"]].copy()
    out["pred_label"] = pred
    if proba is not None:
        P = pd.DataFrame(proba, columns=[f"pred_proba_{c}" for c in pipe.classes_])
        out = pd.concat([out.reset_index(drop=True), P.reset_index(drop=True)], axis=1)
        # ensure all standard columns exist
        for c in {"sb","mb","h"}:
            col = f"pred_proba_{c}"
            if col not in out.columns:
                out[col] = 0.0
    return out
