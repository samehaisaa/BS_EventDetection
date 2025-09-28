from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import os, math, random, time, json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from pathlib import Path
from tqdm import tqdm
from rich.console import Console
from .io import read_wav, read_labels_txt, load_manifest
from .preprocessing import pre_emphasis, highpass, logmel
from .labeling import build_class_map, events_to_frame_targets
from .models.crnn import CRNN

console = Console()

def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

class FrameDataset(Dataset):
    def __init__(self, df: pd.DataFrame, split: str, cfg: dict):
        self.df = df[df["split"]==split].reset_index(drop=True)
        self.cfg = cfg
        self.classes = cfg["data"]["classes"]
        self.class_map = build_class_map(self.classes)
        self.sr = cfg["data"]["sample_rate"]
        self.preemph = float(cfg["data"].get("preemph", 0.0))
        self.highpass_hz = float(cfg["data"].get("highpass_hz", 0.0))
        self.feat = cfg["features"]
        self.seq_len = int(cfg["train"]["seq_len_frames"])
        self.split = split

    def __len__(self): return len(self.df)

    def _compute(self, audio_path: str, label_path: str):
        y, sr = read_wav(audio_path, sr=self.sr)
        if self.preemph > 0: y = pre_emphasis(y, self.preemph)
        if self.highpass_hz > 0: y = highpass(y, self.sr, self.highpass_hz)
        S, hop = logmel(
            y, self.sr,
            n_mels=self.feat["n_mels"],
            n_fft=self.feat["n_fft"],
            hop_length=self.feat["hop_length"],
            fmin=self.feat["fmin"],
            fmax=self.feat.get("fmax", None),
            ref_level_db=self.feat.get("ref_level_db", 20.0),
            top_db=self.feat.get("top_db", 80.0),
            normalize=self.feat.get("normalize","instance")
        )
        # targets
        events = read_labels_txt(label_path, ignore_labels=[self.cfg["data"].get("bg_label")] if self.cfg["data"].get("bg_label") else None)
        tgt = events_to_frame_targets(events, S.shape[1], hop, self.sr, self.class_map)  # (K,T)
        return S, tgt

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        S, tgt = self._compute(row.audio_path, row.label_path)  # (F,T), (K,T)
        T = S.shape[1]
        if self.split == "train" and T > self.seq_len:
            i0 = np.random.randint(0, T - self.seq_len + 1)
            i1 = i0 + self.seq_len
            S = S[:, i0:i1]
            tgt = tgt[:, i0:i1]
        # ensure min length by pad
        if S.shape[1] < self.seq_len:
            pad = self.seq_len - S.shape[1]
            S = np.pad(S, ((0,0),(0,pad)), mode="edge")
            tgt = np.pad(tgt, ((0,0),(0,pad)), mode="edge")
        x = torch.from_numpy(S).unsqueeze(0)   # (1,F,T)
        y = torch.from_numpy(tgt).permute(1,0) # (T,K)
        return x, y

def train_main(cfg: dict):
    set_seed(int(cfg["train"]["seed"]))
    df = load_manifest(cfg["data"]["manifest"])
    classes = cfg["data"]["classes"]
    n_classes = len(classes)
    n_mels = int(cfg["features"]["n_mels"])

    tr_ds = FrameDataset(df, "train", cfg)
    va_ds = FrameDataset(df, "val", cfg)

    tr_dl = DataLoader(tr_ds, batch_size=cfg["train"]["batch_size"], shuffle=True, num_workers=cfg["train"]["num_workers"], pin_memory=True)
    va_dl = DataLoader(va_ds, batch_size=cfg["train"]["batch_size"], shuffle=False, num_workers=cfg["train"]["num_workers"], pin_memory=True)

    model = CRNN(n_mels=n_mels, n_classes=n_classes,
                 conv_channels=cfg["model"]["conv_channels"],
                 gru_hidden=cfg["model"]["gru_hidden"],
                 dropout=cfg["model"]["dropout"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    opt = torch.optim.Adam(model.parameters(), lr=float(cfg["train"]["lr"]))
    crit = nn.BCELoss()
    scaler = GradScaler(enabled=bool(cfg["train"].get("mixed_precision", True)))

    outdir = Path(cfg["train"]["outdir"])
    outdir.mkdir(parents=True, exist_ok=True)
    best = {"val_loss": float("inf"), "epoch": -1}
    log_rows = []

    epochs = int(cfg["train"]["epochs"])
    patience = int(cfg["train"]["early_stop_patience"])
    no_improve = 0

    console.log(f"Start training for {epochs} epochs on {device}")
    for epoch in range(1, epochs+1):
        model.train()
        tr_loss = 0.0
        for x, y in tqdm(tr_dl, desc=f"Epoch {epoch} [train]"):
            x = x.to(device)                 # (B,1,F,T)
            y = y.to(device)                 # (B,T,K)
            with autocast(enabled=bool(cfg["train"].get("mixed_precision", True))):
                p = model(x)                 # (B,T',K)
                Tm = min(p.shape[1], y.shape[1])
                loss = crit(p[:, :Tm, :], y[:, :Tm, :])
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).step(opt)
            scaler.update()
            tr_loss += loss.item() * x.size(0)
        tr_loss /= len(tr_ds)

        # val
        model.eval()
        va_loss = 0.0
        with torch.no_grad():
            for x, y in tqdm(va_dl, desc=f"Epoch {epoch} [val]"):
                x = x.to(device); y = y.to(device)
                p = model(x)
                Tm = min(p.shape[1], y.shape[1])
                loss = crit(p[:, :Tm, :], y[:, :Tm, :])
                va_loss += loss.item() * x.size(0)
        va_loss /= len(va_ds)

        console.log(f"Epoch {epoch:03d} | train {tr_loss:.4f} | val {va_loss:.4f}")
        log_rows.append({"epoch": epoch, "train_loss": tr_loss, "val_loss": va_loss})
        pd.DataFrame(log_rows).to_csv(outdir/"metrics.csv", index=False)

        if va_loss + 1e-6 < best["val_loss"]:
            best.update({"val_loss": va_loss, "epoch": epoch})
            torch.save({"model": model.state_dict(), "cfg": cfg}, outdir/"model_best.pt")
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= patience:
            console.log(f"Early stopping at epoch {epoch}")
            break

    console.log(f"Best @ epoch {best['epoch']} | val_loss={best['val_loss']:.4f}")
    return outdir
