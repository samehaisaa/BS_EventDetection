#!/usr/bin/env python3
import argparse, json, os, sys, yaml
from pathlib import Path

# Make src importable when running from repo root
sys.path.append(str(Path(__file__).parent / "src"))

from bsed.pipeline import build_DATA
from bsed.viz import show_segment_wave_and_mel, preview_grid, show_segment_wave_mel_with_labels, show_binary_gantt_for_segment, plot_prob_track, show_gantt_gt_vs_pred
from bsed.frames import build_binary_targets_inplace
from bsed.train import train_crnn
from bsed.infer import load_model, predict_probs_on_DATA, infer_events_for_DATA
from bsed.tsne_features import run_tsne_pipeline
from bsed.metrics import print_bs_coverage

def load_cfg(cfg_path):
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)

def common_parser(sub):
    sub.add_argument("--base-dir", type=Path, required=True, help="Folder with *.wav + *.txt")
    sub.add_argument("--cfg", type=Path, default=Path("configs/default.yaml"))

def main():
    p = argparse.ArgumentParser("BSED CLI")
    sp = p.add_subparsers(dest="cmd", required=True)

    # info
    s = sp.add_parser("info", help="Print sampling info and segment/mel shapes")
    common_parser(s)

    # plot segment
    s = sp.add_parser("plot-seg", help="Waveform+mel figure for a segment")
    common_parser(s)
    s.add_argument("--key", default="wav1")
    s.add_argument("--idx", type=int, default=0)

    # preview grid
    s = sp.add_parser("preview-grid", help="Grid of first N mels")
    common_parser(s)
    s.add_argument("--key", default="wav1")
    s.add_argument("--n", type=int, default=4)

    # labels gantt composite
    s = sp.add_parser("labels-gantt", help="Wave+labels+mel tri-plot for a segment")
    common_parser(s)
    s.add_argument("--key", default="wav1")
    s.add_argument("--idx", type=int, default=0)

    # build binary frame targets
    s = sp.add_parser("build-binary", help="Build per-frame BS/NULL masks (50%% rule)")
    common_parser(s)

    # binary gantt
    s = sp.add_parser("binary-gantt", help="Gantt for BS (1) vs NULL (0) for a segment")
    common_parser(s)
    s.add_argument("--key", default="wav1")
    s.add_argument("--idx", type=int, default=0)

    # train model
    s = sp.add_parser("train", help="Train CRNN-lite spotter")
    common_parser(s)
    s.add_argument("--epochs", type=int)
    s.add_argument("--batch-size", type=int)
    s.add_argument("--lr", type=float)
    s.add_argument("--weight-decay", type=float)
    s.add_argument("--rnn-hidden", type=int)
    s.add_argument("--rnn-layers", type=int)
    s.add_argument("--out", type=Path, default=Path("crnn_spotter.pt"))

    # coverage table
    s = sp.add_parser("coverage", help="Frame coverage summary (BS vs NULL)")
    common_parser(s)

    # probability track
    s = sp.add_parser("prob-track", help="Plot p(BS) track with threshold for a segment")
    common_parser(s)
    s.add_argument("--model", type=Path, required=True)
    s.add_argument("--key", default="wav1")
    s.add_argument("--idx", type=int, default=0)
    s.add_argument("--thr", type=float, default=0.5)

    # gt vs pred gantt
    s = sp.add_parser("gt-vs-pred", help="Gantt: GT vs predicted events for a segment")
    common_parser(s)
    s.add_argument("--model", type=Path, required=True)
    s.add_argument("--key", default="wav2")
    s.add_argument("--idx", type=int, default=0)
    s.add_argument("--thr", type=float, default=0.5)
    s.add_argument("--gap-merge", type=float, default=0.100)
    s.add_argument("--min-dur", type=float, default=0.018)

    # t-SNE feature viz
    s = sp.add_parser("tsne", help="Extract features + t-SNE; save PNG")
    common_parser(s)
    s.add_argument("--out", type=Path, default=Path("bowel_sound_tsne.png"))

    args = p.parse_args()
    cfg = load_cfg(args.cfg)

    DATA = build_DATA(
        base_dir=args.base_dir,
        sr=cfg["audio"]["sr"],
        seg_dur_s=cfg["audio"]["seg_dur_s"],
        hp_cut_hz=cfg["audio"]["hp_cut_hz"],
        n_mels=cfg["mel"]["n_mels"],
        win_ms=cfg["mel"]["win_ms"],
        hop_ms=cfg["mel"]["hop_ms"],
        wanted=set(cfg["labels"]["wanted"]),
    )

    if args.cmd == "info":
        # Print basic info already produced by build_DATA
        # (counts, shapes warnings are logged there)
        pass

    elif args.cmd == "plot-seg":
        show_segment_wave_and_mel(DATA, key=args.key, idx=args.idx)

    elif args.cmd == "preview-grid":
        preview_grid(DATA, key=args.key, n=args.n)

    elif args.cmd == "labels-gantt":
        show_segment_wave_mel_with_labels(DATA, key=args.key, idx=args.idx)

    elif args.cmd == "build-binary":
        build_binary_targets_inplace(DATA,
            win_ms=cfg["mel"]["win_ms"],
            hop_ms=cfg["mel"]["hop_ms"],
            seg_len_s=cfg["audio"]["seg_dur_s"],
            overlap_threshold=cfg["binary"]["overlap_threshold"],
            wanted_labels=tuple(cfg["labels"]["wanted"]))
        print("Built frame_targets for all segments.")

    elif args.cmd == "binary-gantt":
        if "frame_targets" not in DATA[args.key]:
            build_binary_targets_inplace(DATA,
                win_ms=cfg["mel"]["win_ms"],
                hop_ms=cfg["mel"]["hop_ms"],
                seg_len_s=cfg["audio"]["seg_dur_s"],
                overlap_threshold=cfg["binary"]["overlap_threshold"],
                wanted_labels=tuple(cfg["labels"]["wanted"]))
        show_binary_gantt_for_segment(DATA, key=args.key, idx=args.idx)

    elif args.cmd == "train":
        if args.epochs: cfg["train"]["epochs"] = args.epochs
        if args.batch_size: cfg["train"]["batch_size"] = args.batch_size
        if args.lr: cfg["train"]["lr"] = args.lr
        if args.weight_decay: cfg["train"]["weight_decay"] = args.weight_decay
        if args.rnn_hidden: cfg["train"]["rnn_hidden"] = args.rnn_hidden
        if args.rnn_layers: cfg["train"]["rnn_layers"] = args.rnn_layers

        build_binary_targets_inplace(DATA,
            win_ms=cfg["mel"]["win_ms"], hop_ms=cfg["mel"]["hop_ms"],
            seg_len_s=cfg["audio"]["seg_dur_s"],
            overlap_threshold=cfg["binary"]["overlap_threshold"],
            wanted_labels=tuple(cfg["labels"]["wanted"]))
        train_crnn(DATA, n_mels=cfg["mel"]["n_mels"], **cfg["train"], out_path=str(args.out))

    elif args.cmd == "coverage":
        build_binary_targets_inplace(DATA,
            win_ms=cfg["mel"]["win_ms"], hop_ms=cfg["mel"]["hop_ms"],
            seg_len_s=cfg["audio"]["seg_dur_s"],
            overlap_threshold=cfg["binary"]["overlap_threshold"],
            wanted_labels=tuple(cfg["labels"]["wanted"]))
        print_bs_coverage(DATA)

    elif args.cmd == "prob-track":
        model = load_model(args.model, n_mels=cfg["mel"]["n_mels"],
                           rnn_hidden=cfg["train"]["rnn_hidden"],
                           rnn_layers=cfg["train"]["rnn_layers"])
        probs = predict_probs_on_DATA(DATA, model)
        plot_prob_track(DATA, probs, key=args.key, idx=args.idx, thr=args.thr)

    elif args.cmd == "gt-vs-pred":
        model = load_model(args.model, n_mels=cfg["mel"]["n_mels"],
                           rnn_hidden=cfg["train"]["rnn_hidden"],
                           rnn_layers=cfg["train"]["rnn_layers"])
        probs = predict_probs_on_DATA(DATA, model)
        show_gantt_gt_vs_pred(
            DATA, key=args.key, idx=args.idx, probs=probs[args.key][args.idx],
            thr=args.thr,
            gap_merge_s=args.gap_merge,
            min_dur_s=args.min_dur
        )

    elif args.cmd == "tsne":
        out = run_tsne_pipeline(base_dir=args.base_dir, out_png=args.out)
        print(f"Saved t-SNE to: {out}")

if __name__ == "__main__":
    main()
