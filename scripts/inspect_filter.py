import argparse
import numpy as np
import matplotlib.pyplot as plt
from gut_sed.io import read_wav, load_labels
from gut_sed.windowing import slice_labels
from gut_sed.filters import bandpass, envelope
from gut_sed.viz import plot_wave_with_labels, plot_overlay

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--wav", required=True)
    p.add_argument("--labels", required=True)
    p.add_argument("--t0", type=float, default=0.0)
    p.add_argument("--dur", type=float, default=4.0)
    p.add_argument("--low", type=float, default=100.0)
    p.add_argument("--high", type=float, default=1000.0)
    p.add_argument("--order", type=int, default=4)
    p.add_argument("--smooth_ms", type=float, default=10.0)
    args = p.parse_args()

    sr, x = read_wav(args.wav)
    n = int((args.t0+args.dur)*sr)
    x = x[:n]
    t = np.arange(len(x))/sr
    df = load_labels(args.labels)
    win = slice_labels(df, args.t0, args.t0+args.dur)
    y = bandpass(x[int(args.t0*sr):], sr, args.low, args.high, order=args.order)
    t_seg = t[int(args.t0*sr):int((args.t0+args.dur)*sr)]
    plt.figure(figsize=(14,8))
    plt.subplot(2,1,1)
    plot_wave_with_labels(t_seg, x[int(args.t0*sr):int((args.t0+args.dur)*sr)], win, title=f"Original (t0={args.t0}s, dur={args.dur}s)")
    plt.subplot(2,1,2)
    plot_overlay(t_seg, x[int(args.t0*sr):int((args.t0+args.dur)*sr)], y, win, title=f"Band-pass {int(args.low)}–{int(args.high)} Hz")
    plt.tight_layout()
    plt.show()

if __name__=="__main__":
    main()
