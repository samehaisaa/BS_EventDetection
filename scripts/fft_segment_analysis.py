	import argparse, numpy as np, pathlib, random
from scipy.io import wavfile
import matplotlib.pyplot as plt

def pick_one(d):
    q = sorted([p for p in d.glob("*") if p.suffix.lower()==".wav"])
    if not q: return None
    return q[0]

def fft_bits(x, sr):
    x = np.asarray(x, float)
    w = np.hanning(len(x))
    z = np.fft.rfft(w * x)
    f = np.fft.rfftfreq(len(x), 1.0/sr)
    m = np.abs(z)
    return f, m

def show_spectrum(bag, out_png):
    plt.figure()
    for name,(f,m) in bag.items():
        plt.plot(f, m, lw=1.2, label=str(name))
    plt.xlabel("Hz"); plt.ylabel("mag"); plt.legend(); plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()

def show_hist(bag, out_png, bins=100):
    plt.figure()
    for name,(f,m) in bag.items():
        mm = m / (m.sum() + 1e-12)
        h, edges = np.histogram(f, bins=bins, range=(0, f.max() if len(f) else 1), weights=mm)
        mids = 0.5*(edges[1:]+edges[:-1])
        plt.step(mids, h, where="mid", label=str(name))
    plt.xlabel("Hz"); plt.ylabel("weighted freq-count"); plt.legend(); plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--segments_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--classes", nargs="*", default=None)
    args = ap.parse_args()

    root = pathlib.Path(args.segments_dir)
    out = pathlib.Path(args.out); out.mkdir(parents=True, exist_ok=True)

    folders = [p for p in root.iterdir() if p.is_dir()]
    if args.classes: folders = [root / c for c in args.classes if (root / c).is_dir()]
    stash = {}
    for fdir in folders:
        p = pick_one(fdir)
        if p is None: continue
        sr, y = wavfile.read(str(p))
        if y.ndim>1: y = y.mean(axis=1)
        freqs, mags = fft_bits(y, sr)
        stash[fdir.name] = (freqs, mags)

    if not stash: return
    show_spectrum(stash, str(out / "fft_spectrum_all.png"))
    show_hist(stash, str(out / "histogram_of_frequency_components.png"))

if __name__ == "__main__":
    main()

