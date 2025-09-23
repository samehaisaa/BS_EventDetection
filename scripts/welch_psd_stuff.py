import argparse, pathlib, numpy as np, pandas as pd
from scipy.io import wavfile
from scipy.signal import welch

def welchy(a, sr, nper=2048, nover=None):
    if nover is None: nover = nper // 2
    f, p = welch(np.asarray(a, float), fs=sr, nperseg=nper, noverlap=nover)
    return f, p

def munch_folder(d):
    files = sorted([p for p in d.glob("*.wav")])
    return files

def glue_psds(file_list):
    bank = []
    freq_ref = None
    wts = []
    for p in file_list:
        sr, y = wavfile.read(str(p))
        if y.ndim>1: y = y.mean(axis=1)
        f, psd = welchy(y, sr)
        if freq_ref is None: freq_ref = f
        if len(f)!=len(freq_ref) or np.max(np.abs(f-freq_ref))>1e-9: continue
        dur = len(y)/sr
        bank.append(psd * dur)
        wts.append(dur)
    if not bank: return None, None
    bank = np.vstack(bank)
    wts = np.asarray(wts)[:,None]
    agg = bank.sum(axis=0) / (wts.sum())
    return freq_ref, agg

def read_full(wav_path):
    sr, y = wavfile.read(str(wav_path))
    if y.ndim>1: y = y.mean(axis=1)
    f, p = welchy(y, sr)
    return f, p

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--segments_dir", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--full_wav", default=None)
    ap.add_argument("--classes", nargs="*", default=None)
    args = ap.parse_args()

    root = pathlib.Path(args.segments_dir)
    groups = [d for d in root.iterdir() if d.is_dir()]
    if args.classes: groups = [root / c for c in args.classes if (root / c).is_dir()]

    records = {}
    fgrid = None
    for g in sorted(groups, key=lambda z: z.name):
        files = munch_folder(g)
        f, p = glue_psds(files)
        if f is None: continue
        if fgrid is None: fgrid = f
        records[g.name] = p

    if args.full_wav:
        f_full, p_full = read_full(args.full_wav)
        if fgrid is None: fgrid = f_full
        if len(f_full)==len(fgrid) and np.max(np.abs(f_full-fgrid))<1e-9:
            records["full"] = p_full

    if fgrid is None or not records: 
        return

    data = {"freq_hz": fgrid}
    for k,v in records.items():
        data[f"psd_{k}"] = v
    df = pd.DataFrame(data)
    pathlib.Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_csv, index=False)

if __name__ == "__main__":
    main()
