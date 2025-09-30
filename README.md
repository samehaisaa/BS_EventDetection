# BSED — Bowel Sound Event Detection (CLI)

Command-line mirror of the notebook pipeline: preprocessing, visualization, per-frame **binary** labels (50% rule), lightweight **CRNN** training, inference + eventization, coverage stats, and **t-SNE** feature visualization.

> Python ≥ 3.9. GPU optional (PyTorch will use CPU if CUDA not available).

---

## Install

```bash
# clone your repo, then:
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# or (editable install if you want 'import bsed'):
pip install -e .
```

## Data Layout

Place `.wav` and matching `.txt` labels in the same base directory:

```
/path/to/dige/
├── file1.wav
├── file1.txt
├── file2.wav
└── file2.txt
```

### Label File Format

Label file format (`fileX.txt`): whitespace- or comma-separated `start end label` in seconds, e.g.:

```
0.38 0.505 b
0.96 2.015 mb
5.11 5.555 h
```

Classes are expected to be among `{b, mb, h}` (configurable).  
The t-SNE command additionally normalizes `b → sb` for that visualization pipeline.

---

## Configuration

Defaults live in `configs/default.yaml`:

```yaml
audio:  { sr: 16000, seg_dur_s: 10.0, hp_cut_hz: 60.0 }
mel:    { n_mels: 128, win_ms: 25.0, hop_ms: 10.0 }
labels: { wanted: ["b", "mb", "h"] }
binary: { overlap_threshold: 0.5 }
train:
  epochs: 20
  batch_size: 8
  lr: 0.001
  weight_decay: 0.0001
  rnn_hidden: 64
  rnn_layers: 1
inference:
  thr_open: 0.5
  thr_close: 0.5
  gap_merge_s: 0.100
  min_dur_s: 0.018
```

Override any value by editing the YAML or by passing CLI flags where available.  
Every command accepts `--cfg` to point to a different YAML.

---

## Global CLI

```bash
python main.py -h
```

**Global required option for every subcommand:**

- `--base-dir PATH`: Folder containing your `.wav` + `.txt`
- `--cfg PATH`: (optional) Config YAML (default: `configs/default.yaml`)

---

## Commands (with all options)

### 1) `info` — dataset summary

Prints sampling info, number of 10s segments per file, and mel shapes.

```bash
python main.py info \
  --base-dir /path/to/dige \
  [--cfg configs/default.yaml]
```

---

### 2) `plot-seg` — waveform + mel of a segment

```bash
python main.py plot-seg \
  --base-dir /path/to/dige \
  [--cfg configs/default.yaml] \
  [--key wav1] \
  [--idx 0]
```

- `--key`: which file key (auto-assigned as `wav1`, `wav2`, …)
- `--idx`: segment index within that file

---

### 3) `preview-grid` — grid of first N mels (quick scan)

```bash
python main.py preview-grid \
  --base-dir /path/to/dige \
  [--cfg configs/default.yaml] \
  [--key wav1] \
  [--n 4]
```

---

### 4) `labels-gantt` — waveform + label Gantt + mel

```bash
python main.py labels-gantt \
  --base-dir /path/to/dige \
  [--cfg configs/default.yaml] \
  [--key wav1] \
  [--idx 0]
```

---

### 5) `build-binary` — per-frame BS vs NULL labels (50% overlap rule)

Creates `frame_targets` and `frame_times_s` in memory for all segments.

```bash
python main.py build-binary \
  --base-dir /path/to/dige \
  [--cfg configs/default.yaml]
```

Uses `mel.win_ms`, `mel.hop_ms`, `audio.seg_dur_s`, and `binary.overlap_threshold` from config.

---

### 6) `binary-gantt` — visualize BS (1) vs NULL (0) for one segment

```bash
python main.py binary-gantt \
  --base-dir /path/to/dige \
  [--cfg configs/default.yaml] \
  [--key wav1] \
  [--idx 0]
```

If you didn't run `build-binary` yet, this command will compute it for you.

---

### 7) `train` — train CRNN-lite on frame targets

```bash
python main.py train \
  --base-dir /path/to/dige \
  [--cfg configs/default.yaml] \
  [--epochs 20] \
  [--batch-size 8] \
  [--lr 0.001] \
  [--weight-decay 0.0001] \
  [--rnn-hidden 64] \
  [--rnn-layers 1] \
  [--out crnn_spotter.pt]
```

**Notes:**

- Uses an 80/20 split across segments by default.
- Saves best-F1 weights to `--out` (default `crnn_spotter.pt`).

---

### 8) `coverage` — frame coverage (BS ratio)

Prints a table of total frames vs BS frames per file.

```bash
python main.py coverage \
  --base-dir /path/to/dige \
  [--cfg configs/default.yaml]
```

---

### 9) `prob-track` — plot p(BS) over time for a segment

```bash
python main.py prob-track \
  --base-dir /path/to/dige \
  [--cfg configs/default.yaml] \
  --model crnn_spotter.pt \
  [--key wav1] \
  [--idx 0] \
  [--thr 0.5]
```

- `--model`: path to saved `.pt` weights
- `--thr`: horizontal reference line for visualization

---

### 10) `gt-vs-pred` — Gantt of GT vs predicted events

Converts frame probabilities to events with simple thresholding + gap-merge + min-dur.

```bash
python main.py gt-vs-pred \
  --base-dir /path/to/dige \
  [--cfg configs/default.yaml] \
  --model crnn_spotter.pt \
  [--key wav2] \
  [--idx 0] \
  [--thr 0.5] \
  [--gap-merge 0.100] \
  [--min-dur 0.018]
```

- `--thr`: frame prob threshold → mask
- `--gap-merge`: merge consecutive events if the gap ≤ this (seconds)
- `--min-dur`: drop events shorter than this (seconds)

---

### 11) `tsne` — feature extraction + t-SNE plot

Extracts temporal + spectral + MFCC(+Δ,+ΔΔ) descriptors per labeled segment and saves a 2-panel PNG (t-SNE scatter + class histogram).

```bash
python main.py tsne \
  --base-dir /path/to/dige \
  [--cfg configs/default.yaml] \
  [--out bowel_sound_tsne.png]
```

The internal t-SNE pipeline normalizes `b → sb` so output classes are `{sb, mb, h}` in that plot only.

---

## End-to-end Quickstarts

### A) Visual sanity-check of data + labels

```bash
python main.py info --base-dir /path/to/dige
python main.py plot-seg --base-dir /path/to/dige --key wav1 --idx 0
python main.py labels-gantt --base-dir /path/to/dige --key wav1 --idx 0
```

### B) Build frame labels → train → inspect model

```bash
python main.py build-binary --base-dir /path/to/dige
python main.py train --base-dir /path/to/dige --epochs 10 --out crnn_spotter.pt
python main.py prob-track --base-dir /path/to/dige --model crnn_spotter.pt --key wav1 --idx 0 --thr 0.5
python main.py gt-vs-pred --base-dir /path/to/dige --model crnn_spotter.pt --key wav2 --idx 21 --thr 0.5 --gap-merge 0.05 --min-dur 0.02
python main.py coverage --base-dir /path/to/dige
```

### C) Feature exploration (t-SNE)

```bash
python main.py tsne --base-dir /path/to/dige --out docs/figs/tsne_classes.png
```

---

## Notes & Tips

- **Mel geometry**: `win_ms=25`, `hop_ms=10` ⇒ ~100 fps; 10 s segments ≈ 1000 frames.
- **Binary (50% rule)**: a frame is BS if its temporal overlap with any label ≥ 50% of the frame window.
- **Edge cases**: The code pads very short segments for MFCC deltas; plotting functions auto-show with matplotlib.
- **Classes**: You can restrict/rename classes via `labels.wanted` in `configs/default.yaml`.

---

## Help

Any command supports `-h` for help, e.g.:

```bash
python main.py gt-vs-pred -h
```

This will print the same options listed above with defaults.