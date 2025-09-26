# Gut SED – Dataset Investigation

This repo right now is just for poking at the dataset before we build the actual ML pipeline.  
Main branch will later be used for the real pipeline, and this early work will move to its own branch.

---

## What’s here

- **gut_sed/**: small helper package
  - `io.py` – load wav + labels
  - `filters.py` – band-pass filter + envelope
  - `windowing.py` – slice labels, count frames
  - `metrics.py` – overlaps + coverage
  - `stats.py` – compute per-recording stats
  - `viz.py` – quick plots

- **scripts/**
  - `inspect_filter.py` – show raw vs. filtered signal with labels
  - `dataset_stats.py` – dump stats to csvs

---

## Why

- check signal values, spikes, ranges  
- test band-pass filtering (~80–1200 Hz) to see bursts more clearly  
- look at label coverage, event durations, overlaps  
- prep CSV summaries ?

---

## Log-Mel Spectrogram Tool

Added a new utility to visualize audio as a log-mel spectrogram.  
It supports the full signal or a chosen event segment, with adjustable FFT/hop settings, frequency limits, time zoom, and optional window overlays.  
Also includes dB dynamic range clipping for clear visuals ..  exploring time–frequency content and comparing different classes of events.


## Next steps

- finish dataset exploration  
- branch this off as `investigation`  
- keep `main` clean for the ML pipeline
# BS_EventDetection
