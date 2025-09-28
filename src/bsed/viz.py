from __future__ import annotations
from typing import List, Sequence, Optional
import numpy as np
import matplotlib.pyplot as plt
from .io import Event

def plot_spectrogram_with_events(S: np.ndarray, sr: int, hop_length: int,
                                 gt: Optional[List[Event]] = None,
                                 pred: Optional[List[Event]] = None,
                                 class_colors: Optional[dict] = None,
                                 title: str = ""):
    T = S.shape[1]
    times = np.arange(T) * hop_length / sr
    fig, ax = plt.subplots(figsize=(12, 5))
    im = ax.imshow(S, origin="lower", aspect="auto", extent=[times[0], times[-1], 0, S.shape[0]])
    ax.set_ylabel("mel bins")
    ax.set_xlabel("time (s)")
    ax.set_title(title or "Log-mel spectrogram")
    fig.colorbar(im, ax=ax)
    def _draw(evts, y0, y1):
        for e in evts:
            ax.axvspan(e.onset, e.offset, ymin=y0, ymax=y1, alpha=0.25, label=e.label)
    if gt:
        _draw(gt, 0.0, 0.15)
    if pred:
        _draw(pred, 0.85, 1.0)
    if gt or pred:
        # deduplicate legend
        handles, labels = ax.get_legend_handles_labels()
        uniq = {}
        for h,l in zip(handles, labels):
            uniq[l] = h
        ax.legend(uniq.values(), uniq.keys(), loc="upper right")
    plt.tight_layout()
    return fig, ax

def plot_gantt(events: List[Event], classes: Sequence[str]):
    fig, ax = plt.subplots(figsize=(10, 2 + 0.4*len(classes)))
    cls_to_y = {c:i for i,c in enumerate(classes)}
    for e in events:
        y = cls_to_y.get(e.label, 0)
        ax.broken_barh([(e.onset, e.offset - e.onset)], (y-0.4, 0.8))
    ax.set_yticks(range(len(classes)))
    ax.set_yticklabels(classes)
    ax.set_xlabel("time (s)")
    ax.set_title("Events Gantt")
    plt.tight_layout()
    return fig, ax
