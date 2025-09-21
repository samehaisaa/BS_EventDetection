import numpy as np
import matplotlib.pyplot as plt

def plot_wave_with_labels(t, x, df, legend=True, title=None):
    plt.plot(t, x, color="gray")
    y0,y1=plt.ylim()
    h=(y1-y0)*0.1
    labs=sorted(df["label"].unique())
    palette=plt.cm.tab10(np.linspace(0,1,len(labs)))
    cmap={k:palette[i] for i,k in enumerate(labs)}
    for _,r in df.iterrows():
        plt.fill_betweenx([y0,y0+h], r["start"], r["end"], color=cmap[r["label"]], alpha=0.5)
    if legend:
        for k,c in cmap.items():
            plt.plot([],[], color=c, lw=10, alpha=0.5, label=k)
        plt.legend(loc="upper right")
    if title:
        plt.title(title)
    plt.xlabel("Time [s]"); plt.ylabel("Amplitude"); plt.grid(True)

def plot_overlay(t, x, y, df, legend=True, title=None, y_label="Amplitude"):
    plt.plot(t, y, label="Filtered", alpha=0.8)
    y0,y1=plt.ylim()
    h=(y1-y0)*0.1
    labs=sorted(df["label"].unique())
    palette=plt.cm.tab10(np.linspace(0,1,len(labs)))
    cmap={k:palette[i] for i,k in enumerate(labs)}
    for _,r in df.iterrows():
        plt.fill_betweenx([y0,y0+h], r["start"], r["end"], color=cmap[r["label"]], alpha=0.5)
    if legend:
        for k,c in cmap.items():
            plt.plot([],[], color=c, lw=10, alpha=0.5, label=k)
        plt.legend(loc="upper right")
    if title:
        plt.title(title)
    plt.xlabel("Time [s]"); plt.ylabel(y_label); plt.grid(True)
