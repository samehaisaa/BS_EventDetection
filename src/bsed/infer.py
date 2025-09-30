import numpy as np, torch
from .models.crnn import CRNNLite

def load_model(path, n_mels=128, rnn_hidden=64, rnn_layers=1, device=None):
    if device is None: device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CRNNLite(n_mels=n_mels, rnn_hidden=rnn_hidden, rnn_layers=rnn_layers).to(device)
    state = torch.load(str(path), map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model

def predict_probs_on_DATA(DATA, model, device=None):
    if device is None: device = "cuda" if torch.cuda.is_available() else "cpu"
    all_probs = {}
    with torch.no_grad():
        for key, it in DATA.items():
            probs = []
            for X in it["specs_z"]:
                Xt = torch.tensor(X, dtype=torch.float32)[None,None].to(device)
                logits = model(Xt).cpu().squeeze(0).numpy()
                probs.append(1.0/(1.0+np.exp(-logits)))
            all_probs[key] = probs
    return all_probs

def mask_to_events(prob, sr, hop_ms=10.0, win_ms=25.0, thr_open=0.5, thr_close=None,
                   gap_merge_s=0.100, min_dur_s=0.018):
    if thr_close is None: thr_close = thr_open
    hop_s, win_s = hop_ms/1000.0, win_ms/1000.0
    T = prob.shape[0]
    open_mask = np.zeros(T, dtype=np.uint8); state = 0
    for i in range(T):
        p = prob[i]
        if state==0 and p>=thr_open: state=1
        elif state==1 and p<thr_close: state=0
        open_mask[i] = state
    starts = np.where(np.diff(np.r_[0, open_mask, 0])==1)[0]
    ends   = np.where(np.diff(np.r_[0, open_mask, 0])==-1)[0]
    intervals = []
    for s_idx, e_idx in zip(starts, ends):
        start_t = s_idx*hop_s
        end_t   = (e_idx-1)*hop_s + win_s
        intervals.append([start_t, end_t])
    merged = []
    for st, en in intervals:
        if not merged: merged.append([st,en]); continue
        if st - merged[-1][1] <= gap_merge_s:
            merged[-1][1] = max(merged[-1][1], en)
        else:
            merged.append([st,en])
    return [(st,en) for st,en in merged if (en-st) >= min_dur_s]

def infer_events_for_DATA(DATA, all_probs, thr_open=0.5, thr_close=0.5,
                          gap_merge_s=0.100, min_dur_s=0.018):
    all_events = {}
    for key, plist in all_probs.items():
        sr = DATA[key]["sr"]; evs = []
        for p in plist:
            evs.append(mask_to_events(p, sr=sr, hop_ms=10.0, win_ms=25.0,
                                      thr_open=thr_open, thr_close=thr_close,
                                      gap_merge_s=gap_merge_s, min_dur_s=min_dur_s))
        all_events[key] = evs
    return all_events

def probs_to_events_simple(prob, thr=0.5, hop_ms=10.0, win_ms=25.0, gap_merge_s=0.100, min_dur_s=0.018):
    hop_s, win_s = hop_ms/1000.0, win_ms/1000.0
    m = (prob >= thr).astype(np.int32)
    d = np.diff(np.r_[0, m, 0])
    starts, ends = np.where(d==1)[0], np.where(d==-1)[0]
    intervals = [(s*hop_s, (e-1)*hop_s + win_s) for s,e in zip(starts, ends)]
    merged = []
    for st,en in intervals:
        if not merged or st - merged[-1][1] > gap_merge_s:
            merged.append([st,en])
        else:
            merged[-1][1] = max(merged[-1][1], en)
    return [(st,en) for st,en in merged if (en-st) >= min_dur_s]
