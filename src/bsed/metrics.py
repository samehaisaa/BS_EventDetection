import numpy as np, pandas as pd, torch

def frame_f1(pred, target, thr=0.5, eps=1e-8):
    if pred.dtype.is_floating_point:
        pred = (torch.sigmoid(pred) >= thr).to(target.dtype)
    tp = (pred*target).sum(); fp = (pred*(1-target)).sum(); fn = ((1-pred)*target).sum()
    precision = tp / (tp+fp+eps); recall = tp / (tp+fn+eps)
    f1 = 2*precision*recall / (precision+recall+eps)
    return f1.item(), precision.item(), recall.item()

def print_bs_coverage(DATA):
    import numpy as np, pandas as pd
    rows = []
    for key, item in DATA.items():
        masks = item.get("frame_targets", [])
        total = sum(int(np.asarray(m, dtype=np.uint8).size) for m in masks)
        bs = sum(int(np.asarray(m, dtype=np.uint8).sum()) for m in masks)
        rows.append({"file_key": key, "frames_total": total, "frames_bs": bs, "frames_non_bs": total-bs, "bs_ratio_frames": (bs/total if total else 0.0)})
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
