import numpy as np, torch
from torch.utils.data import Dataset, DataLoader

class MelFrameDataset(Dataset):
    def __init__(self, DATA, index_map, augment=False):
        self.DATA = DATA; self.index_map = index_map; self.augment = augment

    def __len__(self): return len(self.index_map)

    def _augment(self, X):
        Xm = X.copy(); n_mels, T = Xm.shape
        if np.random.rand() < 0.5:
            f = np.random.randint(4, min(24, n_mels//2)); f0 = np.random.randint(0, n_mels - f); Xm[f0:f0+f,:] = 0.0
        if np.random.rand() < 0.5:
            t = max(1, int(0.1*T)); t0 = np.random.randint(0, T - t); Xm[:, t0:t0+t] = 0.0
        if np.random.rand() < 0.5:
            Xm = Xm + 0.01*np.random.randn(*Xm.shape).astype(Xm.dtype)
        return Xm

    def __getitem__(self, idx):
        key, seg_idx = self.index_map[idx]
        X = self.DATA[key]["specs_z"][seg_idx]
        y = self.DATA[key]["frame_targets"][seg_idx].astype(np.float32)
        if self.augment: X = self._augment(X)
        X = torch.tensor(X, dtype=torch.float32).unsqueeze(0)
        y = torch.tensor(y, dtype=torch.float32)
        return X, y

def collect_index_map(DATA):
    idx = []
    for k,it in DATA.items():
        n = min(len(it.get("specs_z", [])), len(it.get("frame_targets", [])))
        for i in range(n): idx.append((k,i))
    return idx

def build_loaders(DATA, val_ratio=0.2, batch_size=8, augment=True, seed=1337):
    import numpy as np
    index_map = collect_index_map(DATA)
    n_total = len(index_map); assert n_total > 1, "Need at least 2 segments"
    rng = np.random.default_rng(seed); rng.shuffle(index_map)
    n_val = max(1, int(round(val_ratio*n_total))); n_train = max(1, n_total - n_val)
    train_ds = MelFrameDataset(DATA, index_map[:n_train], augment=augment)
    val_ds   = MelFrameDataset(DATA, index_map[n_train:], augment=False)
    return (DataLoader(train_ds, batch_size=batch_size, shuffle=True),
            DataLoader(val_ds, batch_size=batch_size, shuffle=False))
