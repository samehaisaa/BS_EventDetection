import torch, torch.nn as nn
from .data import build_loaders
from .models.crnn import CRNNLite
from .metrics import frame_f1

def train_crnn(DATA, n_mels=128, epochs=10, batch_size=8, lr=1e-3, weight_decay=1e-4,
               rnn_hidden=64, rnn_layers=1, out_path="crnn_spotter.pt", device=None):
    if device is None: device = "cuda" if torch.cuda.is_available() else "cpu"
    train_loader, val_loader = build_loaders(DATA, val_ratio=0.2, batch_size=batch_size, augment=True)
    model = CRNNLite(n_mels=n_mels, rnn_hidden=rnn_hidden, rnn_layers=rnn_layers).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=max(1, epochs-1))
    loss_fn = nn.BCEWithLogitsLoss()

    best_f1, best_state = -1.0, None
    for ep in range(1, epochs+1):
        model.train(); tr_loss = 0.0; n_batches = 0
        for X,y in train_loader:
            X,y = X.to(device), y.to(device)
            logits = model(X); loss = loss_fn(logits, y)
            optim.zero_grad(set_to_none=True); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optim.step()
            tr_loss += loss.item(); n_batches += 1
        sched.step()

        model.eval()
        with torch.no_grad():
            all_logits, all_targets = [], []
            for X,y in val_loader:
                X,y = X.to(device), y.to(device)
                L = model(X)
                all_logits.append(L.cpu()); all_targets.append(y.cpu())
            if all_logits:
                L = torch.cat(all_logits, dim=0); T = torch.cat(all_targets, dim=0)
                f1,p,r = frame_f1(L, T, thr=0.5)
                val_loss = nn.BCEWithLogitsLoss()(L, T).item()
            else:
                f1=p=r=val_loss=0.0
        print(f"Epoch {ep:02d} | train {tr_loss/max(1,n_batches):.4f} | val {val_loss:.4f} | F1 {f1:.3f} (P {p:.3f} / R {r:.3f})")
        if f1 > best_f1:
            best_f1 = f1
            best_state = {k: v.cpu() for k,v in model.state_dict().items()}
            torch.save(best_state, out_path)
    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"Best val frame-F1: {best_f1:.3f} | saved to {out_path}")
    return model.to(device)
