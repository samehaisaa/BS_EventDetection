import torch
import torch.nn as nn
import torch.nn.functional as F

class CRNNSED(nn.Module):
    def __init__(self, n_mels, n_classes, c=(32,64,128), rnn_h=256, rnn_l=2, p=0.3, agg="max"):
        super().__init__()
        self.b1 = nn.Sequential(nn.Conv2d(1, c[0], 3, padding=1), nn.BatchNorm2d(c[0]), nn.ReLU(), nn.Conv2d(c[0], c[0], 3, padding=1), nn.BatchNorm2d(c[0]), nn.ReLU(), nn.MaxPool2d((2,2)), nn.Dropout(p))
        self.b2 = nn.Sequential(nn.Conv2d(c[0], c[1], 3, padding=1), nn.BatchNorm2d(c[1]), nn.ReLU(), nn.Conv2d(c[1], c[1], 3, padding=1), nn.BatchNorm2d(c[1]), nn.ReLU(), nn.MaxPool2d((2,2)), nn.Dropout(p))
        self.b3 = nn.Sequential(nn.Conv2d(c[1], c[2], 3, padding=1), nn.BatchNorm2d(c[2]), nn.ReLU(), nn.Conv2d(c[2], c[2], 3, padding=1), nn.BatchNorm2d(c[2]), nn.ReLU(), nn.MaxPool2d((2,2)), nn.Dropout(p))
        f_out = (n_mels // 8)
        d_in = c[2] * f_out
        self.rnn = nn.GRU(d_in, rnn_h, num_layers=rnn_l, batch_first=True, dropout=p, bidirectional=True)
        self.hd = nn.Sequential(nn.Linear(2*rnn_h, 2*rnn_h), nn.ReLU(), nn.Dropout(p))
        self.fc = nn.Linear(2*rnn_h, n_classes)
        self.agg = agg

    def forward(self, x, lengths=None, return_clip=True):
        x = x.unsqueeze(1).transpose(2,3)
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        x = x.permute(0,3,1,2).contiguous()
        b,t,c,f = x.shape
        x = x.view(b,t,c*f)
        if lengths is not None:
            L = (lengths.float()/8).ceil().clamp(min=1, max=t).long()
            x_p = nn.utils.rnn.pack_padded_sequence(x, L.cpu(), batch_first=True, enforce_sorted=False)
            x_p,_ = self.rnn(x_p)
            x,_ = nn.utils.rnn.pad_packed_sequence(x_p, batch_first=True, total_length=t)
        else:
            x,_ = self.rnn(x)
        x = self.hd(x)
        y = torch.sigmoid(self.fc(x))
        if not return_clip:
            return y
        if lengths is not None:
            m = torch.arange(t, device=x.device).unsqueeze(0) >= L.unsqueeze(1)
            y_masked = y.masked_fill(m.unsqueeze(-1), 0)
            denom = L.clamp(min=1).unsqueeze(1).float()
            if self.agg == "mean":
                clip = y_masked.sum(1) / denom
            elif self.agg == "max":
                clip = y_masked.max(1).values
            else:
                clip = y_masked.sum(1) / denom
        else:
            clip = y.mean(1) if self.agg == "mean" else y.max(1).values
        return y, clip

def frame_bce(pred, target, lengths=None):
    if lengths is None:
        return F.binary_cross_entropy(pred, target)
    b,t,c = pred.shape
    L = lengths.clamp(min=1, max=t).long()
    m = torch.arange(t, device=pred.device).unsqueeze(0) < L.unsqueeze(1)
    pred = pred[m].view(-1,c)
    target = target[m].view(-1,c)
    return F.binary_cross_entropy(pred, target)

if __name__ == "__main__":
    B,T,Fm,C = 4, 200, 64, 3
    x = torch.randn(B,T,Fm)
    L = torch.tensor([200,180,150,120])
    y = torch.randint(0,2,(B,T,C)).float()
    m = CRNNSED(n_mels=Fm, n_classes=C)
    f, c = m(x, lengths=L)
    loss = frame_bce(f, y, lengths=L)
    print(f.shape, c.shape, float(loss))
