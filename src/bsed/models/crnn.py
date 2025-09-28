import torch, torch.nn as nn, torch.nn.functional as F

class CRNNLite(nn.Module):
    def __init__(self, n_mels=128, conv_chs=(32,64), rnn_hidden=64, rnn_layers=1, dropout=0.2):
        super().__init__()
        c1, c2 = conv_chs
        self.conv1 = nn.Conv2d(1, c1, 3, padding=1)
        self.bn1   = nn.BatchNorm2d(c1)
        self.conv2 = nn.Conv2d(c1, c2, 3, padding=1)
        self.bn2   = nn.BatchNorm2d(c2)
        self.pool  = nn.MaxPool2d(kernel_size=(2,1))
        self.drop  = nn.Dropout(dropout)
        self.bigru = nn.GRU(input_size=c2, hidden_size=rnn_hidden, num_layers=rnn_layers,
                            batch_first=True, bidirectional=True)
        self.head  = nn.Linear(2*rnn_hidden, 1)

    def forward(self, x):
        # x: (B,1,M,T)
        x = self.pool(F.relu(self.bn1(self.conv1(x))))   # (B,c1,M/2,T)
        x = self.pool(F.relu(self.bn2(self.conv2(x))))   # (B,c2,M/4,T)
        x = self.drop(x)
        x = x.mean(dim=2)                                # (B,c2,T)
        x = x.permute(0,2,1)                             # (B,T,c2)
        x,_ = self.bigru(x)                              # (B,T,2*h)
        x = self.head(x).squeeze(-1)                     # (B,T) logits
        return x
