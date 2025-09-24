import torch
from src.sed.models.crnn import CRNNSED, frame_bce

def test_forward_shapes():
    B,T,Fm,C = 3, 200, 64, 3
    x = torch.randn(B, T, Fm)
    L = torch.tensor([200,150,80])
    y = torch.randint(0,2,(B,T,C)).float()
    m = CRNNSED(n_mels=Fm, n_classes=C)
    f, c = m(x, lengths=L)
    assert f.shape == (B,T,C)
    assert c.shape == (B,C)
    loss = frame_bce(f, y, lengths=L)
    assert loss.item() > 0
