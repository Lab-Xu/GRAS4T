import torch
import torch.nn as nn

class AvgReadout2(nn.Module):
    def __init__(self):
        super(AvgReadout2, self).__init__()

    def forward(self, seq, msk):
        if msk is None:
            return torch.mean(seq, 1)
        else:
            # msk = torch.unsqueeze(msk, -1) # torch.Size([N, N, 1])
            # ch = torch.sum(seq * msk, 1)

            # print("ch shape:", ch.shape)
            return torch.matmul(msk, torch.squeeze(seq))