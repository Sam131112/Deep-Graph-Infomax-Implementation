import torch

import torch.nn as nn

class Readout(nn.Module):
    def __init__(self):
        super(Readout,self).__init__()
        self.sigm = torch.nn.Sigmoid()

    def forward(self,feats):
        summary = torch.mean(feats,0)
        return self.sigm(summary)
