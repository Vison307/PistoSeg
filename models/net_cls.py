import torch
from torch import nn
import torch.nn.functional as F

import models.resnet38d


class NetCLS(models.resnet38d.Net):
    def __init__(self):
        super(NetCLS, self).__init__()
        self.dropout7 = torch.nn.Dropout2d(0.5)

        self.fc8 = nn.Conv2d(4096, 4, 1, bias=False)
        
    def forward(self, x):
        N, C, H, W = x.size()  # [B, 3, 256, 256]
        d = super().forward_as_dict(x)
        assert not d['conv4'].isnan().any()
        assert not d['conv5'].isnan().any()
        assert not d['conv6'].isnan().any()

        cam = self.fc8(self.dropout7(d['conv6'])) # d['conv6']: [2, 4096, 32, 32], cam: [2, 4, 32, 32]
        logits = F.adaptive_avg_pool2d(cam, (1, 1))[:, 1:, :, :].squeeze()
        
        return logits
