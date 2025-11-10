from __future__ import annotations
from .conv import Conv
from .dw import DW

import torch.nn as nn
import torch

# MAM : Multi-Scale Attention Module
# 중요한 곳에 집중할 수 있도록 하는 모듈
# 2단계로 동작 : multi scale conv  >  spatial attention
class MAM(nn.Module):
    def __init__(self, c: int):
        super().__init__()
        # 입력 이미지를 DWConv 5x5 먼저 해줌
        self.pre = nn.Sequential(DW(c, 5, 5), nn.BatchNorm2d(c), nn.SiLU(True))

        # DWConv 5x5 해준 다음 DWConv 1x7 > 7x1, 1x11 > 11x1, 1x21 > 21x1 해줌
        def branch(k: int):
            return nn.Sequential(
                DW(c, 1, k),
                nn.BatchNorm2d(c),
                nn.SiLU(True),
                DW(c, k, 1),
                nn.BatchNorm2d(c),
                nn.SiLU(True),
            )

        self.b7 = branch(7)
        self.b11 = branch(11)
        self.b21 = branch(21)

        # 분기 합성하고 Conv 1x1
        self.fuse = Conv(c_in=c, c_out=c, k=1, s=1)

        self.spatial = nn.Conv2d(2, 1, 7, 3, bias=False)

    def forward(self, x):
        res = x
        # 제일 첫 작업인 DWConv 5x5
        x = self.pre(x)
        # 3개의 분기를 다 더해줌
        y = self.b7(x) + self.b11(x) + self.b21(x)
        # 합치고 난 후 Conv 1x1
        y = self.fuse(y)

        # 원본이랑 위에 Conv 1x1 까지 한 결과를 곱함
        y = y * res

        # max pooling
        mp = torch.amax(y, dim=1, keepdim=True)

        # avg pooling
        ap = torch.mean(y, dim=1, keepdim=True)

        # Conv2d 7x7 + Sigmoid
        spatial = torch.sigmoid(self.spatial(torch.cat([mp, ap], 1)))

        return y * spatial