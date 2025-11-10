from __future__ import annotations
from .conv import Conv
from .emcm import EMCM

import torch.nn as nn
import torch 

# YOLOv8n의 C2f에서 Bottleneck 부분을 EMCM으로 교체
class C2f_EMCM(nn.Module):
    def __init__(self, c_in: int, c_out: int, n: int = 2, r: float = 0.5):
        super().__init__()
        # Conv 1x1
        self.conv1 = Conv(c_in, c_out, 1, 1)

        # EMCM을 여러번 반복 (그림에는 중간에 생략되어 있어서 일단 2회 진행)
        # 각 EMCM > 1x1 > 1x1, 3x3, 5x5, 7x7 > concat > MAM > 1x1
        self.blocks = nn.ModuleList([EMCM(c_out, c_out, r=r) for _ in range(n)])

        # Conv 1x1
        self.conv2 = Conv(c_in, c_out, 1, 1)

    def forward(self, x):
        # Conv 1x1
        x = self.conv1(x)

        # C2f 핵심 : 일부는 우회 (바로 concat), 일부는 EMCM 거치기
        parts = [x]  # 초기 특징 저장 (parts는 concat 할 것들을 모아두는 곳)
        y = x

        for b in self.blocks:
            # 현재 특징만 EMCM
            y = b(y)
            # 반환 결과를 concat 후보에 추가
            parts.append(y)

        # 채널 방향으로 연결
        z = torch.cat(parts, dim=1)
        # Conv 1x1 하고 다음으로 전달
        return self.conv2(z)