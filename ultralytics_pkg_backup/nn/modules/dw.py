from __future__ import annotations

import torch.nn as nn

# MAM에 들어가는 DWConv
# 모델을 경량화 하는데 도움이 된다고 함 (연산량이나 파라미터 수 절감)
class DW(nn.Module):
    # DWConv : Depthwise Conv, 채널마다 따로 필터를 학습시킴
    # 각 단일 채널에 대해서만 수행되는 필터를 사용
    # 출력 채널 수 = 입력 채널 수
    def __init__(self, c, k_h: int, k_w: int, s=1, dilation=1, bias=False):
        super().__init__()
        # dilation = 팽창
        pad_h = dilation * (k_h // 2)
        pad_w = dilation * (k_w // 2)
        # 입력 채널 수와 출력 채널 수가 동일하므로 둘 다 c
        # groups = c라는건 입력 채널을 c개의 그룹으로 쪼개고 각 채널(여기서는 1 채널)에 대해 서로 다른 필터를 적용한다는 의미
        self.conv = nn.Conv2d(
            c, c, (k_h, k_w), s, (pad_h, pad_w), dilation=dilation, groups=c, bias=bias
        )

    def forward(self, x):
        return self.conv(x)