from __future__ import annotations
from .conv import Conv
from .mam import MAM

import torch.nn as nn
import torch

# Conv 1x1 > Conv 1x1, 3x3, 5x5, 7x7 > Concat > MAM
# YOLOv8n의 C2f 모듈 내 Bottleneck 모듈의 표준 컨볼루션을 EMCM(Efficient Multi-scale Convolution Module)으로 대체
# 여러 그룹으로 분할하여 다양한 크기의 컨볼루션 커널을 병렬로 사용 > 다중 스케일 특징 추출
# 추출된 특징과 원본 특징을 채널 방향으로 연결 > 1x1 컨볼루션으로 차원 조정
# 효과 : 연산량 감소, 모델의 식별 능력 향상
class EMCM(nn.Module):
    def __init__(self, c_in: int, c_out: int, r: float = 0.5):
        super().__init__()

        c_mid = max(8, int(c_in * r))
        # 처음 Conv 1x1
        self.reduce = Conv(c_in, c_mid, 1, 1)

        # Conv 1x1, 3x3, 5x5, 7x7
        # 서로 다른 커널 크기를 사용 > 다양한 수용영역을 동시에
        self.conv1 = Conv(c_mid, c_mid, 1, 1)
        self.conv3 = Conv(c_mid, c_mid, 3, 1)
        self.conv5 = Conv(c_mid, c_mid, 5, 1)
        self.conv7 = Conv(c_mid, c_mid, 7, 1)

        # Concat 하고 나서 MAM으로 중요도 강조
        self.att = MAM(c_mid * 4)
        # 출력 > 1x1 컨볼루션으로 차원 조정
        self.out = Conv(c_mid * 4, c_out, k=1, s=1)

    def forward(self, x):
        # 처음 채널 축소
        x = self.reduce(x)

        y1 = self.conv1(x)
        y2 = self.conv3(x)
        y3 = self.conv5(x)
        y4 = self.conv7(x)

        # 채널 방향으로 concat
        y = torch.cat([y1, y2, y3, y4], dim=1)
        # MAM
        y = self.att(y)
        return self.out(y)
    

class C2fEMCM(nn.Module):
    """
    YOLOv8 C2f의 인터페이스/플로우를 따르는 EMCM 버전.
    Args는 C2f와 호환: (c1, c2, n=1, shortcut=True, g=1, e=0.5)
    - c1: in channels
    - c2: out channels
    - n : 반복 블록 수
    - e : 확장비(내부 채널 c_ = int(c2*e)), C2f 관례 유지
    """
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, **kwargs):
        super().__init__()
        c_ = int(c2 * e)  # 내부 채널(C2f 관례)
        # C2f: cv1은 c1 -> 2*c_ (두 갈래로 쪼개기 위함), cv2는 concat된 채널을 c2로 복원
        self.cv1 = Conv(c1, 2 * c_, k=1, s=1)
        self.cv2 = Conv((2 + n) * c_, c2, k=1, s=1)

        # 반복 블록: EMCM(c_, c_)를 n개.
        # EMCM 내부에서 MAM까지 쓰고 있으니 여기선 별도 MAM 안 둬도 됨.
        self.m = nn.ModuleList(EMCM(c_, c_, r=0.5) for _ in range(n))
        self.shortcut = shortcut  # 필요 시 EMCM 출력에 잔차를 줄 수도 있으나, 기본은 C2f 플로우 유지

    def forward(self, x):
        # cv1 결과를 두 갈래로 split (C2f 전형)
        y = list(self.cv1(x).split(self.cv1.conv.out_channels // 2, dim=1))  # [c_, c_]
        x = y[-1]  # 두 번째 갈래를 순환 입력으로 사용
        for m in self.m:
            z = m(x)
            x = x + z if self.shortcut else z
            y.append(x)  # 진행 중 피처를 쌓음
        # [c_, c_, c_, ..., c_] concat -> (2+n)*c_ 채널
        return self.cv2(torch.cat(y, dim=1))