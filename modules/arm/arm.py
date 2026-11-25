"""팔 (Piper 6자유도) 장치 인터페이스.

같은 메서드 집합을 실물용과 시뮬레이션용으로 둔다. ../../pipeline/details_by_modules/unified.md §3 참고.

  read()        -> 현재 관절 각도 q (rad, 길이 6)
  command(q)    -> 목표 관절 각도 보내기 (q 는 길이 6, 안전 한계 안)

Robot (manager/robot.py) 이 이 장치를 가져다 쓴다. 보간·안전 검사는 Robot 쪽에서 한다.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Arm(ABC):
    LIMITS = np.array([
        [-2.6179938, 2.6179938],   # joint1
        [0.0, 3.1415926],          # joint2
        [-2.9670597, 0.0],         # joint3
        [-1.7453292, 1.7453292],   # joint4
        [-1.2217304, 1.2217304],   # joint5
        [-2.0943951, 2.0943951],   # joint6
    ])

    @abstractmethod
    def read(self) -> np.ndarray: ...
    @abstractmethod
    def command(self, q: np.ndarray) -> None: ...

    def _check(self, q):
        q = np.asarray(q, float)
        if q.shape != (6,):
            raise ValueError(f"관절 각도는 길이 6 이어야 한다: {q.shape}")
        lo, hi = self.LIMITS[:, 0], self.LIMITS[:, 1]
        if np.any(q < lo - 1e-6) or np.any(q > hi + 1e-6):
            raise ValueError(f"관절 한계 초과: q={q.tolist()}")
        return q


class SimArm(Arm):
    """MuJoCo 모델의 `ctrl[1:7]` 을 통해 팔을 움직인다. ctrl 7개 중 0번은 레일, 1~6번이 조인트."""

    def __init__(self, model, data):
        self.m, self.d = model, data

    def read(self):
        return self.d.qpos[1:7].copy()

    def command(self, q):
        q = self._check(q)
        self.d.ctrl[1:7] = q
        self.d.qpos[1:7] = q


class RealArm(Arm):
    """실물 Piper. piper_sdk C_PiperInterface_V2 로 CAN 을 쓴다.
    아직 미구현 — piper_keyboard.py 가 "한 관절씩 보간해서 움직이는" 참고 예시다."""

    def __init__(self, can: str = "can0"):
        raise NotImplementedError("실물 팔은 아직 연결 안 함. SimArm 을 쓴다.")

    def read(self):
        raise NotImplementedError

    def command(self, q):
        raise NotImplementedError


class DualArm(Arm):
    """실물과 시뮬 양쪽에 같은 명령 (타이밍 A). 상태(read) 는 실물 기준."""

    def __init__(self, real: Arm, sim: Arm):
        self.real, self.sim = real, sim

    def read(self):
        return self.real.read()

    def command(self, q):
        q = self._check(q)
        self.real.command(q)
        self.sim.command(q)
