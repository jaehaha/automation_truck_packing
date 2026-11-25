"""카메라 (손목 카메라 + 전체뷰 카메라) 장치 인터페이스.

같은 메서드 집합을 실물용과 시뮬레이션용으로 둔다. ../../pipeline/details_by_modules/unified.md §3 참고.

  frame()       -> 지금 프레임 (H×W×3 uint8 ndarray, BGR)
  markers()     -> ArUco 마커 목록 [(id, xyz, yaw)]. 로봇 좌표계. 없으면 빈 리스트
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Camera(ABC):
    @abstractmethod
    def frame(self) -> np.ndarray: ...
    @abstractmethod
    def markers(self) -> list[tuple[int, np.ndarray, float]]: ...


class SimCamera(Camera):
    """MuJoCo 모델의 카메라 (wrist, overview 등) 로 렌더링한다. ArUco 아직 미구현."""

    def __init__(self, model, data, cam_name: str = "wrist", width: int = 640, height: int = 480):
        import mujoco
        self.m, self.d, self.cam_name = model, data, cam_name
        self._renderer = mujoco.Renderer(model, width=width, height=height)

    def frame(self):
        import mujoco
        mujoco.mj_forward(self.m, self.d)
        self._renderer.update_scene(self.d, camera=self.cam_name)
        return self._renderer.render()

    def markers(self):
        return []


class RealCamera(Camera):
    """실물: OpenCV 로 USB/CSI 캡처 + ArUco 검출. 아직 미구현."""

    def __init__(self, device: int = 0):
        raise NotImplementedError("실물 카메라는 아직 연결 안 함. SimCamera 를 쓴다.")

    def frame(self):
        raise NotImplementedError

    def markers(self):
        raise NotImplementedError


class DualCamera(Camera):
    """실물과 시뮬 양쪽. frame() 은 실물 기본, markers() 도 실물 기준."""

    def __init__(self, real: Camera, sim: Camera):
        self.real, self.sim = real, sim

    def frame(self):
        return self.real.frame()

    def markers(self):
        return self.real.markers()
