"""Robot: 팔·레일·흡착·카메라 네 장치를 묶어 하나의 API 로 노출한다.

../pipeline/details_by_modules/unified.md §2, §4 참고.

  state()                        -> 지금 상태 dict
  follow(waypoints, speed=0.3)   -> [[레일, j1..j6], ...] 순서대로 따라간다
  suction_on/off()               -> 흡착 켜고 상자가 붙을 때까지 기다림 / 끔
  measure_box()                  -> 손목 카메라로 상자 측정 (dx, dy, yaw)
  stop()                         -> 그 자리에서 정지. 흡착 상태는 유지.

세 가지 모드:
  Robot.sim(m, d)     — 시뮬만. 지금까지와 동일
  Robot.real(...)     — 실물만 (뷰어 없이 돌릴 때)
  Robot.dual(m, d, ..)— 실물 + 시뮬 (시뮬이 뷰어 역할). 타이밍 A: 같은 명령을 양쪽에 동시 전송
"""
from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "modules" / "arm"))
sys.path.insert(0, str(ROOT / "modules" / "rail"))
sys.path.insert(0, str(ROOT / "modules" / "vacuum"))
sys.path.insert(0, str(ROOT / "modules" / "camera"))

from arm import Arm, SimArm, RealArm, DualArm                     # noqa: E402
from rail import Rail, SimRail, RealRail, DualRail                 # noqa: E402
from suction import Suction, SimSuction, RealSuction, DualSuction  # noqa: E402
from camera import Camera, SimCamera, RealCamera, DualCamera       # noqa: E402

import arduino as _arduino                                          # noqa: E402


@dataclass
class RobotState:
    rail: float
    q: np.ndarray                                 # 길이 6
    suction: bool
    holding: bool
    fault: str | None = None


class Robot:
    """네 장치의 집합. 시뮬/실물/듀얼 모두 같은 메서드로 쓴다."""

    def __init__(self, arm: Arm, rail: Rail, suction: Suction, camera: Camera | None = None,
                 on_step: Callable | None = None):
        self.arm, self.rail, self.suction, self.camera = arm, rail, suction, camera
        self.on_step = on_step                    # 매 보간 step 마다 호출 (뷰어 갱신 등)
        self._stopped = False

    # ------- 팩토리 -------
    @classmethod
    def sim(cls, model, data, cam_name: str = "wrist", on_step: Callable | None = None, use_camera: bool = False):
        cam = SimCamera(model, data, cam_name) if use_camera else None
        return cls(SimArm(model, data), SimRail(model, data), SimSuction(), cam, on_step=on_step)

    @classmethod
    def real(cls, arduino_port: str = "/dev/tty.usbmodem0001", can: str = "can0",
             cam_device: int = 0, use_camera: bool = False, on_step: Callable | None = None):
        """실물만. 뷰어 없이 하드웨어만 구동."""
        ard = _arduino.init_shared(arduino_port)
        arm = RealArm(can=can)
        rail = RealRail(ard)
        suction = RealSuction(ard)
        camera = RealCamera(device=cam_device) if use_camera else None
        return cls(arm, rail, suction, camera, on_step=on_step)

    @classmethod
    def dual(cls, model, data, arduino_port: str = "/dev/tty.usbmodem0001", can: str = "can0",
             cam_device: int = 0, cam_name: str = "wrist", use_camera: bool = False,
             on_step: Callable | None = None):
        """실물 + 시뮬. 같은 명령을 양쪽에 보낸다 (타이밍 A). 시뮬이 뷰어 역할.
        read()·holding()·measure_box() 는 실물에서 가져온다."""
        ard = _arduino.init_shared(arduino_port)
        arm     = DualArm    (RealArm(can=can),         SimArm(model, data))
        rail    = DualRail   (RealRail(ard),            SimRail(model, data))
        suction = DualSuction(RealSuction(ard),         SimSuction())
        camera  = DualCamera (RealCamera(device=cam_device), SimCamera(model, data, cam_name)) if use_camera else None
        return cls(arm, rail, suction, camera, on_step=on_step)

    # ------- API -------
    def state(self) -> RobotState:
        on = self.suction._on if hasattr(self.suction, "_on") else self.suction.holding()
        return RobotState(rail=self.rail.read(), q=self.arm.read(),
                          suction=bool(on), holding=self.suction.holding())

    def stop(self):
        """그 자리에서 정지. 흡착은 그대로 둔다 (들고 있던 상자가 떨어지지 않게)."""
        self._stopped = True

    def suction_on(self):
        self.suction.on()

    def suction_off(self):
        self.suction.off()

    # ------- 경로 추종 -------
    def follow(self, waypoints: list[np.ndarray], speed: float = 0.3,
               box_follow: Callable | None = None):
        """경유점 사이를 선형으로 보간해 레일·팔에 동시에 명령.
        box_follow(q) 는 들고 있는 상자를 공구에 맞춰 옮기는 콜백 (시뮬 전용)."""
        self._stopped = False
        if len(waypoints) < 2:
            return
        for a, b in zip(waypoints[:-1], waypoints[1:]):
            a = np.asarray(a, float); b = np.asarray(b, float)
            diff = b - a
            n = int(max(2, np.ceil(abs(diff[0]) / 0.01), np.ceil(np.abs(diff[1:]).max() / 0.03))) + 1
            for t in np.linspace(0, 1, n)[1:]:
                if self._stopped:
                    return
                q = a + diff * t
                self.rail.command(float(q[0]))
                self.arm.command(q[1:])
                if box_follow is not None:
                    box_follow(q)
                if self.on_step is not None:
                    self.on_step()
                time.sleep(0.02 / max(speed, 1e-3))

    def measure_box(self) -> tuple[float, float, float]:
        """손목 카메라로 상자 측정: (dx, dy, yaw). 지금은 더미값. TODO: ArUco."""
        return (0.0, 0.0, 0.0)
