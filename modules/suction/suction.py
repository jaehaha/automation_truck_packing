"""흡착 공구 (벨로즈 흡착컵 + 솔레노이드 밸브) 장치 인터페이스.

같은 메서드 집합을 실물용과 시뮬레이션용으로 둔다. ../../pipeline/details_by_modules/unified.md §3 참고.

  on()               -> 밸브 켬  (하드웨어 저수준)
  off()              -> 밸브 끔  (하드웨어 저수준)
  holding()          -> 지금 상자가 붙어 있는지 (압력 센서가 없으면 명령 상태 반환)
  grip(mode)         -> 상자 집기. 현재는 on() 과 같지만, mode 로 추후 pack/unpack 분기 가능
  un_grip()          -> 상자 놓기. off() 와 동일

modules/rail_suction.ino 펌웨어는 "ON" / "OFF" 명령을 직접 받는다 (토글 아님).
Python 쪽에서도 그대로 보내면 됨.

호출 흐름: manager/actions.py → robot.suction.grip(mode)/un_grip().
접근 motion (호버 → 접촉) 은 modules/arm/skills.py 의 alignment() 가 담당한다.
"""
from __future__ import annotations

from abc import ABC, abstractmethod


class Suction(ABC):
    @abstractmethod
    def on(self) -> None: ...
    @abstractmethod
    def off(self) -> None: ...
    @abstractmethod
    def holding(self) -> bool: ...

    # ---------- 고수준 API (actions.py 가 호출) ----------
    def grip(self, mode: str = "pack") -> None:
        """상자 집기 — 흡착 ON. mode 는 pack/unpack 분기용 (현재는 동일).

        호출 시점: skills.alignment(mode) 로 상자 면 접촉까지 접근한 뒤.
        """
        if mode not in ("pack", "unpack"):
            raise ValueError(f"mode 는 'pack' 또는 'unpack': {mode}")
        self.on()

    def un_grip(self) -> None:
        """상자 놓기 — 흡착 OFF."""
        self.off()


class SimSuction(Suction):
    """시뮬레이션: 상태만 추적한다. 상자 시각화는 Robot/skills 쪽에서 한다."""

    def __init__(self):
        self._on = False
        self._holding_box = None

    def on(self):
        self._on = True

    def off(self):
        self._on = False
        self._holding_box = None

    def holding(self):
        return self._on and self._holding_box is not None

    def _attach(self, box_id):
        if self._on:
            self._holding_box = box_id

    def _detach(self):
        self._holding_box = None


class RealSuction(Suction):
    """실물: 공유 아두이노에 "ON" / "OFF" 전송. 압력 센서 없으므로 holding() 은 명령 상태 그대로."""

    def __init__(self, arduino):
        self.arduino = arduino
        self._state = False     # 아두이노 setup() 에서 setValve(false) 라 OFF 로 시작

    def on(self):
        self.arduino.send("ON")        # 응답: "SUCTION ON"
        self._state = True

    def off(self):
        self.arduino.send("OFF")       # 응답: "SUCTION OFF"
        self._state = False

    def holding(self):
        return self._state


class DualSuction(Suction):
    """실물과 시뮬 양쪽에 같은 명령. holding() 은 실물 기준."""

    def __init__(self, real: Suction, sim: Suction):
        self.real, self.sim = real, sim

    def on(self):
        self.real.on()
        self.sim.on()

    def off(self):
        self.real.off()
        self.sim.off()

    def holding(self):
        return self.real.holding()

    # 시뮬 거울용 (actions.pick_and_place 가 호출)
    def _attach(self, box_id):
        if hasattr(self.sim, "_attach"):
            self.sim._attach(box_id)

    def _detach(self):
        if hasattr(self.sim, "_detach"):
            self.sim._detach()

    @property
    def _on(self):
        # Robot.state() 가 참조. Dual 에서는 실물 기준.
        return self.real.holding()
