"""레일 (팔 베이스를 x 방향으로 밀어주는 1축 선형 액추에이터) 장치 인터페이스.

같은 메서드 집합을 실물용과 시뮬레이션용으로 둔다. ../../pipeline/details_by_modules/unified.md §3 참고.

  read()        -> 현재 레일 위치 m (float, 로봇 좌표계)
  command(m)    -> 목표 위치 보내기                     (저수준, RC m)
  home()        -> 원점(0.0) 으로 복귀
  mv(x_cf)      -> 상자 CF x (m) 에 맞춰 레일 이동       (고수준, CF m)
                   팔 베이스가 x_cf - APPROACH_X_OFFSET 에 설 수 있는 위치로.
                   x_cf == 0 은 홈 복귀 (actions 에서 사이클 끝의 rail(0) 용).

범위는 MuJoCo 모델 (modules/arm/model/manibox_modified.xml:165) 의 조인트 range 와 같다: −0.49 ~ +0.51 m.
"""
from __future__ import annotations

import sys
from abc import ABC, abstractmethod
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "transform"))
import transform as _cf                                     # noqa: E402


class Rail(ABC):
    RANGE = (-0.49, 0.51)   # m (로봇 좌표계)
    APPROACH_X_OFFSET = 0.015   # m — 팔 베이스가 상자 CF x 보다 이 만큼 뒤에 서도록

    @abstractmethod
    def read(self) -> float: ...
    @abstractmethod
    def command(self, m: float) -> None: ...
    @abstractmethod
    def home(self) -> None: ...
    @abstractmethod
    def _cf_x_to_rail_m(self, cf_x: float) -> float:
        """CF x (m) → command() 가 받는 RC m 로 변환. 백엔드마다 다른 캘리브레이션."""
        ...

    def mv(self, x_cf: float) -> None:
        """상자 CF x 좌표에 맞춰 레일 이동 — 팔 베이스가 x_cf - APPROACH_X_OFFSET 에 서도록.

        x_cf == 0 은 홈 복귀 (사이클 끝의 rail(0)). 그 외에는 _cf_x_to_rail_m() 으로 변환 후 command().
        """
        x_cf = float(x_cf)
        if x_cf == 0.0:
            self.home()
            return
        arm_base_cf_x = x_cf - self.APPROACH_X_OFFSET
        self.command(self._cf_x_to_rail_m(arm_base_cf_x))

    def _check(self, m: float) -> float:
        m = float(m)
        lo, hi = self.RANGE
        if m < lo - 1e-6 or m > hi + 1e-6:
            raise ValueError(f"레일 범위 초과: {m} (허용 {lo} ~ {hi})")
        return m


class SimRail(Rail):
    """MuJoCo 모델의 `ctrl[0]` (rail actuator) 을 쓴다."""

    def __init__(self, model, data):
        self.m, self.d = model, data

    def read(self):
        return float(self.d.qpos[0])

    def command(self, m):
        m = self._check(m)
        self.d.ctrl[0] = m
        self.d.qpos[0] = m

    def home(self):
        self.command(0.0)

    def _cf_x_to_rail_m(self, cf_x: float) -> float:
        """MW 상의 팔 베이스 x = _ORIGIN_MW[0] + rail_joint.
        CF 는 CF_ORIGIN_IN_MW 로 평행이동만.  →  rail_joint = cf_x - (_ORIGIN_MW[0] - CF_ORIGIN_IN_MW[0])."""
        return float(cf_x) - (_cf._ORIGIN_MW[0] - _cf.CF_ORIGIN_IN_MW[0])


class RealRail(Rail):
    """실물: modules/rail_suction.ino 펌웨어가 올라간 공유 아두이노를 쓴다.
    아두이노 좌표는 0 ~ 1000 mm, Rail API 좌표는 Rail.RANGE (−0.49 ~ +0.51 m, RC 기준).

    두 좌표계의 매핑은 modules/transform/transform.py 의 RAIL_ZERO_CF_X 를 거쳐 통일한다. 다만 지금 Sequencer
    는 아직 MuJoCo/RC 좌표로 돌아가므로 과도기에 호환 매개변수 `zero_offset_m` 로 평행이동만 한다.

        RC m = (Arduino mm) / 1000 + zero_offset_m
        Arduino mm = (RC m - zero_offset_m) * 1000

    컨테이너 좌표계로 완전 통일되면 이 변환은 frame.rail_mm_to_cf_x / cf_x_to_rail_mm 로 대체한다.

    타이밍 A: MOVE 도중 새 MOVE 는 아두이노가 "busy" 로 거부 → 조용히 무시 → 다음 호출로 따라감.
    """

    def __init__(self, arduino, zero_offset_m: float = -0.49, auto_init: bool = True):
        self.arduino = arduino
        self.zero_offset_m = zero_offset_m
        if auto_init:
            self._initialize()

    # ---------- 유틸 ----------
    def _initialize(self):
        """아두이노의 현재 위치를 0 mm 로 설정. 레일을 물리적 기준점에 두고 돌린 뒤 호출."""
        self.arduino.send_await("INITIALIZE", "POSITION:")
        self.arduino.drain()

    def _to_arduino_mm(self, m: float) -> float:
        return (m - self.zero_offset_m) * 1000.0

    def _to_our_m(self, mm: float) -> float:
        return mm / 1000.0 + self.zero_offset_m

    # ---------- API ----------
    def read(self) -> float:
        # "POSITION: X.XXX mm (estimated)" 를 기다림
        lines = self.arduino.send_await("POS", "POSITION:")
        for line in lines:
            if line.startswith("POSITION:"):
                mm = float(line.split(":")[1].strip().split()[0])
                self.arduino.drain()
                return self._to_our_m(mm)
        raise RuntimeError(f"POS 응답 파싱 실패: {lines}")

    def command(self, m: float) -> None:
        m = self._check(m)
        arduino_mm = self._to_arduino_mm(m)
        try:
            # 즉시 응답: "MOVING TO: X.XXX mm"
            self.arduino.send(f"MOVE {arduino_mm:.3f}")
        except RuntimeError as e:
            if "busy" in str(e).lower():
                return      # 타이밍 A: 다음 호출 때 재시도되며 최신 목표로 따라감
            raise

    def home(self) -> None:
        # HOME 은 결국 MOVE 0 과 같음. busy 처리 동일.
        try:
            self.arduino.send("HOME")
        except RuntimeError as e:
            if "busy" in str(e).lower():
                return
            raise

    def _cf_x_to_rail_m(self, cf_x: float) -> float:
        """CF x → Arduino mm (transform.py) → RC m (zero_offset_m 통해 평행이동)."""
        arduino_mm = _cf.cf_x_to_rail_mm(float(cf_x))
        return arduino_mm / 1000.0 + self.zero_offset_m


class DualRail(Rail):
    """실물과 시뮬 양쪽에 같은 명령 (타이밍 A). 상태(read) 는 실물 기준."""

    def __init__(self, real: Rail, sim: Rail):
        self.real, self.sim = real, sim

    def read(self):
        return self.real.read()

    def command(self, m):
        m = self._check(m)
        self.real.command(m)          # 실물 먼저 — 하드웨어 예외가 터지면 시뮬은 반영 안 함
        self.sim.command(m)

    def home(self):
        self.real.home()
        self.sim.home()

    def _cf_x_to_rail_m(self, cf_x: float) -> float:
        # Dual 에서는 실물 기준 (sim 쪽은 자기 변환으로 따라감)
        return self.real._cf_x_to_rail_m(cf_x)

    def mv(self, x_cf: float) -> None:
        # 양쪽에 각자 변환을 쓰게 하려고 override — 기본 구현 (base.mv) 은 한쪽 변환만 씀.
        self.real.mv(x_cf)
        self.sim.mv(x_cf)
