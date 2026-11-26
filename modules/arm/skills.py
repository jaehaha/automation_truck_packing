"""skills — 팔(arm) 쪽 저수준 원자 동작. Packing/UnPacking 사이클의 arm-motion 단계.

manager/actions.py 가 이들을 순서대로 호출해 전체 사이클을 구성한다.
구체 구현은 전부 이 파일에만 있고, manager 쪽은 호출만 한다.

이 파일에는 **arm 움직임만** 담는다:
  * demo 기반     — 사용자 녹화 궤적/자세 재생 (mv_pos, approach_to_buffer)
  * 모드별 분기   — pack / unpack 인자로 방향·값 선택 (alignment, approach_to_box)
  * 상자 놓기     — put_to_buffer

흡착 ON/OFF 는 `modules/suction/suction.py` 의 Suction.grip()/un_grip() 로,
레일 이동은 `modules/rail/rail.py` 의 Rail.mv() 로 각각 분리.

좌표는 모두 Container Frame (CF). modules/transform/transform.py 가 변환.
저장된 자세·궤적은 manager/demos/*.npz (사용자 녹화). 아직 없으면 NotImplementedError.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent              # modules/arm/
MODULES_DIR = HERE.parent                           # modules/
PROJECT_ROOT = MODULES_DIR.parent                   # 프로젝트 루트
DEMOS = PROJECT_ROOT / "manager" / "demos"

sys.path.insert(0, str(MODULES_DIR / "transform"))
import transform as cf                              # noqa: E402

# ===== 사용자 보정값 (실측·녹화로 채워 넣기) — 전부 CF 기준 m =====

# alignment 가 마커 중심으로부터 "맞춰야 하는" offset (호버 위치)
ALIGN_OFFSET_PACK_Z: float = 0.020     # 집기 전 상자 윗면 위 20 mm 에 호버
ALIGN_OFFSET_UNPACK_X: float = 0.020   # 꺼내기 전 상자 옆면 앞 20 mm 에 호버

# alignment 가 호버 뒤 "접촉까지 이동" 하는 양 (흡착 ON 직전 자세)
GRIP_APPROACH_PACK_Z: float = 0.020    # z 로 내려가는 양
GRIP_APPROACH_UNPACK_X: float = 0.020  # x 로 밀어 넣는 양

# approach_to_box 가 자리 위에서 x 로 "뒤로 빠져 있는" 거리
APPROACH_X_OFFSET: float = 0.015       # 15 mm 뒤

# buffer(투입 공간) 에 놓을 때 z 내리는 양
PUT_BUFFER_Z_OFFSET: float = 0.005     # 5 mm
# =================================================================


# ---------- demo 기반 (사용자 녹화 재생) ----------
def _load_demo(name: str):
    """manager/demos/<name>.npz 에서 저장된 자세 또는 궤적 로드."""
    path = DEMOS / f"{name}.npz"
    if not path.exists():
        raise NotImplementedError(
            f"{path} 가 없음. 데모 녹화 후 저장 필요.\n"
            f"저장 형식: np.savez(path, waypoints=np.ndarray[N,7])  # [rail, j1..j6]")
    z = np.load(path)
    return z["waypoints"]


def mv_pos(robot, speed: float = 0.3):
    """레일 이동 자세로 이동. manager/demos/mv_pos.npz 에 녹화된 자세 사용."""
    wps = _load_demo("mv_pos")
    robot.follow(wps, speed=speed)


def approach_to_buffer(robot, speed: float = 0.3):
    """투입 공간 위로 이동. manager/demos/approach_to_buffer.npz 에 녹화된 궤적 사용."""
    wps = _load_demo("approach_to_buffer")
    robot.follow(wps, speed=speed)


# ---------- 모드별 (pack / unpack) ----------
def alignment(robot, mode: str = "pack"):
    """wrist 카메라 ArUco 를 보고 흡착 ON 직전 자세까지 IK 미세 조정.

    1) ALIGN_OFFSET 위치로 호버 (상자 면에서 조금 떨어진 안전 지점)
    2) GRIP_APPROACH 거리만큼 접근해 상자 면에 접촉 (이 뒤에 suction.grip() 호출)

    pack  — 상자 윗면 마커를 흡착컵 아래 (ALIGN_OFFSET_PACK_Z 호버 → GRIP_APPROACH_PACK_Z 하강) 으로
    unpack — 상자 입구 면 마커를 흡착컵 앞 (ALIGN_OFFSET_UNPACK_X 호버 → GRIP_APPROACH_UNPACK_X 전진) 으로
    """
    markers = robot.camera.markers() if robot.camera else []
    if not markers:
        raise RuntimeError(f"alignment({mode}): ArUco 마커 못 봄")
    _mid, xyz_cf, yaw = markers[0]
    if mode == "pack":
        hover_cf   = xyz_cf + np.array([0.0, 0.0, ALIGN_OFFSET_PACK_Z])
        contact_cf = hover_cf - np.array([0.0, 0.0, GRIP_APPROACH_PACK_Z])
    elif mode == "unpack":
        hover_cf   = xyz_cf + np.array([-ALIGN_OFFSET_UNPACK_X, 0.0, 0.0])
        contact_cf = hover_cf + np.array([ GRIP_APPROACH_UNPACK_X, 0.0, 0.0])
    else:
        raise ValueError(f"mode 는 'pack' 또는 'unpack': {mode}")
    raise NotImplementedError(
        f"alignment({mode}) 미구현: hover={hover_cf} → contact={contact_cf} 에 대한 IK + follow 필요. "
        f"RealArm + modules/arm/stacking.py Stacker.ik() 연결")


def approach_to_box(robot, box, mode: str = "pack"):
    """자리의 yz 로 팔 자세를 맞춘 뒤 x 로 미세 이동.

    pack  — box.goal_cf 의 (y, z) 에 맞춘 뒤 +x 로 밀어 놓기
    unpack — box.current 의 (y, z) 에 맞춘 뒤 +x 로 밀어 집기 전 위치
    """
    target_cf = box.goal_cf if mode == "pack" else box.current
    raise NotImplementedError(
        f"approach_to_box({mode}) 미구현: target_cf={target_cf} 로 yz→x 순차 이동. "
        f"Stacker.ik() 또는 Sequencer 틀 재사용")


def put_to_buffer(robot, box):
    """BoxTable 의 상자 size 참고해 투입 공간에 z offset 맞춰 놓기."""
    _h = box.size[2]
    raise NotImplementedError(
        "put_to_buffer 미구현: 현재 자세에서 z 로 (h/2 + PUT_BUFFER_Z_OFFSET) 내리기")
