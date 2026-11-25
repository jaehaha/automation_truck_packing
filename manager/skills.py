"""조합 동작 (skills) — Robot 의 네 장치를 묶어 Packing/UnPacking 사이클을 구성.

pipeline/pipeline.md 의 Packing·UnPacking 시퀀스가 여기서 함수 하나씩 대응된다.

함수 종류:
  * demo 기반     — 사용자가 궤적/자세를 저장해 두고 그걸 재생 (mv_pos, approach_to_buffer)
  * 모드별 분기   — pack / unpack 인자에 따라 다른 값·방향 사용 (alignment, grip, approach_to_box)
  * 단순 래퍼     — Robot 메서드 한두 줄 호출 (un_grip, rail)

모든 좌표는 Container Frame (CF) 기준. modules/transform/transform.py 가 변환.
저장된 자세·궤적은 manager/demos/*.npz (사용자가 녹화). 아직 없으면 NotImplementedError.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DEMOS = HERE / "demos"

sys.path.insert(0, str(ROOT / "modules" / "transform"))
import transform as cf                                   # noqa: E402

# ===== 사용자 보정값 (실측·녹화로 채워 넣기) =====
# 각 상수는 CF 기준 m.

# alignment 가 마커 중심으로부터 "맞춰야 하는" offset.
# pack: 집기 직전 상자 윗면 중심에서 z 로 얼마나 띄워 둘지
# unpack: 놓인 상자의 입구-쪽 면에서 x 로 얼마나 물러나 둘지
ALIGN_OFFSET_PACK_Z: float = 0.020     # 20 mm 위
ALIGN_OFFSET_UNPACK_X: float = 0.020   # 20 mm 앞

# grip 시 접근 거리 (ON 신호 이후 상자 면에 닿을 때까지 이동량).
GRIP_APPROACH_PACK_Z: float = 0.020    # z 로 내려가는 양
GRIP_APPROACH_UNPACK_X: float = 0.020  # x 로 밀어 넣는 양

# approach_to_box 가 자리 위에서 x 로 "뒤로 빠져 있는" 거리.
APPROACH_X_OFFSET: float = 0.015       # 15 mm 뒤

# buffer(투입 공간) 에 놓을 때 z 내리는 양.
PUT_BUFFER_Z_OFFSET: float = 0.005     # 5 mm
# ==================================================


# ---------- demo 기반 (사용자가 녹화한 궤적/자세 재생) ----------
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
    """wrist 카메라로 ArUco 마커를 보고 ALIGN 위치로 IK 미세 조정.

    pack  — 상자 윗면 마커가 흡착컵 아래 ALIGN_OFFSET_PACK_Z 지점에 오도록
    unpack — 상자 입구 면 마커가 흡착컵 앞 ALIGN_OFFSET_UNPACK_X 지점에 오도록
    """
    markers = robot.camera.markers() if robot.camera else []
    if not markers:
        raise RuntimeError(f"alignment({mode}): ArUco 마커 못 봄")
    _mid, xyz_cf, yaw = markers[0]
    if mode == "pack":
        # TODO: 마커 CF 위치 + Z 로 ALIGN_OFFSET_PACK_Z 떨어진 지점으로 IK
        target_cf = xyz_cf + np.array([0.0, 0.0, ALIGN_OFFSET_PACK_Z])
    elif mode == "unpack":
        # TODO: 마커 CF 위치 + X 로 ALIGN_OFFSET_UNPACK_X 떨어진 지점으로 IK
        target_cf = xyz_cf + np.array([-ALIGN_OFFSET_UNPACK_X, 0.0, 0.0])
    else:
        raise ValueError(f"mode 는 'pack' 또는 'unpack': {mode}")
    raise NotImplementedError(
        f"alignment({mode}) 미구현: target_cf={target_cf} 에 대한 IK 호출 필요. "
        f"RealArm + modules/arm/stacking.py Stacker.ik() 연결")


def grip(robot, mode: str = "pack"):
    """정해진 offset 만큼 이동 후 흡착 ON.

    pack  — z 로 GRIP_APPROACH_PACK_Z 내려가면서 상자 윗면 접촉
    unpack — x 로 GRIP_APPROACH_UNPACK_X 밀어 넣어 상자 옆면 접촉
    """
    # TODO: 현재 TCP 위치에서 상대 이동. RealArm 의 EEF 조회 + IK 역행 가능하면 구현.
    raise NotImplementedError(
        f"grip({mode}) 미구현: {'z' if mode=='pack' else 'x'} 방향 상대 이동 후 suction_on")
    # 완성 시:
    # robot.follow(moves)
    # robot.suction_on()


def un_grip(robot):
    """흡착 OFF — 상자 분리."""
    robot.suction_off()


def rail(robot, x_cf: float, speed: float = 0.3):
    """상자 CF x 좌표에 맞는 레일 위치로 이동.

    pack 적재 자리 또는 unpack 꺼낼 자리의 x 를 CF 로 받아
    modules/transform/transform.py 로 Arduino mm 로 변환.
    팔 베이스가 그 x 에서 적절한 거리로 떨어지도록 APPROACH_X_OFFSET 을 뺀 값을 쓴다.
    """
    rail_target_cf = x_cf - APPROACH_X_OFFSET
    rail_mm = cf.cf_x_to_rail_mm(rail_target_cf)
    # robot.rail 은 아직 m 단위라 쓸 때 조정 필요. RealRail(zero_offset_m) 통과 가정.
    rail_m = rail_mm / 1000.0 + robot.rail.zero_offset_m if hasattr(robot.rail, "zero_offset_m") else cf.rail_mm_to_cf_x(rail_mm) - cf.RAIL_ZERO_CF_X
    robot.rail.command(float(rail_m))


def approach_to_box(robot, box, mode: str = "pack"):
    """자리의 yz 로 팔 자세를 맞춘 뒤 x 로 미세 이동.

    pack  — box.goal_cf 의 (y, z) 에 맞춘 뒤 +x 로 밀어 놓기
    unpack — box.current_cf 의 (y, z) 에 맞춘 뒤 +x 로 밀어 집기 전 위치
    """
    target_cf = box.goal_cf if mode == "pack" else box.current
    raise NotImplementedError(
        f"approach_to_box({mode}) 미구현: target_cf={target_cf} 로 yz→x 순차 이동. "
        f"Stacker.ik() 또는 Sequencer 틀 재사용")


def put_to_buffer(robot, box):
    """BoxTable 의 상자 size 를 참고해 투입 공간에 z offset 맞춰 놓기."""
    _h = box.size[2]
    raise NotImplementedError(
        "put_to_buffer 미구현: 현재 자세에서 z 로 (h/2 + PUT_BUFFER_Z_OFFSET) 내리기")


# ---------- 전체 사이클 ----------
def pack_one(robot, box, table, speed: float = 0.3):
    """Packing LOOP 한 번 (사람이 상자를 투입 위치에 둔 뒤 호출)."""
    approach_to_buffer(robot, speed)
    alignment(robot, "pack")
    grip(robot, "pack")
    mv_pos(robot, speed)
    rail(robot, float(box.goal_cf[0]), speed)
    approach_to_box(robot, box, "pack")
    un_grip(robot)
    mv_pos(robot, speed)
    robot.rail.command(0.0)                             # rail(0)
    table.done(box.id)


def unpack_one(robot, box, table, speed: float = 0.3):
    """UnPacking LOOP 한 번."""
    rail(robot, float(box.current[0]), speed)
    approach_to_box(robot, box, "unpack")
    alignment(robot, "unpack")
    grip(robot, "unpack")
    mv_pos(robot, speed)
    robot.rail.command(0.0)                             # rail(0)
    approach_to_buffer(robot, speed)
    put_to_buffer(robot, box)
    un_grip(robot)
    mv_pos(robot, speed)
    table.remove(box.id)                                # TODO: BoxTable.remove 필요


# ---------- 과도기: Sequencer 기반 pick_and_place (현재 돌아가는 버전) ----------
def pick_and_place(robot, plan, s=None):
    """현재 돌아가는 4단계 플래너 기반 실행 (Sequencer.plan 이 낸 Plan).
    새 skills 함수들 (pack_one/unpack_one) 이 완성되면 이것은 대체된다."""
    i = plan.index
    robot.follow(plan.approach)
    robot.suction_on()
    if s is not None:
        robot.suction._attach(i)

    def carry_box(q):
        if s is not None:
            s.d.qpos[:] = q
            s.carry_box(i, plan.held_geom)
    robot.follow(plan.carry, box_follow=carry_box if s is not None else None)

    if s is not None:
        s.set_box(i, plan.center, plan.quat)
        robot.suction._detach()
    robot.suction_off()
    robot.follow(plan.retreat)
