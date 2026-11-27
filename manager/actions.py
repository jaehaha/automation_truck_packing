"""actions — Packing / UnPacking 전체 사이클 (manager 레이어).

이 파일은 세 종류의 하위 호출을 **순서대로 묶기만** 한다:
  * 팔 동작   — modules/arm/skills.py  (approach_to_buffer, alignment, mv_pos, approach_to_box, put_to_buffer)
  * 흡착     — Suction.grip() / un_grip()                modules/suction/suction.py
  * 레일     — Rail.mv(x_cf)                              modules/rail/rail.py

전체 호출 흐름 도식: pipeline/file_call_flow.md
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "modules" / "arm"))

import skills                                      # noqa: E402  저수준 arm 원자 동작


def packing(robot, box, table, speed: float = 0.3):
    """Packing 사이클 — 사람이 상자를 투입 위치에 둔 뒤 호출.

    1 approach_to_buffer   (arm)
    2 alignment(pack)      (arm: 호버 → 접촉)
    3 grip()               (suction)
    4 mv_pos               (arm)
    5 rail.mv(x)           (rail: box.goal_cf x 로 이동)
    6 approach_to_box(pack)(arm)
    7 un_grip              (suction)
    8 mv_pos               (arm)
    9 rail.mv(0)           (rail: 홈 복귀)
    10 table.done(box.id)  (상태 갱신)
    """
    skills.approach_to_buffer(robot, speed)
    skills.alignment(robot, "pack")
    robot.suction.grip()
    skills.mv_pos(robot, speed)
    robot.rail.mv(float(box.goal_cf[0]))
    skills.approach_to_box(robot, box, "pack")
    robot.suction.un_grip()
    skills.mv_pos(robot, speed)
    robot.rail.mv(0.0)
    table.done(box.id)


def unpacking(robot, box, table, speed: float = 0.3):
    """UnPacking 사이클 — BoxTable 의 current 위치에 있는 상자를 버퍼로 꺼낸다.

    1 rail.mv(x)            (rail: box.current x 로 이동)
    2 approach_to_box(unpack)(arm)
    3 alignment(unpack)     (arm)
    4 grip()                (suction)
    5 mv_pos                (arm)
    6 rail.mv(0)            (rail: 홈 복귀)
    7 approach_to_buffer    (arm)
    8 put_to_buffer         (arm)
    9 un_grip               (suction)
    10 mv_pos               (arm)
    11 table.remove(box.id) (상태 갱신)
    """
    robot.rail.mv(float(box.current[0]))
    skills.approach_to_box(robot, box, "unpack")
    skills.alignment(robot, "unpack")
    robot.suction.grip()
    skills.mv_pos(robot, speed)
    robot.rail.mv(0.0)
    skills.approach_to_buffer(robot, speed)
    skills.put_to_buffer(robot, box)
    robot.suction.un_grip()
    skills.mv_pos(robot, speed)
    table.remove(box.id)


# ---------- 과도기: Sequencer 기반 (현재 돌아가는 경로) ----------
def pick_and_place(robot, plan, s=None, speed: float = 1.0):
    """현재 돌아가는 4단계 플래너 기반 실행 (Sequencer.plan 이 낸 Plan).

    speed: 모터 속도 배율 (sim: 보간 step 사이 sleep 를 1/speed 배로 — 빠르게 재생;
                        real: 향후 RealArm 가 Piper MotionCtrl_2 speed_percent 로 변환).
    새 packing/unpacking 이 완성되면 이것은 제거된다.
    """
    i = plan.index
    robot.follow(plan.approach, speed=speed)
    robot.suction_on()
    if s is not None:
        robot.suction._attach(i)

    def carry_box(q):
        if s is not None:
            s.d.qpos[:] = q
            s.carry_box(i, plan.held_geom)
    robot.follow(plan.carry, speed=speed, box_follow=carry_box if s is not None else None)

    if s is not None:
        s.set_box(i, plan.center, plan.quat)
        robot.suction._detach()
    robot.suction_off()
    robot.follow(plan.retreat, speed=speed)
