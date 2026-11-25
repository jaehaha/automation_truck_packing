"""조합 동작 (skills). Robot 의 네 장치를 묶어서 "상자 하나 집고 놓기" 같은 일을 한다.

../pipeline/details_by_modules/unified.md §2 참고.
"""
from __future__ import annotations


def pick_and_place(robot, plan, s=None):
    """approach → suction on → carry → suction off → retreat 순으로 실행.
    plan: mp.sequence.Sequencer 또는 mp.stacking.Stacker 가 낸 Plan.
    s: 시뮬 Stacker (SimSuction 이 상자를 공구에 붙이는 데 필요). 실물에서는 None.
    """
    i = plan.index
    robot.follow(plan.approach)                  # 접근 (빈 손)

    robot.suction_on()
    if s is not None:
        robot.suction._attach(i)                 # 시뮬: 상자를 공구에 붙임

    def carry_box(q):
        if s is not None:
            s.d.qpos[:] = q
            s.carry_box(i, plan.held_geom)
    robot.follow(plan.carry, box_follow=carry_box if s is not None else None)

    if s is not None:
        s.set_box(i, plan.center, plan.quat)     # 시뮬: 상자를 놓인 자리에 세팅
        robot.suction._detach()
    robot.suction_off()

    robot.follow(plan.retreat)                   # 후퇴 (빈 손)
