"""통합 실행: Packing Model → Motion Planning → Robot 실행 → BoxTable 갱신.

../pipeline/details_by_modules/unified.md §2 참고.

  mjpython main.py                              # (기본) 시뮬만 + 뷰어
  mjpython main.py --mode sim --speed 4         # 4배속
  python  main.py --mode sim --no-view          # 뷰어 없이 계획만
  mjpython main.py --mode dual                  # ★ 실물 + 시뮬 (타이밍 A, 시뮬이 뷰어)
  python  main.py --mode real --no-view         # 실물만 (뷰어 없이)

dual / real 은 아두이노 포트를 열어야 한다: --port /dev/tty.usbmodem*
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "modules" / "arm"))
sys.path.insert(0, str(ROOT / "modules" / "transform"))

from sequence import BOXES, Sequencer                 # noqa: E402
from stacking import ORIGIN, set_view                 # noqa: E402
import transform as cf                                 # noqa: E402  Container Frame 변환

from robot import Robot                               # noqa: E402
from actions import pick_and_place                    # noqa: E402
from boxes import BoxTable                            # noqa: E402

MOVES = HERE / "sequence" / "moves.json"           # Packing Model 추론 결과 (표준 위치)
# 이전 호환: moves.json 이 없고 moves_big.json 만 있으면 그걸 쓴다
if not MOVES.exists() and (HERE / "sequence" / "moves_big.json").exists():
    MOVES = HERE / "sequence" / "moves_big.json"


# ---- 뷰어 상태 오버레이 (왼쪽 위에 현재 박스 정보) ----
# user_scn 에 라벨만 가진 투명 구를 하나 두고, 뷰 좌상단에 보이도록 월드 좌표를 잡는다.
# VIEW azimuth/elevation 기준으로 왼쪽 위에 해당하는 CF 영역 ≈ (x=0.0, y=0.56, z=1.0).
_STATUS_ANCHOR = np.array([0.00, 0.56, 1.00])

class _ViewerClosed(Exception):
    """뷰어 창이 사용자에 의해 닫혔음을 알리는 신호. step_cb 에서 발생, 재생 루프가 잡아 깔끔 종료."""
    pass


def _set_viewer_status(viewer_ctx, text: str):
    """MuJoCo 뷰어 좌상단에 상태 문자열을 띄운다 (라벨 1 개). 창 없으면 무시."""
    if viewer_ctx is None:
        return
    import mujoco as _mj
    with viewer_ctx.lock():
        scn = viewer_ctx.user_scn
        if scn.ngeom == 0:
            scn.ngeom = 1
        g = scn.geoms[0]
        _mj.mjv_initGeom(
            g,
            type=_mj.mjtGeom.mjGEOM_SPHERE,
            size=np.array([0.001, 0.001, 0.001]),
            pos=_STATUS_ANCHOR,
            mat=np.eye(3).flatten(),
            rgba=np.array([0.0, 0.0, 0.0, 0.0]),       # 투명 — 구는 안 보이고 라벨만
        )
        g.label = (text or "")[:99]                     # mjMAXLABEL = 100


def load_manifest(path: Path) -> list[dict]:
    """handoff 의 계획 파일 (moves.json) 의 'place' 동작을 뽑아 BoxTable 입력 형식으로.

    Packing 모델이 쓰는 좌표계:  Robot coords (RC) — stacking.ORIGIN 기준
      → CF 로 통일하려면 frame.rc_to_cf() 통과
      → Sequencer 는 아직 MW 좌표를 받으므로 MW (= RC + ORIGIN) 로 변환해서 넘긴다
    컨테이너 좌표계로 완전 통일 후에는 Sequencer 도 CF 를 받도록 바꾼다.
    """
    out = []
    for mv in json.loads(path.read_text())["moves"]:
        if mv["kind"] != "place":
            continue
        sz = mv["to"]["size_xyz_m"]
        dims = (max(sz[1:]), min(sz[1:]), sz[0])
        k = next((k for k, d in BOXES.items() if np.allclose(d, dims, atol=1e-6)), None)
        if k is None:
            raise SystemExit(f"상자 치수 {dims} 가 BOXES 에 없다")
        rc = np.array(mv["to"]["center_m"])              # Packing 원본 (RC)
        goal_mw = rc + ORIGIN                            # 현재 Sequencer 입력 (MW)
        goal_cf = cf.mw_to_cf(goal_mw)                   # 참고용 CF 값 (BoxTable 기록용)
        out.append(dict(box_id=mv["box_id"], type=k,
                        goal=goal_mw, goal_cf=goal_cf, size_xyz=sz))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--moves", default=str(MOVES))
    ap.add_argument("--mode", choices=["sim", "real", "dual"], default="sim",
                    help="sim=시뮬만, real=실물만(뷰어 X), dual=실물+시뮬(시뮬이 뷰어)")
    ap.add_argument("--port", default="/dev/tty.usbmodem0001", help="아두이노 시리얼 포트 (real/dual)")
    ap.add_argument("--can", default="can0", help="Piper 팔 CAN 인터페이스 (real/dual)")
    ap.add_argument("--no-view", action="store_true", help="뷰어 없이 돌린다 (sim 에서만 의미)")
    ap.add_argument("--speed", type=float, default=1.0,
                    help="모터 속도 배율. sim: 보간 step 사이 sleep 를 1/speed 로. "
                         "real/dual: 향후 RealArm 가 Piper MotionCtrl_2 speed_percent (100 × speed, 0-100 clip) 로 전달. "
                         "1.0 = 기본, 4.0 = 4배 빠름, 0.1 = 10배 느림 (안전).")
    args = ap.parse_args()

    # 모드별 뷰어 사용 여부 — real 은 뷰어 없음, dual 은 반드시 뷰어 있음
    use_viewer = (args.mode == "dual") or (args.mode == "sim" and not args.no_view)

    # ---- 1. 상자 목록 읽기 + BoxTable ----
    manifest = load_manifest(Path(args.moves))
    table = BoxTable()
    for e in manifest:
        table.add(e["box_id"], BOXES[e["type"]])
        table.set_goal(e["box_id"], e["goal"], e["size_xyz"])

    # ---- 2. 모션 플래너 (씬 공유) ----
    types = [e["type"] for e in manifest]
    seq = Sequencer(types)

    # ---- 3. Robot 백엔드 (모드별) ----
    # seq.s 가 실제 장면. 시뮬·듀얼에서는 그 model/data 를 Robot 이 가져다 쓴다.
    # 왼쪽 패널만 숨기고, 오른쪽 Control 패널은 유지 — 사용자가 슬라이더로 직접 액추에이터 조작 가능
    viewer_ctx = (mujoco.viewer.launch_passive(seq.s.m, seq.s.d,
                                               show_left_ui=False, show_right_ui=True)
                  if use_viewer else None)
    try:
        if viewer_ctx is not None:
            set_view(viewer_ctx.cam)
            for name in ("wall_end",):     # wall_front / wall_back 제거됨
                seq.s.m.geom_rgba[seq.s.m.geom(name).id, 3] = 0.12

        def step_cb():
            # 뷰어가 닫혔으면 즉시 중단 — 상위에서 잡아 cleanup 하고 subprocess 종료
            if viewer_ctx is not None and not viewer_ctx.is_running():
                raise _ViewerClosed()
            mujoco.mj_forward(seq.s.m, seq.s.d)
            if viewer_ctx is not None:
                viewer_ctx.sync()

        if args.mode == "sim":
            robot = Robot.sim(seq.s.m, seq.s.d, on_step=step_cb)
        elif args.mode == "real":
            robot = Robot.real(arduino_port=args.port, can=args.can, on_step=step_cb)
        else:  # dual
            robot = Robot.dual(seq.s.m, seq.s.d, arduino_port=args.port, can=args.can, on_step=step_cb)

        # ---- 4. 순서대로 돌리기 ----
        # 1단계: 모든 상자 플래닝 (계획 성공/실패를 BoxTable 에 기록)
        t0 = time.time()
        plans = []
        for n, e in enumerate(manifest):
            seq.s._hide(n)
            p = seq.plan(n, e["goal"], e["size_xyz"])
            print(f"{n+1:3d} {e['box_id']} 종류 {e['type']} "
                  f"{'성공 (후보 ' + str(getattr(p, 'candidate', 0)+1) + ')' if p.ok else '실패 - ' + p.reason}", flush=True)
            if p.ok:
                seq.s.commit(p)
                plans.append((e, p))
                table.done(e["box_id"])
            else:
                table.fail(e["box_id"])
        seq.save()
        print(f"\n전체 {len(manifest)}개 / 성공 {len(plans)} / 실패 {len(manifest)-len(plans)}   "
              f"계획 시간 {time.time()-t0:.0f}s")

        if viewer_ctx is None:
            return

        # 2단계: 뷰어에서 skills 로 실행 재생 (장치들에 명령이 들어간다)
        for i in range(len(seq.s.types)):
            seq.s._hide(i)
        total = len(plans)
        try:
            for n, (e, p) in enumerate(plans):
                if not viewer_ctx.is_running():
                    break
                goal_cf = e.get("goal_cf", None)
                if goal_cf is not None:
                    goal_txt = f"goal_cf=({goal_cf[0]:.3f},{goal_cf[1]:.3f},{goal_cf[2]:.3f})"
                else:
                    g = e["goal"]
                    goal_txt = f"goal=({g[0]:.3f},{g[1]:.3f},{g[2]:.3f})"
                status = f"[Packing {n+1}/{total}]  {e['box_id']} type={e['type']}  {goal_txt}"
                _set_viewer_status(viewer_ctx, status)
                print(status, flush=True)
                seq.s.spawn(p.index)
                pick_and_place(robot, p, s=seq.s, speed=args.speed)
                time.sleep(0.3 / max(args.speed, 1e-3))

            _set_viewer_status(viewer_ctx, f"[Packing 완료]  {total}/{total}  (창 닫으면 종료)")

            # 재생이 끝난 뒤에도 창은 유지. 창 닫으면 종료.
            while viewer_ctx.is_running():
                time.sleep(0.1)
        except _ViewerClosed:
            print("\n[중단] 사용자가 뷰어를 닫음 — Packing 중지.", flush=True)
    finally:
        if viewer_ctx is not None:
            viewer_ctx.close()
    os._exit(0)


if __name__ == "__main__":
    main()
