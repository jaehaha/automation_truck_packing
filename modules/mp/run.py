"""handoff 의 계획 파일(moves.json)에 있는 놓기 동작을 네 단계 방식으로 전부 돌려 성공률을 본다.

  uv run python run.py                  # 전체 검사 (결과: result.json)
  uv run python run.py --view           # 검사 뒤 뷰어에서 전체 재생 (창을 닫으면 종료)
  uv run python run.py --view --box 37  # 37번째 놓기만 재생 (그 앞 상자는 이미 놓인 상태)
  uv run python run.py --view --speed 3 # 3배 빠르게

맥에서 뷰어를 띄울 때는 python 대신 mjpython 을 쓴다:
  mjpython run.py --view
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

from sequence import BOXES, HERE, Sequencer
from stacking import ORIGIN, set_view

MOVES = HERE.parent / "packing" / "examples" / "list" / "moves_big.json"


def load(path):
    """moves.json 의 놓기 동작 -> [(상자 id, 종류, 중심(월드), 크기)]. 옆면 방식이라 x 길이가 상자 높이다."""
    job = []
    for mv in json.loads(Path(path).read_text())["moves"]:
        if mv["kind"] != "place":
            continue
        sz = mv["to"]["size_xyz_m"]
        dims = (max(sz[1:]), min(sz[1:]), sz[0])
        k = next((k for k, d in BOXES.items() if np.allclose(d, dims, atol=1e-6)), None)
        if k is None:
            raise SystemExit(f"상자 치수 {dims} 가 BOXES 에 없다. rules.EXTRA_BOXES 에 추가한다.")
        job.append((mv["box_id"], k, np.array(mv["to"]["center_m"]) + ORIGIN, sz))
    return job


def frames(seq, plans, only=None):
    s = seq.s
    for i in range(len(s.types)):
        s._hide(i)
    for p in plans:
        if only is not None and p.index != only:                 # 재생하지 않는 상자는 바로 놓인 상태로
            if only is None or p.index < only:
                s.set_box(p.index, p.center, p.quat)
            continue
        s.spawn(p.index)
        for q, holding, phase in s.frames(p):
            s.d.qpos[:] = q
            if holding:
                s.carry_box(p.index, p.held_geom)
            elif phase == "retreat":
                s.set_box(p.index, p.center, p.quat)
            mujoco.mj_forward(s.m, s.d)
            yield


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--moves", default=str(MOVES), help="handoff 의 계획 파일")
    ap.add_argument("--view", action="store_true", help="뷰어에서 재생")
    ap.add_argument("--box", type=int, help="이 번호(1부터)의 놓기만 재생")
    ap.add_argument("--speed", type=float, default=1.0, help="재생 속도 배율")
    args = ap.parse_args()

    job = load(args.moves)
    seq = Sequencer([k for _, k, _, _ in job])
    plans, rows, t0 = [], [], time.time()
    for i, (box_id, k, c, sz) in enumerate(job):
        t = time.time()
        p = seq.plan(i, c, sz)
        if p.ok:
            seq.s.commit(p)
            plans.append(p)
        rows.append(dict(n=i + 1, id=box_id, type=k, center=np.round(c, 4).tolist(), size=sz, ok=p.ok, reason=p.reason,
                         candidate=getattr(p, "candidate", None)))
        print(f"{i + 1:3d} {box_id} 종류 {k} x={c[0]:.3f} y={c[1]:.3f} z={c[2]:.3f}  "
              f"{'성공 (후보 ' + str(p.candidate + 1) + ')' if p.ok else '실패 - ' + p.reason}  ({time.time() - t:.1f}s)", flush=True)
    seq.save()

    zone = [r for r in rows if r["reason"].startswith("투입 구역")]
    tried = [r for r in rows if r not in zone]
    ok = [r for r in tried if r["ok"]]
    print(f"\n전체 놓기 {len(rows)}개 / 투입 구역이라 제외 {len(zone)}개 / 시도 {len(tried)}개")
    print(f"성공 {len(ok)} / {len(tried)}  ({100 * len(ok) / max(1, len(tried)):.1f}%)   걸린 시간 {time.time() - t0:.0f}s, 새로 계산한 틀 {seq.new_templates}개")
    why = {}
    for r in tried:
        if not r["ok"]:
            why.setdefault(r["reason"], []).append(r["n"])
    for reason, ns in why.items():
        print(f"  실패 {len(ns)}개: {reason}  -> 번호 {ns}")
    (HERE / "result.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1))

    if not args.view:
        return
    s = seq.s
    for name in ("wall_front", "wall_back", "wall_end"):        # 안이 보이도록 벽을 반투명하게
        s.m.geom_rgba[s.m.geom(name).id, 3] = 0.12
    only = None if args.box is None else args.box - 1
    if only is not None and not rows[only]["ok"]:
        raise SystemExit(f"{args.box}번은 실패한 놓기라 재생할 경로가 없다: {rows[only]['reason']}")
    with mujoco.viewer.launch_passive(s.m, s.d) as v:
        set_view(v.cam)
        while v.is_running():
            for _ in frames(seq, plans, only):
                if not v.is_running():
                    break
                v.sync()
                time.sleep(0.02 / args.speed)
            time.sleep(1.5)
    os._exit(0)


if __name__ == "__main__":
    main()
