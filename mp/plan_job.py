"""작업 파일(JSON)의 상자를 순서대로 계획해 경로를 JSON 으로 내고, 원하면 재생한다.

  uv run python plan_job.py example_job.json --out plan.json     # 계획만
  uv run python plan_job.py example_job.json --view              # 뷰어에서 재생 (창을 닫으면 종료)
  uv run python plan_job.py example_job.json --mp4 out.mp4       # 창 없이 MP4 로 저장

작업 파일: 놓는 순서대로 적은 목록. 한 줄이 상자 하나다.
  {"id": "a", "type": 6, "center": [1.45, 0.11, 0.123], "size_xyz": [0.10, 0.16, 0.16]}
    type      stacking.BOXES 의 상자 종류 (1~6)
    center    놓인 상자의 중심 (m)
    size_xyz  놓인 상자의 x, y, z 방향 길이 (m). 상자 치수의 순열이어야 한다
    frame     "world"(기본, 프레임 모서리 바닥 기준) 또는 "robot"(레일 0 일 때 팔 베이스 기준)
    pick_xy   집는 위치 (월드 x, y). 없으면 stacking.PICK_XY
    grip      흡착점이 윗면 중심에서 벗어난 거리 (가로, 세로) m. 없으면 (0, 0)
    mode      "top" / "side". 없으면 size_xyz 로 정한다

결과 파일: 상자마다 {id, ok, reason, mode, approach, carry, retreat}.
  approach / carry / retreat 는 경유 자세 목록이고, 한 자세는 [레일(m), 관절1..6(rad)] 이다.
  approach 끝에서 흡착을 켜고, carry 끝에서 끈다.
  실패한 상자는 건너뛴다 (그 뒤 상자는 그 상자가 없는 장면에서 계획된다).
"""
from __future__ import annotations

import argparse
import json
import os
import time

import mujoco
import mujoco.viewer
import numpy as np

from stacking import Stacker, set_view


def plan_job(job):
    """반환: (Stacker, 성공한 Plan 목록, 결과 dict 목록)."""
    s = Stacker([b["type"] for b in job])
    plans, out = [], []
    for i, b in enumerate(job):
        t = time.time()
        p = s.plan(i, b["center"], b["size_xyz"], mode=b.get("mode"), frame=b.get("frame", "world"),
                   grip=tuple(b.get("grip", (0.0, 0.0))), pick_xy=b.get("pick_xy"))
        name = b.get("id", i)
        print(f"상자 {name} (종류 {b['type']}): {'계획 완료, 방식 ' + p.mode if p.ok else '실패 - ' + p.reason}  ({time.time() - t:.1f}s)")
        r = dict(id=name, ok=p.ok, reason=p.reason, mode=p.mode)
        if p.ok:
            s.commit(p)
            plans.append(p)
            for part in ("approach", "carry", "retreat"):
                r[part] = [np.round(q, 6).tolist() for q in getattr(p, part)]
        out.append(r)
    return s, plans, out


def run(s: Stacker, plans):
    """재생 프레임을 차례로 내놓는다. 내놓을 때마다 s.d 가 그 순간의 장면이다."""
    for i in range(len(s.types)):
        s._hide(i)
    for p in plans:
        s.spawn(p.index, p.pick_xy)
        for q, holding, phase in s.frames(p):
            s.d.qpos[:] = q
            if holding:
                s.carry_box(p.index, p.held_geom)                # 든 상자는 공구를 따라간다
            elif phase == "retreat":
                s.set_box(p.index, p.center, p.quat)
            mujoco.mj_forward(s.m, s.d)
            yield p.index, phase


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("job", help="작업 파일 (JSON)")
    ap.add_argument("--out", help="경로를 이 JSON 파일에 저장")
    ap.add_argument("--view", action="store_true", help="뷰어에서 재생")
    ap.add_argument("--mp4", help="이 경로에 MP4 로 저장 (창을 띄우지 않는다)")
    args = ap.parse_args()

    job = json.loads(open(args.job, encoding="utf-8").read())
    s, plans, out = plan_job(job)
    for p, b in zip(plans, [b for b, r in zip(job, out) if r["ok"]]):
        p.pick_xy = b.get("pick_xy")                             # 재생할 때 상자를 놓아 둘 자리
    print(f"{len(plans)} / {len(job)} 개 계획 성공")
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False)
        print(f"저장: {args.out}")
    if not (args.view or args.mp4):
        return
    for name in ("wall_front", "wall_back", "wall_end"):        # 안이 보이도록 벽을 반투명하게
        s.m.geom_rgba[s.m.geom(name).id, 3] = 0.12

    if args.mp4:
        import subprocess
        import imageio_ffmpeg
        W, H = 1280, 640                                         # 모델의 화면 밖 버퍼(1280x670) 안에 들어가는 크기
        ren = mujoco.Renderer(s.m, H, W)
        ff = subprocess.Popen([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                               "-r", "30", "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", args.mp4], stdin=subprocess.PIPE)
        count = 0
        for n, _ in enumerate(run(s, plans)):
            if n % 2 == 0:
                ren.update_scene(s.d, camera="view")             # 모델에 정의한 기본 시점
                ff.stdin.write(ren.render().tobytes()); count += 1
        for _ in range(45):                                      # 마지막 장면을 잠깐 유지
            ff.stdin.write(ren.render().tobytes()); count += 1
        ff.stdin.close(); ff.wait()
        print(f"저장: {args.mp4} ({count} 프레임, {count / 30:.0f}초)")
        return

    with mujoco.viewer.launch_passive(s.m, s.d) as v:
        set_view(v.cam)
        while v.is_running():
            for _ in run(s, plans):
                if not v.is_running():
                    break
                v.sync()
                time.sleep(0.02)
            time.sleep(1.5)
    os._exit(0)      # 뷰어를 닫은 뒤 인터프리터 종료 과정에서 멈추는 문제를 피한다


if __name__ == "__main__":
    main()
