"""데모 녹화 — MuJoCo 뷰어에서 자세를 잡아 .npz 로 저장.

저장 포맷 (modules/arm/skills.py `_load_demo` 가 읽는 것):
  np.savez(path, waypoints=np.ndarray[N, 7])     # 각 행 [rail, j1..j6]  (m / rad)

조작:
  0           레일 (axis=0)
  1 ~ 6       관절 1 ~ 6
  ← / →       선택 축 감소 / 증가 (한 번당 step 만큼)
  ↑ / ↓       step 증가 / 감소  (레일 1 mm / 관절 1°  단위로 스케일)
  SPACE       현재 자세를 keyframe 으로 추가
  BACKSPACE   마지막 keyframe 삭제
  r           초기 자세 (home) 로 리셋
  s           저장 후 종료
  q / ESC     저장 없이 종료

사용 예 — Mac:
  mjpython manager/demos/record.py --name mv_pos
  mjpython manager/demos/record.py --name approach_to_buffer

사용 예 — Linux:
  python manager/demos/record.py --name mv_pos

저장 경로: manager/demos/<name>.npz
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

HERE = Path(__file__).resolve().parent         # manager/demos/
MANAGER = HERE.parent                          # manager/
ROOT = MANAGER.parent                          # repo root
sys.path.insert(0, str(ROOT / "modules" / "arm"))

from stacking import Stacker, READY            # noqa: E402  MuJoCo 모델 로더 재사용


# 축 이름 (표시용) 과 기본 step (각 축마다 독립 저장)
AXIS_NAMES = ["rail(m)", "j1(rad)", "j2(rad)", "j3(rad)", "j4(rad)", "j5(rad)", "j6(rad)"]
STEPS = [0.005, np.deg2rad(1.0)] + [np.deg2rad(1.0)] * 5     # rail 5 mm, 관절 1°


def banner(axis_idx, step, keyframes):
    axis = AXIS_NAMES[axis_idx]
    unit = "mm" if axis_idx == 0 else "deg"
    sv = step * 1000 if axis_idx == 0 else np.rad2deg(step)
    return (f"[axis={axis_idx} {axis}]  step={sv:.2f} {unit}  "
            f"keyframes={len(keyframes)}   "
            f"(0-6 axis, ←→ 이동, ↑↓ step, SPACE 추가, BACKSPACE 삭제, r home, s 저장, q 취소)")


def main():
    ap = argparse.ArgumentParser(description="MuJoCo sim 뷰어로 데모 녹화 → manager/demos/<name>.npz")
    ap.add_argument("--name", required=True,
                    help="저장 파일 이름 (확장자 없이). 예: mv_pos, approach_to_buffer")
    ap.add_argument("--types", nargs="*", type=int, default=[1],
                    help="MuJoCo Stacker 에 띄울 상자 타입들 (장면 구성용, 자세 녹화에는 영향 없음)")
    args = ap.parse_args()

    out_path = HERE / f"{args.name}.npz"

    # Stacker 로 MuJoCo 모델 로드 (manager/main.py 와 같은 모델·동일 자세)
    s = Stacker(args.types)
    for i in range(len(args.types)):
        s._hide(i)
    s.q = READY.copy()
    s.d.qpos[:] = s.q
    mujoco.mj_forward(s.m, s.d)

    # 녹화 상태
    keyframes: list[np.ndarray] = []
    axis_idx = 0
    step = STEPS[axis_idx]
    done = {"save": False, "quit": False}

    # 키 이벤트 핸들러 — viewer.launch_passive 의 key_callback
    def on_key(key):
        nonlocal axis_idx, step
        # 숫자 0 ~ 6: 축 선택
        if ord('0') <= key <= ord('6'):
            axis_idx = key - ord('0')
            step = STEPS[axis_idx]
            print(banner(axis_idx, step, keyframes), flush=True)
            return
        # 화살표 (MuJoCo GLFW 키 코드)
        if key == 262:   # RIGHT
            s.q[axis_idx] = float(np.clip(s.q[axis_idx] + step, s.lo[axis_idx], s.hi[axis_idx]))
            s.d.qpos[:] = s.q; mujoco.mj_forward(s.m, s.d)
        elif key == 263:   # LEFT
            s.q[axis_idx] = float(np.clip(s.q[axis_idx] - step, s.lo[axis_idx], s.hi[axis_idx]))
            s.d.qpos[:] = s.q; mujoco.mj_forward(s.m, s.d)
        elif key == 265:   # UP — step 증가
            step = STEPS[axis_idx] = min(STEPS[axis_idx] * 2, np.deg2rad(20) if axis_idx else 0.05)
            print(banner(axis_idx, step, keyframes), flush=True)
        elif key == 264:   # DOWN — step 감소
            step = STEPS[axis_idx] = max(STEPS[axis_idx] / 2, np.deg2rad(0.1) if axis_idx else 0.0005)
            print(banner(axis_idx, step, keyframes), flush=True)
        elif key == 32:    # SPACE — keyframe 추가
            keyframes.append(s.q.copy())
            print(f"  + keyframe #{len(keyframes)}: {np.array2string(s.q, precision=4)}", flush=True)
        elif key == 259:   # BACKSPACE — keyframe 삭제
            if keyframes:
                dropped = keyframes.pop()
                print(f"  - keyframe #{len(keyframes)+1} 삭제: {np.array2string(dropped, precision=4)}", flush=True)
        elif key == ord('R') or key == ord('r'):   # home 리셋
            s.q = READY.copy()
            s.d.qpos[:] = s.q; mujoco.mj_forward(s.m, s.d)
            print("  home 자세로 리셋", flush=True)
        elif key == ord('S') or key == ord('s'):   # 저장 종료
            done["save"] = True
        elif key == ord('Q') or key == ord('q') or key == 256:   # ESC/Q — 취소 종료
            done["quit"] = True

    print(banner(axis_idx, step, keyframes), flush=True)
    print(f"출력 파일: {out_path}", flush=True)

    with mujoco.viewer.launch_passive(s.m, s.d, key_callback=on_key) as v:
        while v.is_running() and not done["save"] and not done["quit"]:
            mujoco.mj_forward(s.m, s.d)
            v.sync()
            time.sleep(0.02)

    if done["quit"]:
        print("취소됨 (저장 안 함)", flush=True)
        return

    if not keyframes:
        print("keyframe 이 없음 → 저장 안 함", flush=True)
        return

    wps = np.stack(keyframes)        # (N, 7)
    HERE.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, waypoints=wps)
    print(f"저장 완료: {out_path}  ({len(wps)} keyframes)", flush=True)
    print(f"재생 확인: python -c \"import numpy as np; z=np.load('{out_path}'); print(z['waypoints'].shape)\"", flush=True)


if __name__ == "__main__":
    main()
