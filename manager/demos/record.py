"""데모 녹화 — 실물 Piper(CAN) 또는 시뮬레이션에서 자세·궤적을 `.npz` 로 저장.

저장 포맷 (modules/arm/skills.py `_load_demo` 가 읽음):
  np.savez(path, waypoints=np.ndarray[N, 7])     # 각 행 [rail, j1..j6]  (m / rad)

사용 예
  # 실물 Piper (기본). 키보드로 조인트 + 레일 조작, SPACE 로 현재 자세 기록
  python manager/demos/record.py --name mv_pos                      # Piper CAN only, 레일=0
  python manager/demos/record.py --name mv_pos --port /dev/ttyACM0  # Piper + Arduino 레일

  # 시뮬 (뷰어로 자세 잡기)
  mjpython manager/demos/record.py --name mv_pos --backend sim

공통 조작
  1 ~ 6       관절 1 ~ 6 선택
  0           레일 선택 (--port 있을 때만 의미)
  ← / →       선택 축 감소 / 증가
  ↑ / ↓       step 증가 / 감소
  SPACE       현재 자세를 keyframe 추가
  BACKSPACE   마지막 keyframe 삭제
  h           관절 모두 0° (home)  [실물 전용 — 팔이 움직임, 주의]
  s           저장 후 종료
  q / ESC     저장 없이 종료
"""
from __future__ import annotations

import argparse
import math
import os
import select
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent         # manager/demos/
MANAGER = HERE.parent                          # manager/
ROOT = MANAGER.parent                          # repo root
sys.path.insert(0, str(ROOT / "modules" / "arm"))
sys.path.insert(0, str(ROOT / "modules" / "rail"))
sys.path.insert(0, str(ROOT / "manager"))


# ===== 공통 상수 =====
AXIS_NAMES = ["rail(m)", "j1", "j2", "j3", "j4", "j5", "j6"]
# Piper 조인트 한계 (deg)
JOINT_LIMITS_DEG = [
    (-149.0, 149.0), (0.0, 179.0), (-169.0, 0.0),
    (-99.0, 99.0), (-69.0, 69.0), (-119.0, 119.0),
]
# 레일 한계 (m). Rail.RANGE 와 일치.
RAIL_RANGE = (0.0, 1.0)         # 0 = 투입 위치쪽 끝


# ============================================================
#  실물 Piper + Arduino 레일 녹화
# ============================================================
def _read_keys(fd):
    """stdin 에서 눌린 키 비차단 읽기. piper_keyboard.read_keys 와 동일."""
    if not select.select([fd], [], [], 0)[0]:
        return []
    data = os.read(fd, 64).decode(errors="ignore")
    arrows = {"A": "up", "B": "down", "C": "right", "D": "left"}
    keys, i = [], 0
    while i < len(data):
        c = data[i]
        if c == "\x1b" and i + 2 < len(data) and data[i + 1] in "[O":
            keys.append(arrows.get(data[i + 2], ""))
            i += 3
        elif c == "\x1b":          # ESC 혼자
            keys.append("esc"); i += 1
        elif c == "\x7f" or c == "\x08":   # BACKSPACE / DEL
            keys.append("backspace"); i += 1
        elif c == " ":
            keys.append("space"); i += 1
        else:
            keys.append(c); i += 1
    return keys


def _step_toward(pos, vel, goal, vmax, acc, dt):
    """사다리꼴 속도 프로파일 — piper_keyboard.step_toward 동일."""
    err = goal - pos
    v_des = math.copysign(min(vmax, math.sqrt(2 * acc * abs(err))), err)
    dv = max(-acc * dt, min(acc * dt, v_des - vel))
    vel += dv
    nxt = pos + vel * dt
    if abs(err) < 0.002 and abs(vel) < acc * dt * 2 or (goal - nxt) * err < 0:
        return goal, 0.0
    return nxt, vel


def record_real(args):
    import termios, tty

    from piper_sdk import C_PiperInterface_V2

    LOOP_HZ, DISPLAY_HZ = 200, 20
    MAX_LEAD_DEG = 10.0
    MOVE_J = 0x01

    # --- Piper 연결 + enable ---
    try:
        piper = C_PiperInterface_V2(args.can)
        piper.ConnectPort()
    except Exception as e:
        sys.exit(f"[에러] {args.can} 연결 실패: {e}\n"
                f"  → sudo ip link set {args.can} up type can bitrate 1000000")

    def read_joints_deg():
        js = piper.GetArmJointMsgs().joint_state
        return [getattr(js, f"joint_{i}") / 1000.0 for i in range(1, 7)]

    # 피드백 + enable 대기
    dl = time.time() + 3.0
    while time.time() < dl and piper.GetArmJointMsgs().time_stamp == 0:
        time.sleep(0.05)
    if piper.GetArmJointMsgs().time_stamp == 0:
        sys.exit("[에러] Piper 피드백 없음. 전원·CAN·bitrate(1000000) 확인.")
    dl = time.time() + 5.0
    while time.time() < dl and not piper.EnablePiper():
        time.sleep(0.01)
    print("[piper] enable OK  현재(deg):",
          ", ".join(f"{a:+.1f}" for a in read_joints_deg()), flush=True)

    # --- Arduino 레일 연결 (선택) ---
    rail = None
    if args.port:
        import arduino as _arduino
        from rail import RealRail
        ard = _arduino.init_shared(args.port)
        rail = RealRail(ard)
        print(f"[rail] 연결 OK  현재 {rail.read():.3f} m", flush=True)
    else:
        print("[rail] --port 없음 → 레일 값은 0.0 으로 기록됨", flush=True)

    # --- 초기 상태 ---
    joint_goal = read_joints_deg()       # 키보드로 바꾸는 최종 목표 (deg)
    joint_cmd = list(joint_goal)          # Piper 에 실제 보내는 보간값 (deg)
    joint_vel = [0.0] * 6
    rail_goal = rail.read() if rail else 0.0
    selected = 1                          # 0=rail, 1~6=joint
    STEP_JOINT = [0.1, 0.5, 1.0, 2.0, 5.0]   # deg
    STEP_RAIL  = [0.001, 0.005, 0.01, 0.02, 0.05]  # m
    step_idx_joint = 2                    # 1.0 deg
    step_idx_rail = 2                     # 0.01 m

    keyframes: list[np.ndarray] = []
    dt = 1.0 / LOOP_HZ
    next_tick = time.monotonic()
    next_display = 0.0

    out_path = HERE / f"{args.name}.npz"
    print(f"\n출력: {out_path}", flush=True)
    print("조작: 1~6 조인트, 0 레일, ←→ 이동, ↑↓ step, SPACE 기록, "
          "BACKSPACE 삭제, h home, s 저장, q 취소\n", flush=True)

    fd = sys.stdin.fileno()
    old_term = termios.tcgetattr(fd)
    tty.setcbreak(fd)
    done = {"save": False, "quit": False}

    try:
        while not done["save"] and not done["quit"]:
            # ---- 키 처리 ----
            for key in _read_keys(fd):
                if key in ("q", "Q", "esc"):
                    done["quit"] = True; break
                if key in ("s", "S"):
                    done["save"] = True; break
                if key == "space":
                    actual_deg = read_joints_deg()
                    rail_m = rail.read() if rail else 0.0
                    kf = np.array([rail_m] + [math.radians(a) for a in actual_deg])
                    keyframes.append(kf)
                    print(f"\r\x1b[K  + keyframe #{len(keyframes)}  rail={rail_m:+.3f} m  "
                          + " ".join(f"j{i+1}={actual_deg[i]:+6.1f}°" for i in range(6)), flush=True)
                    continue
                if key == "backspace":
                    if keyframes:
                        dropped = keyframes.pop()
                        print(f"\r\x1b[K  - keyframe #{len(keyframes)+1} 삭제", flush=True)
                    continue
                if key == "h":   # home — 조인트 모두 0도
                    joint_goal = [0.0] * 6
                    continue
                if key == "0":
                    selected = 0; continue
                if len(key) == 1 and key in "123456":
                    selected = int(key); continue
                if key in ("left", "right"):
                    sign = 1 if key == "right" else -1
                    if selected == 0:
                        if rail is None:
                            continue
                        rail_goal = float(np.clip(rail_goal + sign * STEP_RAIL[step_idx_rail],
                                                  RAIL_RANGE[0], RAIL_RANGE[1]))
                        rail.command(rail_goal)
                    else:
                        j = selected - 1
                        remaining = joint_goal[j] - joint_cmd[j]
                        if remaining * sign < 0:
                            joint_goal = list(joint_cmd); joint_vel = [0.0] * 6
                        else:
                            lo, hi = JOINT_LIMITS_DEG[j]
                            joint_goal[j] = max(lo, min(hi, joint_goal[j] + sign * STEP_JOINT[step_idx_joint]))
                    continue
                if key == "up":
                    if selected == 0:
                        step_idx_rail = min(len(STEP_RAIL) - 1, step_idx_rail + 1)
                    else:
                        step_idx_joint = min(len(STEP_JOINT) - 1, step_idx_joint + 1)
                    continue
                if key == "down":
                    if selected == 0:
                        step_idx_rail = max(0, step_idx_rail - 1)
                    else:
                        step_idx_joint = max(0, step_idx_joint - 1)
                    continue

            # ---- 조인트 보간 → Piper 전송 ----
            actual = read_joints_deg()
            for i in range(6):
                joint_cmd[i], joint_vel[i] = _step_toward(
                    joint_cmd[i], joint_vel[i], joint_goal[i],
                    args.joint_speed, args.joint_acc, dt)
                lead = joint_cmd[i] - actual[i]
                if abs(lead) > MAX_LEAD_DEG:
                    joint_cmd[i] = actual[i] + math.copysign(MAX_LEAD_DEG, lead)
                    joint_vel[i] = 0.0
            piper.MotionCtrl_2(0x01, MOVE_J, 100, 0x00)
            piper.JointCtrl(*[round(a * 1000) for a in joint_cmd])

            # ---- 화면 ----
            now = time.monotonic()
            if now >= next_display:
                next_display = now + 1.0 / DISPLAY_HZ
                sel_name = AXIS_NAMES[selected]
                if selected == 0:
                    sel_info = f"rail step={STEP_RAIL[step_idx_rail]*1000:.1f} mm  goal={rail_goal:+.3f}"
                else:
                    sel_info = f"joint step={STEP_JOINT[step_idx_joint]:.1f}°  goal={joint_goal[selected-1]:+.1f}°"
                rail_m = rail.read() if rail else 0.0
                cells = [f"\x1b[7mrail {rail_m:+.3f}\x1b[0m" if selected == 0 else f"rail {rail_m:+.3f}"]
                for i in range(6):
                    s = f"J{i+1} {actual[i]:+6.1f}"
                    if selected == i + 1:
                        s = f"\x1b[7m{s}\x1b[0m"
                    cells.append(s)
                sys.stdout.write(f"\r\x1b[K[sel={sel_name}]  " + " | ".join(cells)
                                 + f"   {sel_info}   kf={len(keyframes)}")
                sys.stdout.flush()

            next_tick += dt
            delay = next_tick - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            else:
                next_tick = time.monotonic()
    except KeyboardInterrupt:
        done["quit"] = True
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_term)
        sys.stdout.write("\n")
        if not args.disable_on_exit:
            print("[piper] enable 상태 유지. 팔 내리려면 수동으로 전원 끄거나 --disable-on-exit 로 재실행.", flush=True)
        else:
            print("[piper] disable 중 ...", flush=True)
            dl = time.time() + 3.0
            while piper.DisablePiper() and time.time() < dl:
                time.sleep(0.01)
        piper.DisconnectPort()

    if done["quit"] or not keyframes:
        print("저장 안 함.", flush=True); return

    wps = np.stack(keyframes)
    HERE.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, waypoints=wps)
    print(f"저장 완료: {out_path}  ({len(wps)} keyframes, shape={wps.shape})", flush=True)


# ============================================================
#  시뮬 녹화 (MuJoCo 뷰어)
# ============================================================
def record_sim(args):
    import mujoco, mujoco.viewer
    from stacking import Stacker, READY

    types = args.types or [1]
    s = Stacker(types)
    for i in range(len(types)):
        s._hide(i)
    s.q = READY.copy()
    s.d.qpos[:] = s.q
    mujoco.mj_forward(s.m, s.d)

    STEPS = [0.005, np.deg2rad(1.0)] + [np.deg2rad(1.0)] * 5
    keyframes: list[np.ndarray] = []
    axis_idx = 0
    step = STEPS[axis_idx]
    done = {"save": False, "quit": False}

    def banner():
        return (f"[axis={axis_idx} {AXIS_NAMES[axis_idx]}]  "
                f"step={step*1000:.2f} mm" if axis_idx == 0 else
                f"[axis={axis_idx} {AXIS_NAMES[axis_idx]}]  step={np.rad2deg(step):.2f} deg")

    def on_key(key):
        nonlocal axis_idx, step
        if ord('0') <= key <= ord('6'):
            axis_idx = key - ord('0'); step = STEPS[axis_idx]
            print(f"[sim] {banner()}  kf={len(keyframes)}", flush=True); return
        if key == 262:
            s.q[axis_idx] = float(np.clip(s.q[axis_idx] + step, s.lo[axis_idx], s.hi[axis_idx]))
            s.d.qpos[:] = s.q; mujoco.mj_forward(s.m, s.d)
        elif key == 263:
            s.q[axis_idx] = float(np.clip(s.q[axis_idx] - step, s.lo[axis_idx], s.hi[axis_idx]))
            s.d.qpos[:] = s.q; mujoco.mj_forward(s.m, s.d)
        elif key == 265:
            step = STEPS[axis_idx] = min(STEPS[axis_idx] * 2, np.deg2rad(20) if axis_idx else 0.05)
            print(f"[sim] {banner()}", flush=True)
        elif key == 264:
            step = STEPS[axis_idx] = max(STEPS[axis_idx] / 2, np.deg2rad(0.1) if axis_idx else 0.0005)
            print(f"[sim] {banner()}", flush=True)
        elif key == 32:
            keyframes.append(s.q.copy())
            print(f"  + keyframe #{len(keyframes)}: {np.array2string(s.q, precision=4)}", flush=True)
        elif key == 259:
            if keyframes:
                keyframes.pop(); print(f"  - keyframe 삭제 → {len(keyframes)}", flush=True)
        elif key in (ord('R'), ord('r'), ord('H'), ord('h')):
            s.q = READY.copy(); s.d.qpos[:] = s.q; mujoco.mj_forward(s.m, s.d)
            print("[sim] home 자세로 리셋", flush=True)
        elif key in (ord('S'), ord('s')):
            done["save"] = True
        elif key in (ord('Q'), ord('q'), 256):
            done["quit"] = True

    print(f"[sim] {banner()}  kf=0", flush=True)
    with mujoco.viewer.launch_passive(s.m, s.d, key_callback=on_key) as v:
        while v.is_running() and not done["save"] and not done["quit"]:
            mujoco.mj_forward(s.m, s.d); v.sync(); time.sleep(0.02)

    out_path = HERE / f"{args.name}.npz"
    if done["quit"] or not keyframes:
        print("저장 안 함.", flush=True); return
    wps = np.stack(keyframes)
    HERE.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, waypoints=wps)
    print(f"저장 완료: {out_path}  ({len(wps)} keyframes)", flush=True)


# ============================================================
def main():
    ap = argparse.ArgumentParser(description="데모 녹화 — manager/demos/<name>.npz")
    ap.add_argument("--name", required=True,
                    help="저장 파일 이름 (확장자 X). 예: mv_pos, approach_to_buffer")
    ap.add_argument("--backend", choices=["real", "sim"], default="real",
                    help="real: 실물 Piper CAN(기본), sim: MuJoCo 뷰어")
    # real 전용
    ap.add_argument("--can", default="can0", help="Piper CAN 인터페이스 (real)")
    ap.add_argument("--port", default=None, help="Arduino 시리얼 포트 — 레일까지 녹화할 때 (real, 선택)")
    ap.add_argument("--joint-speed", type=float, default=30.0, help="deg/s (real)")
    ap.add_argument("--joint-acc", type=float, default=90.0, help="deg/s^2 (real)")
    ap.add_argument("--disable-on-exit", action="store_true", help="종료 시 모터 disable (real)")
    # sim 전용
    ap.add_argument("--types", nargs="*", type=int, default=None,
                    help="MuJoCo 에 띄울 상자 타입들 (sim)")
    args = ap.parse_args()

    if args.backend == "real":
        record_real(args)
    else:
        record_sim(args)


if __name__ == "__main__":
    main()
