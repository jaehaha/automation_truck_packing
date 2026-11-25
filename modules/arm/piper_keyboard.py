#!/usr/bin/env python3
"""AgileX Piper 키보드 조작 프로그램.

  1~6      : 움직일 조인트 선택
  0        : 모든 조인트를 0도(영점 자세)로 복귀
  ← / →    : 선택된 조인트 각도를 감소 / 증가
             움직이는 중에 반대 방향 키를 누르면 그 자리에서 바로 정지
  ↑ / ↓    : 조인트 한 번 누를 때 움직이는 각도(step) 증가 / 감소
             (키 입력은 "목표 각도"만 바꾸고, 팔은 그 목표까지 부드럽게 보간해서 따라감)
  q        : 종료 (팔은 현재 자세를 유지한 채로 둠)

실행 전 CAN 인터페이스가 올라와 있어야 합니다:
  sudo ip link set can0 up type can bitrate 1000000
"""
import argparse
import math
import os
import select
import sys
import termios
import time
import tty

from piper_sdk import C_PiperInterface_V2

# 조인트 한계 (deg). SDK 문서 기준에서 1도씩 여유를 두되, 영점(0도)은 갈 수 있게 둠.
JOINT_LIMITS_DEG = [
    (-149.0, 149.0),
    (0.0, 179.0),
    (-169.0, 0.0),
    (-99.0, 99.0),
    (-69.0, 69.0),
    (-119.0, 119.0),
]
STEPS_DEG = [0.1, 0.5, 1.0, 2.0, 5.0]
LOOP_HZ = 200         # 보간된 명령을 보내는 주기
DISPLAY_HZ = 20
MAX_LEAD_DEG = 10.0   # 보간 명령이 실제 각도보다 이 이상 앞서가지 않게 함 (팔이 못 따라올 때 대비)
MOVE_J = 0x01


def read_joints_deg(piper):
    js = piper.GetArmJointMsgs().joint_state
    return [getattr(js, f"joint_{i}") / 1000.0 for i in range(1, 7)]


def step_toward(pos, vel, goal, vmax, acc, dt):
    """가감속 제한(사다리꼴 속도 프로파일)으로 pos를 goal 쪽으로 dt만큼 전진. (pos, vel) 반환."""
    err = goal - pos
    # 남은 거리에서 멈출 수 있는 최대 속도 → 목표 근처에서 자연스럽게 감속
    v_des = math.copysign(min(vmax, math.sqrt(2 * acc * abs(err))), err)
    dv = max(-acc * dt, min(acc * dt, v_des - vel))
    vel += dv
    nxt = pos + vel * dt
    if abs(err) < 0.002 and abs(vel) < acc * dt * 2 or (goal - nxt) * err < 0:
        return goal, 0.0   # 도착(또는 지나침) → 목표에 고정
    return nxt, vel


def wait_for_feedback(piper, timeout=3.0):
    """조인트 피드백이 실제로 들어올 때까지 대기. 못 받으면 False."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if piper.GetArmJointMsgs().time_stamp > 0:
            return True
        time.sleep(0.05)
    return False


def enable_arm(piper, timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if piper.EnablePiper():
            return True
        time.sleep(0.01)
    return False


def read_keys(fd):
    """대기 없이 눌린 키들을 읽어 'left'/'right'/'up'/'down' 또는 문자 리스트로 반환."""
    if not select.select([fd], [], [], 0)[0]:
        return []
    data = os.read(fd, 64).decode(errors="ignore")
    arrows = {"A": "up", "B": "down", "C": "right", "D": "left"}
    keys, i = [], 0
    while i < len(data):
        if data[i] == "\x1b" and i + 2 < len(data) and data[i + 1] in "[O":
            keys.append(arrows.get(data[i + 2], ""))
            i += 3
        else:
            keys.append(data[i])
            i += 1
    return keys


def highlight(text, on):
    return f"\x1b[7m{text}\x1b[0m" if on else text


def main():
    ap = argparse.ArgumentParser(description="Piper 키보드 조인트 조작")
    ap.add_argument("--can", default="can0", help="CAN 인터페이스 이름 (기본 can0)")
    ap.add_argument("--joint-speed", type=float, default=30.0, help="조인트 최대 속도 deg/s (기본 30)")
    ap.add_argument("--joint-acc", type=float, default=90.0, help="조인트 가속도 deg/s^2 (기본 90)")
    ap.add_argument("--disable-on-exit", action="store_true",
                    help="종료 시 모터를 disable (팔이 떨어질 수 있음!)")
    args = ap.parse_args()

    try:
        piper = C_PiperInterface_V2(args.can)
        piper.ConnectPort()
    except Exception as e:
        sys.exit(f"[에러] {args.can} 연결 실패: {e}\n"
                 f"  → sudo ip link set {args.can} up type can bitrate 1000000")

    if not wait_for_feedback(piper):
        sys.exit("[에러] 로봇팔에서 피드백이 오지 않습니다. 전원 / CAN 케이블 / bitrate(1000000)를 확인하세요.")

    print("현재 조인트(deg):", ", ".join(f"{a:.1f}" for a in read_joints_deg(piper)))
    print("모터 enable 중...")
    if not enable_arm(piper):
        sys.exit("[에러] 모터 enable 실패. 비상정지 / 티칭 모드 여부를 확인하세요.")

    # 현재 자세를 목표값으로 삼아야 enable 직후 팔이 튀지 않는다.
    joint_goal = read_joints_deg(piper)   # 키보드로 바꾸는 최종 목표
    joint_cmd = list(joint_goal)          # 매 주기 팔에 보내는, 목표를 향해 보간되는 값
    joint_vel = [0.0] * 6
    selected = 0
    step_idx = STEPS_DEG.index(1.0)
    dt = 1.0 / LOOP_HZ
    next_tick = time.monotonic()
    next_display = 0.0

    print(__doc__)
    fd = sys.stdin.fileno()
    old_term = termios.tcgetattr(fd)
    tty.setcbreak(fd)
    try:
        while True:
            for key in read_keys(fd):
                if key in ("q", "Q"):
                    return
                if key == "0":
                    joint_goal = [0.0] * 6
                elif len(key) == 1 and key in "123456":
                    selected = int(key) - 1
                elif key in ("left", "right"):
                    sign = 1 if key == "right" else -1
                    remaining = joint_goal[selected] - joint_cmd[selected]
                    if remaining * sign < 0:
                        # 움직이는 중에 반대 방향 키 → 반대로 가지 않고 그 자리에서 바로 정지.
                        # (0번 복귀 중이면 모든 조인트를 함께 멈춘다)
                        joint_goal = list(joint_cmd)
                        joint_vel = [0.0] * 6
                    else:
                        lo, hi = JOINT_LIMITS_DEG[selected]
                        joint_goal[selected] = max(
                            lo, min(hi, joint_goal[selected] + sign * STEPS_DEG[step_idx]))
                elif key == "up":
                    step_idx = min(len(STEPS_DEG) - 1, step_idx + 1)
                elif key == "down":
                    step_idx = max(0, step_idx - 1)

            status = piper.GetArmStatus().arm_status
            actual = read_joints_deg(piper)
            for i in range(6):
                joint_cmd[i], joint_vel[i] = step_toward(
                    joint_cmd[i], joint_vel[i], joint_goal[i],
                    args.joint_speed, args.joint_acc, dt)
                # 팔이 못 따라오면(비상정지 등) 명령이 혼자 멀리 가버리지 않게 묶어 둔다.
                lead = joint_cmd[i] - actual[i]
                if abs(lead) > MAX_LEAD_DEG:
                    joint_cmd[i] = actual[i] + math.copysign(MAX_LEAD_DEG, lead)
                    joint_vel[i] = 0.0
            # 속도는 위 보간이 정하므로 컨트롤러 속도 비율은 100%로 두고 촘촘한 명령을 그대로 따라가게 한다.
            piper.MotionCtrl_2(0x01, MOVE_J, 100, 0x00)
            piper.JointCtrl(*[round(a * 1000) for a in joint_cmd])

            now = time.monotonic()
            if now >= next_display:
                next_display = now + 1.0 / DISPLAY_HZ
                cells = [highlight(f"J{i + 1} {actual[i]:7.1f}", i == selected) for i in range(6)]
                sys.stdout.write(
                    f"\r\x1b[K{' | '.join(cells)}  →목표 J{selected + 1}: {joint_goal[selected]:.1f}°"
                    f"  step {STEPS_DEG[step_idx]}°  [{status.ctrl_mode.name}/{status.arm_status.name}]")
                sys.stdout.flush()
            next_tick += dt
            delay = next_tick - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            else:
                next_tick = time.monotonic()   # 밀렸으면 따라잡으려 하지 않고 다시 맞춤
    except KeyboardInterrupt:
        pass
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_term)
        print()
        if args.disable_on_exit:
            print("모터 disable 중...")
            deadline = time.time() + 3.0
            while piper.DisablePiper() and time.time() < deadline:
                time.sleep(0.01)
        else:
            print("종료: 팔은 enable 상태로 현재 자세를 유지합니다.")
        piper.DisconnectPort()


if __name__ == "__main__":
    main()
