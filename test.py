"""장치별 단위 테스트 — Suction / Rail 이 각각 잘 돌아가는지 확인.

실물 Arduino 로 바로 명령을 보내서 동작 확인한다. Main pipeline 과 무관하게 독립 호출.

사용 예
  python test.py suction --port /dev/ttyACM0        # 밸브만 ON/OFF/grip/un_grip
  python test.py rail    --port /dev/ttyACM0        # 레일만 command / mv / home
  python test.py all     --port /dev/ttyACM0        # 둘 다 순차 실행

Mac 포트: /dev/tty.usbmodem*     Linux 포트: /dev/ttyACM* 또는 /dev/ttyUSB*
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "modules" / "suction"))
sys.path.insert(0, str(ROOT / "modules" / "rail"))
sys.path.insert(0, str(ROOT / "modules" / "transform"))
sys.path.insert(0, str(ROOT / "manager"))

import arduino as _arduino             # noqa: E402
from suction import RealSuction        # noqa: E402
from rail import RealRail              # noqa: E402


def section(title: str):
    print(f"\n==== {title} ====", flush=True)


# ---------- Suction ----------
def test_suction(ard, dwell: float = 1.0):
    """SimSuction/RealSuction 모두 쓰는 API 한 바퀴.

    on/off      : 밸브 저수준
    grip(mode)  : 상위 API (suction.py 가 on() 호출)
    un_grip()   : off() 호출
    """
    s = RealSuction(ard)

    section("Suction — 초기 상태")
    print(f"holding() = {s.holding()}   (기대: False)")

    section("Suction — on() → off()")
    s.on()
    time.sleep(dwell)
    print(f"holding() = {s.holding()}   (기대: True)")
    s.off()
    time.sleep(dwell)
    print(f"holding() = {s.holding()}   (기대: False)")

    section("Suction — grip('pack') → un_grip()")
    s.grip("pack")
    time.sleep(dwell)
    print(f"holding() = {s.holding()}   (기대: True)")
    s.un_grip()
    time.sleep(dwell)
    print(f"holding() = {s.holding()}   (기대: False)")

    section("Suction — grip('unpack') → un_grip()")
    s.grip("unpack")
    time.sleep(dwell)
    print(f"holding() = {s.holding()}   (기대: True)")
    s.un_grip()
    print(f"holding() = {s.holding()}   (기대: False)")


# ---------- Rail ----------
def test_rail(ard, dwell: float = 2.0):
    """RealRail 저수준(command) + 고수준(mv) + home 테스트.

    - command(m)  : RC m 단위 저수준 (SimRail/RealRail 공통)
    - mv(x_cf)    : CF x (m) → 내부 _cf_x_to_rail_m 캘리브를 거쳐 command() 호출
                    x_cf == 0 은 홈 복귀
    - home()      : 레일 0 mm 복귀
    """
    r = RealRail(ard)        # __init__ 안에서 INITIALIZE 자동 전송
    print(f"zero_offset_m = {r.zero_offset_m}")

    section("Rail — read() 초기값")
    print(f"read() = {r.read():.3f} m")

    section("Rail — command(m) 저수준")
    for m in (0.0, 0.1, -0.1, 0.0):
        print(f"command({m:+.2f}) → ...", flush=True)
        r.command(m)
        time.sleep(dwell)
        print(f"  read() = {r.read():.3f} m")

    section("Rail — mv(x_cf) 고수준 (CF x)")
    # 안전한 CF x 몇 지점 (RealRail 의 zero_offset_m=-0.49, RAIL_ZERO_CF_X=0.04 기본값 기준
    # arm_base CF x = x_cf - APPROACH_X_OFFSET(0.015) 가 rail 범위에 들어가는 값)
    for x_cf in (0.3, 0.6, 0.9, 0.3):
        print(f"mv({x_cf:+.2f}) → ...", flush=True)
        try:
            r.mv(x_cf)
            time.sleep(dwell)
            print(f"  read() = {r.read():.3f} m")
        except ValueError as e:
            print(f"  범위 초과: {e}")

    section("Rail — mv(0) 홈 복귀")
    r.mv(0.0)
    time.sleep(dwell)
    print(f"read() = {r.read():.3f} m   (기대: ≈{r.zero_offset_m:.3f})")

    section("Rail — home() 직접 호출")
    r.home()
    time.sleep(dwell)
    print(f"read() = {r.read():.3f} m")


# ---------- main ----------
def main():
    ap = argparse.ArgumentParser(description="Suction/Rail 개별 단위 테스트")
    ap.add_argument("device", choices=["suction", "rail", "all"],
                    help="테스트 대상")
    ap.add_argument("--port", default="/dev/ttyACM0",
                    help="Arduino 시리얼 포트 (Mac: /dev/tty.usbmodem*, Linux: /dev/ttyACM0)")
    ap.add_argument("--dwell", type=float, default=1.0,
                    help="각 명령 사이 대기 (초). rail 은 2배로 늘어남")
    args = ap.parse_args()

    print(f"Arduino 연결 {args.port} ...", flush=True)
    ard = _arduino.init_shared(args.port)
    print("연결 OK\n", flush=True)

    try:
        if args.device in ("suction", "all"):
            test_suction(ard, dwell=args.dwell)
        if args.device in ("rail", "all"):
            test_rail(ard, dwell=args.dwell * 2)
        print("\n모든 테스트 완료.", flush=True)
    except KeyboardInterrupt:
        print("\n중단 (Ctrl-C)", flush=True)
    except Exception as e:
        print(f"\n예외: {type(e).__name__}: {e}", flush=True)
        raise


if __name__ == "__main__":
    main()
