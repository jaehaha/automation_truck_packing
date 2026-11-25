"""하나의 아두이노에 레일 + 흡착 밸브가 같이 물려 있을 때 쓰는 공유 시리얼 연결.
펌웨어: modules/rail_vacuum.ino

프로토콜 (ASCII, 개행 종료, 115200 baud, 모든 명령 대문자).
주요 명령:
  INITIALIZE | INIT  현재 위치를 0 mm 로 설정. 응답: "INITIALIZED\\nPOSITION: ..."
  MOVE <mm>          절대 위치 (mm) 로 이동. 응답: "MOVING TO: <mm>" (즉시) → "DONE\\nPOSITION:...\\nSTEPS:..." (완료시)
  REL <mm>           상대 이동. 응답 MOVE 와 동일
  HOME | H           0 mm 복귀
  STOP | S           정지
  POS                현재 위치 조회. 응답: "POSITION: <mm> mm (estimated)\\nSTEPS: <int>"
  STATUS             여러 줄 상태 출력
  ON                 흡착 ON. 응답: "SUCTION ON"
  OFF                흡착 OFF. 응답: "SUCTION OFF"
  MEASURE <mm>       위치 보정 (아두이노 위치가 실제와 어긋났을 때)
  CAL <mm>           스텝/mm 재조정

주의:
  - 이동 중에 MOVE/REL/F/B/HOME 보내면 "ERROR: Motor busy" 가 돌아온다 → 타이밍 A 에서는 무시.
  - 여러 줄 응답이 많아, 다음 명령 보내기 전에 input buffer 를 비운다.
  - 하나의 포트를 레일·흡착이 공유하므로 모든 송수신은 _lock 아래에서 한 명령 단위로.
"""
from __future__ import annotations

import threading
import time
from typing import Optional

import serial


class Arduino:
    def __init__(self, port: str = "/dev/tty.usbmodem0001", baud: int = 115200, timeout: float = 2.0):
        self._lock = threading.Lock()
        self.ser = serial.Serial(port, baud, timeout=timeout)
        time.sleep(2.0)                             # 아두이노 리셋 대기
        self.ser.reset_input_buffer()

    # ---------- 저수준 ----------
    def _write(self, cmd: str):
        self.ser.write((cmd + "\n").encode("ascii"))
        self.ser.flush()

    def _readline(self) -> str:
        return self.ser.readline().decode("ascii", errors="replace").strip()

    # ---------- 고수준 (lock 안에서 쓰는 API) ----------
    def send(self, cmd: str) -> str:
        """명령 쓰고 첫 응답 한 줄 받기. 이전 명령의 지연 응답은 버린다 (타이밍 A)."""
        with self._lock:
            self.ser.reset_input_buffer()           # 이전 DONE/POSITION 같은 지연 응답 버림
            self._write(cmd)
            resp = self._readline()
            if resp.startswith("ERROR"):
                raise RuntimeError(f"아두이노 오류: {resp} (명령: {cmd})")
            if not resp:
                raise RuntimeError(f"아두이노 응답 없음 (명령: {cmd})")
            return resp

    def send_await(self, cmd: str, marker: str, timeout: float = 5.0) -> list[str]:
        """명령 쓰고 marker 가 포함된 줄이 나올 때까지 모든 줄 모아 반환.
        INITIALIZE 처럼 여러 줄 응답을 다 받아야 할 때 쓴다."""
        with self._lock:
            self.ser.reset_input_buffer()
            self._write(cmd)
            lines: list[str] = []
            t0 = time.time()
            while time.time() - t0 < timeout:
                line = self._readline()
                if not line:
                    continue
                lines.append(line)
                if line.startswith("ERROR"):
                    raise RuntimeError(f"아두이노 오류: {line} (명령: {cmd}, 받은: {lines})")
                if marker in line:
                    return lines
            raise TimeoutError(f"'{marker}' 응답 안 옴 (명령: {cmd}, 받은: {lines})")

    def drain(self, max_lines: int = 50) -> int:
        """지금 들어와 있는 줄을 다 읽어서 비움. 반환: 비운 줄 수."""
        with self._lock:
            n = 0
            while self.ser.in_waiting > 0 and n < max_lines:
                self._readline()
                n += 1
            return n

    def close(self):
        try:
            self.ser.close()
        except Exception:
            pass


# 한 프로세스에서 하나만 두는 전역 핸들.
_shared: Optional[Arduino] = None


def init_shared(port: str, baud: int = 115200) -> Arduino:
    global _shared
    if _shared is None:
        _shared = Arduino(port=port, baud=baud)
    return _shared


def get_shared() -> Arduino:
    if _shared is None:
        raise RuntimeError("Arduino 가 아직 열려 있지 않다. init_shared(port) 를 먼저 부른다.")
    return _shared


def close_shared():
    global _shared
    if _shared is not None:
        _shared.close()
        _shared = None
