"""ManiBox 런처 — main 파이프라인 / 데모 녹화 / 장치 단위 테스트 를 GUI 하나로.

실행:
    python gui.py          # 어느 OS 든. tkinter(표준 라이브러리) 로 돌아감.

내부 동작:
  - 각 작업 (Run / Record / Test) 은 하위 프로세스로 띄워 stdout 을 실시간 로그에 표시.
  - Mac 에서는 MuJoCo 뷰어가 필요한 작업 (sim/dual 모드 + 뷰어 ON, record.py) 은 mjpython 으로 자동 전환.
  - Linux 는 그대로 python 사용. "뷰어 OFF (--no-view)" 체크 시 Mac 도 python 으로.
"""
from __future__ import annotations

import os
import queue
import shutil
import signal
import subprocess
import sys
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import ttk, scrolledtext, messagebox

ROOT = Path(__file__).resolve().parent
IS_MAC = sys.platform == "darwin"


# ---------- subprocess 실행 + stdout 스트리밍 ----------
class JobRunner:
    """한 번에 한 작업만 돌아가는 백그라운드 subprocess 매니저."""

    def __init__(self, log_q: queue.Queue):
        self.log_q = log_q
        self.proc: subprocess.Popen | None = None
        self.reader: threading.Thread | None = None

    def is_running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def start(self, argv: list[str], cwd: Path = ROOT) -> bool:
        if self.is_running():
            self.log_q.put(("warn", "이미 작업이 돌고 있습니다. 먼저 Stop 하세요.\n"))
            return False
        self.log_q.put(("cmd", f"$ {' '.join(argv)}  (cwd={cwd})\n"))
        try:
            self.proc = subprocess.Popen(
                argv, cwd=str(cwd),
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                bufsize=1, universal_newlines=True,
                env={**os.environ, "PYTHONUNBUFFERED": "1"},
            )
        except FileNotFoundError as e:
            self.log_q.put(("err", f"실행 파일 못 찾음: {e}\n"))
            self.proc = None
            return False
        self.reader = threading.Thread(target=self._pump, daemon=True)
        self.reader.start()
        return True

    def _pump(self):
        assert self.proc and self.proc.stdout
        for line in self.proc.stdout:
            self.log_q.put(("out", line))
        rc = self.proc.wait()
        self.log_q.put(("done", f"[종료 코드 {rc}]\n"))

    def stop(self):
        if not self.is_running():
            self.log_q.put(("warn", "돌고 있는 작업이 없습니다.\n"))
            return
        self.log_q.put(("warn", "중지 신호 전송 ...\n"))
        try:
            if IS_MAC or sys.platform.startswith("linux"):
                self.proc.send_signal(signal.SIGTERM)
            else:
                self.proc.terminate()
            # 3 초 대기 후 안 죽으면 kill
            for _ in range(30):
                if self.proc.poll() is not None:
                    break
                time.sleep(0.1)
            if self.proc.poll() is None:
                self.proc.kill()
        except Exception as e:
            self.log_q.put(("err", f"stop 실패: {e}\n"))


def python_cmd(need_viewer: bool) -> str:
    """Mac + 뷰어 필요 → mjpython. 그 외 → 현재 python."""
    if IS_MAC and need_viewer and shutil.which("mjpython"):
        return "mjpython"
    return sys.executable


# ---------- GUI ----------
class App:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("ManiBox 런처")
        root.geometry("920x680")

        self.log_q: queue.Queue = queue.Queue()
        self.runner = JobRunner(self.log_q)

        # --- 설정 영역 ---
        cfg = ttk.LabelFrame(root, text="설정 (모든 작업 공통)")
        cfg.pack(fill="x", padx=10, pady=(10, 4))

        # ---- 1행: Mode / Speed / 뷰어 OFF ----
        row1 = ttk.Frame(cfg); row1.pack(fill="x", padx=8, pady=(6, 2))
        ttk.Label(row1, text="Mode").pack(side="left")
        # 표시 이름 (GUI) ↔ 내부 값 (main.py --mode)
        self.MODE_MAP = {"sim": "sim", "real": "real", "digital twin": "dual"}
        self.mode = tk.StringVar(value="sim")
        ttk.Combobox(row1, textvariable=self.mode,
                     values=list(self.MODE_MAP.keys()),
                     width=12, state="readonly").pack(side="left", padx=(4, 20))

        ttk.Label(row1, text="Speed").pack(side="left")
        self.speed = tk.StringVar(value="4.0")
        ttk.Entry(row1, textvariable=self.speed, width=6).pack(side="left", padx=(4, 20))

        self.no_view = tk.BooleanVar(value=False)
        ttk.Checkbutton(row1, text="뷰어 OFF (--no-view)",
                        variable=self.no_view).pack(side="left", padx=(0, 4))

        # ---- 2행: Arduino port / CAN ----
        row2 = ttk.Frame(cfg); row2.pack(fill="x", padx=8, pady=(2, 6))
        ttk.Label(row2, text="Arduino port").pack(side="left")
        default_port = "/dev/tty.usbmodem0001" if IS_MAC else "/dev/ttyACM0"
        self.port = tk.StringVar(value=default_port)
        ttk.Entry(row2, textvariable=self.port, width=28).pack(side="left", padx=(4, 20))

        ttk.Label(row2, text="CAN").pack(side="left")
        self.can = tk.StringVar(value="can0")
        ttk.Entry(row2, textvariable=self.can, width=10).pack(side="left", padx=(4, 0))

        # --- 작업 버튼 ---
        acts = ttk.LabelFrame(root, text="작업")
        acts.pack(fill="x", padx=10, pady=4)

        # Row 1 — main pipeline
        r1 = ttk.Frame(acts); r1.pack(fill="x", padx=8, pady=4)
        ttk.Button(r1, text="▶ 적재 시작",
                   command=self.run_pipeline, width=42).pack(side="left")
        ttk.Button(r1, text="■ Stop",
                   command=self.runner.stop, width=18).pack(side="left", padx=8)

        # Row 2 — demo recording
        r2 = ttk.Frame(acts); r2.pack(fill="x", padx=8, pady=4)
        ttk.Button(r2, text="● 데모 녹화",
                   command=self.run_record, width=42).pack(side="left")
        ttk.Label(r2, text="name").pack(side="left", padx=(8, 2))
        self.demo_name = tk.StringVar(value="mv_pos")
        ttk.Combobox(r2, textvariable=self.demo_name,
                     values=["mv_pos", "approach_to_buffer"],
                     width=20, state="readonly").pack(side="left")

        # Row 3 — device test
        r3 = ttk.Frame(acts); r3.pack(fill="x", padx=8, pady=4)
        ttk.Button(r3, text="⚙ 장치 단위 테스트",
                   command=self.run_test, width=42).pack(side="left")
        ttk.Label(r3, text="device").pack(side="left", padx=(8, 2))
        self.test_dev = tk.StringVar(value="suction")
        ttk.Combobox(r3, textvariable=self.test_dev,
                     values=["suction", "rail", "all"],
                     width=10, state="readonly").pack(side="left")

        # Row 4 — env check
        r4 = ttk.Frame(acts); r4.pack(fill="x", padx=8, pady=4)
        ttk.Button(r4, text="✓ 환경 점검  (conda env + 파일 존재 확인)",
                   command=self.run_envcheck, width=42).pack(side="left")
        ttk.Button(r4, text="⌫ 로그 정리",
                   command=self.clear_log, width=18).pack(side="left", padx=8)

        # --- 로그 영역 ---
        logf = ttk.LabelFrame(root, text="로그")
        logf.pack(fill="both", expand=True, padx=10, pady=(4, 10))
        self.log = scrolledtext.ScrolledText(logf, height=22, wrap="word",
                                             font=("Menlo" if IS_MAC else "Monospace", 10))
        self.log.pack(fill="both", expand=True, padx=4, pady=4)
        self.log.tag_config("cmd",  foreground="#005fa3")
        self.log.tag_config("warn", foreground="#aa6600")
        self.log.tag_config("err",  foreground="#b00020")
        self.log.tag_config("done", foreground="#2a7a2a")
        self.log.tag_config("note", foreground="#555555")

        self._emit("note", f"플랫폼: {sys.platform}   python: {sys.executable}\n")
        if IS_MAC:
            if shutil.which("mjpython"):
                self._emit("note", "Mac: mjpython 감지 — 뷰어 작업은 자동으로 mjpython 사용.\n")
            else:
                self._emit("warn", "Mac 인데 mjpython 이 PATH 에 없음. 'conda activate capstone2' 후 실행하세요.\n")

        root.after(100, self._pump_log)

    # ---------- helpers ----------
    def _emit(self, tag: str, text: str):
        self.log.insert("end", text, (tag,))
        self.log.see("end")

    def _pump_log(self):
        try:
            while True:
                tag, text = self.log_q.get_nowait()
                self._emit(tag, text)
        except queue.Empty:
            pass
        self.root.after(100, self._pump_log)

    def clear_log(self):
        self.log.delete("1.0", "end")

    # ---------- actions ----------
    def run_pipeline(self):
        mode = self.MODE_MAP[self.mode.get()]           # "digital twin" → "dual" 등
        need_viewer = (not self.no_view.get()) and (mode in ("sim", "dual"))
        py = python_cmd(need_viewer=need_viewer)
        argv = [py, "manager/main.py", "--mode", mode, "--speed", self.speed.get()]
        if self.no_view.get():
            argv.append("--no-view")
        if mode in ("real", "dual"):
            argv += ["--port", self.port.get(), "--can", self.can.get()]
        self.runner.start(argv)

    def run_record(self):
        """실물 Piper(CAN) 녹화 — raw 터미널 입력이 필요하므로 **새 터미널 창**에서 띄움."""
        py = python_cmd(need_viewer=False)
        cmd_parts = [py, "manager/demos/record.py",
                     "--name", self.demo_name.get(),
                     "--backend", "real",
                     "--can", self.can.get()]
        if self.port.get():
            cmd_parts += ["--port", self.port.get()]
        # 쉘에서 실행할 명령 문자열 (cd + python)
        shell_cmd = f'cd "{ROOT}" && {" ".join(cmd_parts)}; echo; echo "[종료] 아무 키나 누르면 창이 닫힙니다"; read -n 1'

        if IS_MAC:
            # Mac: Terminal.app 에 새 창으로 실행
            apple = (
                'tell application "Terminal"\n'
                '  activate\n'
                f'  do script "{shell_cmd}"\n'
                'end tell'
            )
            try:
                subprocess.Popen(["osascript", "-e", apple])
                self.log_q.put(("note", "Terminal.app 새 창에서 record.py 실행 중. 그 창에 포커스 두고 키 입력하세요.\n"))
            except Exception as e:
                self.log_q.put(("err", f"Terminal 창 열기 실패: {e}\n"))
        elif sys.platform.startswith("linux"):
            # Linux: 사용 가능한 터미널 에뮬레이터 찾아서 실행
            for term in ("gnome-terminal", "konsole", "xfce4-terminal", "xterm"):
                if shutil.which(term):
                    if term == "gnome-terminal":
                        argv = [term, "--", "bash", "-c", shell_cmd]
                    elif term == "konsole":
                        argv = [term, "-e", "bash", "-c", shell_cmd]
                    else:
                        argv = [term, "-e", f"bash -c '{shell_cmd}'"]
                    try:
                        subprocess.Popen(argv)
                        self.log_q.put(("note", f"{term} 새 창에서 record.py 실행 중. 그 창에 포커스 두고 키 입력하세요.\n"))
                        return
                    except Exception as e:
                        self.log_q.put(("warn", f"{term} 실패: {e}\n"))
            self.log_q.put(("err", "터미널 에뮬레이터를 못 찾았습니다. 아래를 직접 실행하세요:\n"))
            self.log_q.put(("note", f"  {' '.join(cmd_parts)}\n"))
        else:
            self.log_q.put(("err", f"지원 안 되는 플랫폼: {sys.platform}\n"))
            self.log_q.put(("note", f"  {' '.join(cmd_parts)}\n"))

    def run_test(self):
        # test.py 는 Arduino 포트 직접 쓴다. 뷰어 불필요.
        py = python_cmd(need_viewer=False)
        argv = [py, "test.py", self.test_dev.get(), "--port", self.port.get()]
        self.runner.start(argv)

    def run_envcheck(self):
        # 별도 프로세스로 띄워 import 확인
        py = python_cmd(need_viewer=False)
        script = (
            "import sys, importlib.util; "
            "print('python:', sys.version.split()[0]); "
            "import os; print('cwd:', os.getcwd()); "
            "mods=['mujoco','numpy','scipy','serial','cv2','piper_sdk']; "
            "[print(f'  {m}:', 'OK' if importlib.util.find_spec(m) else 'MISSING') for m in mods]; "
            "from pathlib import Path; "
            "need=['manager/main.py','manager/demos/record.py','test.py',"
            "'modules/arm/skills.py','modules/rail/rail.py','modules/suction/suction.py',"
            "'modules/camera/camera.py','modules/transform/transform.py',"
            "'modules/arm/model/manibox_modified.xml']; "
            "[print(f'  {p}:', 'OK' if Path(p).exists() else 'MISSING') for p in need]; "
            "demos=['manager/demos/mv_pos.npz','manager/demos/approach_to_buffer.npz']; "
            "[print(f'  {p}:', 'OK' if Path(p).exists() else '없음 (녹화 필요)') for p in demos]"
        )
        argv = [py, "-c", script]
        self.runner.start(argv)


def main():
    # 진단 출력: 창이 안 뜨면 이 로그라도 보고 상태 파악
    print(f"[gui] 시작  python={sys.executable}  platform={sys.platform}", flush=True)
    print(f"[gui] tkinter {tk.TkVersion}  tcl {tk.TclVersion}", flush=True)
    try:
        root = tk.Tk()
    except tk.TclError as e:
        print(f"[gui] tk.Tk() 실패: {e}", flush=True)
        print("[gui] python 이 Tk 지원 없이 빌드됐을 수 있음. "
              "해결: conda install -c conda-forge tk  또는 python -m tkinter 로 확인", flush=True)
        sys.exit(1)

    root.title("ManiBox 런처")
    root.geometry("980x720+120+80")
    root.minsize(760, 560)
    App(root)

    # Mac: 창이 터미널 뒤로 숨거나 Dock 에 안 뜨는 문제 → 앞으로 끌어올리고 포커스 주기
    root.update_idletasks()
    root.deiconify()
    root.lift()
    root.attributes("-topmost", True)
    root.after(500, lambda: root.attributes("-topmost", False))   # 0.5 초 뒤 해제
    root.focus_force()
    if IS_MAC:
        # macOS 전용: 프로세스를 Foreground 로 승격 (Dock 에 아이콘 띄움)
        try:
            root.tk.call("::tk::unsupported::MacWindowStyle", "style", root._w, "document", "closeBox collapseBox resizable")
        except tk.TclError:
            pass

    print("[gui] 창 표시됨. 앞으로 안 뜨면 Cmd-Tab 으로 python 프로세스 전환하거나 Dock 확인.", flush=True)
    try:
        root.mainloop()
    except KeyboardInterrupt:
        print("[gui] 종료 (Ctrl-C)", flush=True)


if __name__ == "__main__":
    main()
