"""ManiBox 런처 — 멀티화면 GUI.

구조 (pipeline/file_call_flow.md ## GUI FLOW 참조):

  [Welcome 화면]
       ├─ 실행 ────────▶ [실행 입력]  ──(다음)──▶ [추론 진행]  ──▶ [적재 실행]
       │                     │        (스킵)           │             (sim 뷰어 자동 띄움)
       │                     └──────────────▶ [적재 실행]
       │
       └─ 개발자 모드 ──▶ [개발자 메뉴]
                              ├─ 알고리즘 개발
                              ├─ arm          (demo 녹화 / 함수 호출 테스트)
                              ├─ suction      (ON/OFF, rail 과 시리얼 연결 공유)
                              ├─ camera       (wrist + overview RGB)
                              └─ rail         (INIT / MOVE / H / SPEED 직접 명령)

실행:
    python gui.py          # tkinter (표준 라이브러리).
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
from tkinter import ttk, scrolledtext, messagebox, filedialog

ROOT = Path(__file__).resolve().parent
IS_MAC = sys.platform == "darwin"
SEQUENCE_DIR = ROOT / "manager" / "sequence"
DEFAULT_INPUT = SEQUENCE_DIR / "input.json"
DEFAULT_MOVES = SEQUENCE_DIR / "moves.json"
PACKING_DIR = ROOT / "modules" / "packing_algorithm"
ICON_PATH = ROOT / "assets" / "icon.png"

# Linux 에는 Arial / Menlo 가 없고, 한글은 CJK 글꼴이 있어야 보인다 (tk 는 xft 빌드 필요).
UI_FONT = "Arial" if IS_MAC else "Noto Sans CJK KR"
MONO_FONT = "Menlo" if IS_MAC else "Noto Sans Mono CJK KR"


def _apply_fonts(root: tk.Tk) -> None:
    """ttk 위젯 기본 글꼴을 한글 되는 글꼴로. tk 가 noxft 빌드면 한글이 깨진다."""
    import tkinter.font as tkfont
    if not IS_MAC and "Noto Sans CJK KR" not in tkfont.families(root):
        print("[gui] Noto Sans CJK KR 없음 — 한글이 깨질 수 있음", flush=True)
    for name in ("TkDefaultFont", "TkTextFont", "TkMenuFont", "TkHeadingFont",
                 "TkCaptionFont", "TkSmallCaptionFont", "TkIconFont", "TkTooltipFont"):
        try:
            tkfont.nametofont(name, root).configure(family=UI_FONT)
        except tk.TclError:
            pass
    try:
        tkfont.nametofont("TkFixedFont", root).configure(family=MONO_FONT)
    except tk.TclError:
        pass


def _apply_icon(root: tk.Tk) -> None:
    """창 아이콘(assets/icon.png) 적용. Mac Dock 아이콘은 .app 번들 밖에서는 바뀌지 않음."""
    if not ICON_PATH.exists():
        return
    try:
        img = tk.PhotoImage(file=str(ICON_PATH))
        root.iconphoto(True, img)
        root._icon_img = img  # GC 방지
    except tk.TclError as e:
        print(f"[gui] 아이콘 로드 실패: {e}", flush=True)

# ============================================================
#  subprocess 실행 + stdout 스트리밍
# ============================================================
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
            self.proc.send_signal(signal.SIGTERM)
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


def _front(root: tk.Tk):
    """창을 앞으로 끌어올리고 포커스 주기 — Mac 에서 숨는 문제 대응."""
    root.update_idletasks()
    root.deiconify()
    root.lift()
    root.attributes("-topmost", True)
    root.after(500, lambda: root.attributes("-topmost", False))
    root.focus_force()
    if IS_MAC:
        try:
            root.tk.call("::tk::unsupported::MacWindowStyle", "style", root._w,
                         "document", "closeBox collapseBox resizable")
        except tk.TclError:
            pass


# ============================================================
#  뷰어 자동 배치 (Mac: osascript, Linux: wmctrl)
# ============================================================
def dock_launcher_and_viewer(root: tk.Tk, log_q: queue.Queue):
    """런처 왼쪽 반, MuJoCo 뷰어 오른쪽 반."""
    sw = root.winfo_screenwidth()
    sh = root.winfo_screenheight()
    half = sw // 2
    root.geometry(f"{half - 20}x{sh - 100}+10+20")
    root.update_idletasks()
    right_x = half + 10
    right_w = sw - half - 30
    right_h = sh - 100
    log_q.put(("note",
               f"런처: 왼쪽 반 ({half-20}×{sh-100}).  "
               f"뷰어: 오른쪽 ({right_x},40) 에 {right_w}×{right_h} 배치 시도.\n"))
    state = {"done": False, "ticks": 0}
    def try_position():
        if state["done"]:
            return
        state["ticks"] += 1
        placed = False
        if IS_MAC:
            apple = (
                'set moved to "none"\n'
                'tell application "System Events"\n'
                '  repeat with p in (every process)\n'
                '    try\n'
                '      repeat with w in (every window of p)\n'
                '        try\n'
                '          if name of w contains "MuJoCo" then\n'
                f'            set position of w to {{{right_x}, 40}}\n'
                f'            set size of w to {{{right_w}, {right_h}}}\n'
                '            set moved to (name of p as string)\n'
                '          end if\n'
                '        end try\n'
                '      end repeat\n'
                '    end try\n'
                '  end repeat\n'
                'end tell\n'
                'return moved'
            )
            try:
                r = subprocess.run(["osascript", "-e", apple],
                                   capture_output=True, text=True, timeout=2)
                out = r.stdout.strip()
                if out and out != "none":
                    placed = True
                    log_q.put(("note", f"[viewer] {out} 의 MuJoCo 창을 오른쪽으로 이동\n"))
            except Exception:
                pass
        elif sys.platform.startswith("linux") and shutil.which("wmctrl"):
            try:
                r = subprocess.run(["wmctrl", "-l"], capture_output=True, text=True, timeout=2)
                if "MuJoCo" in r.stdout:
                    subprocess.run(["wmctrl", "-r", "MuJoCo", "-e",
                                    f"0,{right_x},40,{right_w},{right_h}"], timeout=2)
                    placed = True
                    log_q.put(("note", "[viewer] MuJoCo 창을 오른쪽으로 이동\n"))
            except Exception:
                pass
        if placed:
            state["done"] = True
            return
        if state["ticks"] >= 30:
            log_q.put(("warn",
                       "MuJoCo 창을 못 찾음. 수동으로 오른쪽 반에 드래그하세요.\n"))
            return
        root.after(500, try_position)
    root.after(300, try_position)


# ============================================================
#  공유 시리얼 연결 (레일 + 흡착이 한 아두이노)
# ============================================================
DEFAULT_PORT = "/dev/tty.usbmodem0001" if IS_MAC else "/dev/ttyACM0"


class SerialLink:
    """GUI 전체에서 하나만 두는 아두이노 연결. 화면을 옮겨도 유지된다.

    포트를 열 때마다 아두이노가 리셋되어 레일 INIT / SPEED / ACCEL / 흡착 상태가 날아가므로,
    버튼마다 서브프로세스로 포트를 여는 대신 여기서 한 번 열어 두고 계속 쓴다.
    수신 줄은 지금 보이는 화면의 로그(log_q) 로 보낸다.
    """

    def __init__(self):
        self.ser = None
        self.port = DEFAULT_PORT
        self.log_q: queue.Queue | None = None
        self._stop = threading.Event()
        self._connecting = False

    @property
    def connected(self) -> bool:
        return self.ser is not None

    def attach(self, log_q: queue.Queue):
        self.log_q = log_q

    def _log(self, tag: str, text: str):
        if self.log_q is not None:
            self.log_q.put((tag, text))

    def connect(self, port: str, on_result):
        """백그라운드로 포트 열기 + 리셋 대기. 끝나면 on_result(ok) 를 Tk 스레드 밖에서 호출."""
        if self.connected or self._connecting:
            return
        try:
            import serial
        except ImportError:
            self._log("err", "pyserial 없음 — capstone2 conda 환경에서 gui.py 를 실행하세요.\n")
            on_result(False)
            return
        self._connecting = True

        def worker():
            try:
                ser = serial.Serial(port, 115200, timeout=0.2)
            except Exception as e:
                self._log("err", f"포트 열기 실패: {e}\n")
                self._connecting = False
                on_result(False)
                return
            time.sleep(2.0)                     # 포트 열면 아두이노가 리셋됨
            ser.reset_input_buffer()            # 부팅 배너 버림
            self.ser, self.port = ser, port
            self._stop.clear()
            threading.Thread(target=self._reader, daemon=True).start()
            self._connecting = False
            self._log("done", f"[연결됨] {port} — 아두이노 리셋됨: 레일은 INIT 부터.\n")
            on_result(True)

        threading.Thread(target=worker, daemon=True).start()

    def disconnect(self):
        self._stop.set()
        ser, self.ser = self.ser, None
        if ser is not None:
            try:
                ser.close()
            except Exception:
                pass
            self._log("note", "[연결 끊김]\n")

    def send(self, cmd: str) -> bool:
        if self.ser is None:
            self._log("warn", "먼저 '연결' 하세요.\n")
            return False
        self._log("cmd", f"> {cmd}\n")
        try:
            self.ser.write((cmd + "\n").encode("ascii"))
            self.ser.flush()
            return True
        except Exception as e:
            self._log("err", f"송신 실패: {e}\n")
            return False

    def _reader(self):
        while not self._stop.is_set():
            ser = self.ser
            if ser is None:
                return
            try:
                line = ser.readline().decode("ascii", errors="replace").strip()
            except Exception as e:
                if not self._stop.is_set():
                    self._log("err", f"수신 오류: {e}\n")
                    self.ser = None
                return
            if line:
                self._log("err" if line.startswith("ERROR") else "out", line + "\n")


LINK = SerialLink()


def release_link(log_q: queue.Queue):
    """서브프로세스가 같은 포트를 열어야 할 때 GUI 연결을 먼저 놓는다."""
    if LINK.connected:
        LINK.disconnect()
        log_q.put(("note", "GUI 시리얼 연결을 해제하고 실행합니다.\n"))


# ============================================================
#  Screen 기반 전환
# ============================================================
class Screen:
    """모든 화면의 공통 베이스. 모든 위젯은 self.container 아래에 두고,
    destroy 에서 container 하나 삭제로 전부 정리."""

    def __init__(self, root: tk.Tk, nav: "Navigator", **kwargs):
        self.root = root
        self.nav = nav
        self.kwargs = kwargs
        self.log_q: queue.Queue = queue.Queue()
        self.runner = JobRunner(self.log_q)
        self._stopped = False
        self._pump_log_scheduled = False
        # 화면 전체를 감싸는 컨테이너
        self.container = ttk.Frame(root)
        self.container.pack(expand=True, fill="both")

    def build(self):
        raise NotImplementedError

    def destroy(self):
        self._stopped = True
        if self.runner.is_running():
            try:
                self.runner.stop()
            except Exception:
                pass
        try:
            self.container.destroy()
        except Exception:
            pass

    # ---------- 공통: 상단 네비게이션 바 ----------
    def header(self, title: str, back_to=None):
        bar = ttk.Frame(self.container)
        bar.pack(fill="x", padx=10, pady=(10, 4))
        if back_to is not None:
            ttk.Button(bar, text="← 뒤로", width=10,
                       command=lambda: self.nav.go(back_to)).pack(side="left")
        ttk.Label(bar, text=title, font=(UI_FONT, 16, "bold")).pack(side="left", padx=20)

    # ---------- 공통: 아두이노 연결 바 (LINK 공유) ----------
    def make_conn_bar(self):
        cfg = ttk.LabelFrame(self.container, text="아두이노 연결 (레일 + 흡착 공유, 화면 이동해도 유지)")
        cfg.pack(fill="x", padx=10, pady=4)
        r = ttk.Frame(cfg); r.pack(fill="x", padx=8, pady=6)
        ttk.Label(r, text="Arduino port").pack(side="left")
        self.port = tk.StringVar(value=LINK.port)
        ttk.Entry(r, textvariable=self.port, width=28).pack(side="left", padx=(4, 16))
        self.btn_conn = ttk.Button(r, width=10, command=self._toggle_conn)
        self.btn_conn.pack(side="left", padx=4)
        self.conn_status = tk.StringVar()
        ttk.Label(r, textvariable=self.conn_status, foreground="#666").pack(side="left", padx=8)
        self._refresh_conn()

    def _refresh_conn(self):
        try:
            if LINK.connected:
                self.btn_conn.config(text="연결 끊기", state="normal")
                self.conn_status.set(f"연결됨: {LINK.port}")
            else:
                self.btn_conn.config(text="연결", state="normal")
                self.conn_status.set("연결 안 됨")
        except tk.TclError:
            pass

    def _toggle_conn(self):
        if LINK.connected:
            LINK.disconnect()
            self._refresh_conn()
            return
        self.btn_conn.config(state="disabled")
        self.conn_status.set("연결 중… (아두이노 리셋 대기 2 s)")
        LINK.connect(self.port.get(),
                     lambda ok: self.root.after(0, self._refresh_conn))

    # ---------- 공통: 로그 영역 ----------
    def make_log(self, parent=None, height=12) -> scrolledtext.ScrolledText:
        parent = parent if parent is not None else self.container
        logf = ttk.LabelFrame(parent, text="로그")
        logf.pack(fill="both", expand=True, padx=10, pady=10)
        log = scrolledtext.ScrolledText(logf, height=height, wrap="word",
                                         font=(MONO_FONT, 10))
        log.pack(fill="both", expand=True, padx=4, pady=4)
        for tag, color in (("cmd", "#005fa3"), ("warn", "#aa6600"),
                           ("err", "#b00020"), ("done", "#2a7a2a"), ("note", "#555555")):
            log.tag_config(tag, foreground=color)
        self.log = log
        LINK.attach(self.log_q)                # 아두이노 수신 줄은 지금 화면 로그로
        self._start_log_pump()
        return log

    def _start_log_pump(self):
        if self._pump_log_scheduled:
            return
        self._pump_log_scheduled = True
        self._pump_log()

    def _pump_log(self):
        if self._stopped:
            return
        try:
            while True:
                tag, text = self.log_q.get_nowait()
                if hasattr(self, "log"):
                    try:
                        self.log.insert("end", text, (tag,))
                        self.log.see("end")
                    except tk.TclError:
                        return
                if tag == "done" and hasattr(self, "on_done"):
                    self.on_done()
        except queue.Empty:
            pass
        if not self._stopped:
            self.root.after(100, self._pump_log)


class Navigator:
    """스크린 전환 — 현재 화면 destroy 후 새 화면 build."""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.current: Screen | None = None

    def go(self, screen_cls, **kwargs):
        if self.current:
            self.current.destroy()
        self.current = screen_cls(self.root, self, **kwargs)
        self.current.build()


# ============================================================
#  화면들
# ============================================================
class WelcomeScreen(Screen):
    """메인화면 — 실행 / 개발자 모드 선택."""

    def build(self):
        self.root.title("ManiBox")

        wrap = (ttk.Frame(self.container))
        wrap.pack(expand=True, fill="both", padx=40, pady=40)

        ttk.Label(wrap, text="ManiBox", font=(UI_FONT, 36, "bold")).pack(pady=(40, 10))
        ttk.Label(wrap, text="Automated Truck Packing",
                  font=(UI_FONT, 14)).pack(pady=(0, 40))

        btns = ttk.Frame(wrap); btns.pack(pady=20)
        ttk.Button(btns, text="▶  실행", width=18,
                   command=lambda: self.nav.go(ExecutionInputScreen)).pack(side="left", padx=10, ipady=10)
        ttk.Button(btns, text="⚙  개발자 모드", width=18,
                   command=lambda: self.nav.go(DeveloperMenuScreen)).pack(side="left", padx=10, ipady=10)


# ---------- 실행 ----------
class ExecutionInputScreen(Screen):
    """실행: 입력/출력 파일 경로 + 추론 스킵."""

    def build(self):
        self.root.title("실행 — 입력 설정")
        self.header("실행 — 입력 설정", back_to=WelcomeScreen)

        box = (ttk.LabelFrame(self.container, text="파일 경로"))
        box.pack(fill="x", padx=10, pady=4)

        r1 = ttk.Frame(box); r1.pack(fill="x", padx=8, pady=6)
        ttk.Label(r1, text="입력:", width=8).pack(side="left")
        self.in_var = tk.StringVar(value=str(DEFAULT_INPUT))
        ttk.Entry(r1, textvariable=self.in_var).pack(side="left", fill="x", expand=True, padx=(4, 4))
        ttk.Button(r1, text="찾기…", command=self._pick_input, width=8).pack(side="left")

        r2 = ttk.Frame(box); r2.pack(fill="x", padx=8, pady=6)
        ttk.Label(r2, text="출력:", width=8).pack(side="left")
        self.out_var = tk.StringVar(value=str(DEFAULT_MOVES))
        ttk.Entry(r2, textvariable=self.out_var).pack(side="left", fill="x", expand=True, padx=(4, 4))
        ttk.Button(r2, text="찾기…", command=self._pick_output, width=8).pack(side="left")

        opts = (ttk.LabelFrame(self.container, text="옵션"))
        opts.pack(fill="x", padx=10, pady=4)
        self.skip = tk.BooleanVar(value=False)
        ttk.Checkbutton(opts, text="추론 스킵 (기존 출력 파일을 그대로 사용)",
                        variable=self.skip).pack(anchor="w", padx=10, pady=8)
        self.motion = tk.BooleanVar(value=False)
        ttk.Checkbutton(opts, text="모션 검사 (--motion, Piper reach 검증)",
                        variable=self.motion).pack(anchor="w", padx=10, pady=4)

        btns = (ttk.Frame(self.container)); btns.pack(fill="x", padx=10, pady=20)
        ttk.Button(btns, text="다음 →", width=14, command=self._next).pack(side="right", padx=8)
        ttk.Button(btns, text="취소", width=10,
                   command=lambda: self.nav.go(WelcomeScreen)).pack(side="right")

    def _pick_input(self):
        p = filedialog.askopenfilename(initialdir=str(SEQUENCE_DIR),
                                        title="입력 시퀀스 선택",
                                        filetypes=[("JSON/CSV", "*.json *.csv")])
        if p: self.in_var.set(p)

    def _pick_output(self):
        p = filedialog.asksaveasfilename(initialdir=str(SEQUENCE_DIR),
                                         title="출력 moves 파일",
                                         defaultextension=".json")
        if p: self.out_var.set(p)

    def _next(self):
        in_path = Path(self.in_var.get())
        out_path = Path(self.out_var.get())
        if self.skip.get():
            if not out_path.exists():
                messagebox.showerror("출력 파일 없음",
                                     f"스킵하려면 기존 moves 파일이 있어야 합니다: {out_path}")
                return
            self.nav.go(ExecutionRunScreen, moves_path=out_path)
            return
        if not in_path.exists():
            messagebox.showerror("입력 파일 없음", f"{in_path}")
            return
        self.nav.go(ExecutionInferenceScreen,
                    in_path=in_path, out_path=out_path, motion=self.motion.get())


class ExecutionInferenceScreen(Screen):
    """추론 진행 — 프로그레스 표시, 완료 시 ExecutionRunScreen 으로."""

    def build(self):
        self.root.title("실행 — 추론 진행")
        self.header("적재 알고리즘 추론 중…", back_to=None)

        info = (ttk.Frame(self.container)); info.pack(fill="x", padx=20, pady=20)
        self.status = tk.StringVar(value="추론 시작 중…")
        ttk.Label(info, textvariable=self.status, font=(UI_FONT, 13)).pack()
        self.pct = tk.StringVar(value="0%")
        ttk.Label(info, textvariable=self.pct, font=(UI_FONT, 24, "bold")).pack(pady=10)

        self.bar = ttk.Progressbar(self.root, mode="indeterminate", length=600)
        self.bar.pack(padx=20, pady=10)
        self.bar.start(15)

        self.make_log(height=12)

        btns = (ttk.Frame(self.container)); btns.pack(fill="x", padx=10, pady=8)
        self.btn_next = ttk.Button(btns, text="다음 →", width=14, state="disabled",
                                    command=self._to_run)
        self.btn_next.pack(side="right", padx=8)
        ttk.Button(btns, text="■ Stop", width=10,
                   command=self.runner.stop).pack(side="right")

        out_path = self.kwargs["out_path"]
        self._out_path = out_path
        argv = [sys.executable, "plan_moves.py", "--mode", "big",
                "--manifest", str(self.kwargs["in_path"]),
                "--out", str(out_path),
                "--plans", "16", "--workers", "16"]
        if self.kwargs.get("motion"):
            argv.append("--motion")
        self.runner.start(argv, cwd=PACKING_DIR)

    def on_done(self):
        self.bar.stop()
        self.bar.config(mode="determinate", value=100)
        self.pct.set("100%")
        if self._out_path.exists():
            self.status.set("추론 완료 — '다음' 으로 적재 실행으로 이동")
            self.btn_next.config(state="normal")
        else:
            self.status.set("실패 — 출력 파일이 생성되지 않음")

    def _to_run(self):
        self.nav.go(ExecutionRunScreen, moves_path=self._out_path)


class ExecutionRunScreen(Screen):
    """적재 실행 — sim 뷰어 자동 띄움 + 로봇 동기화 (dual 모드)."""

    def build(self):
        self.root.title("실행 — 적재")
        self.header("적재 실행", back_to=WelcomeScreen)

        info = (ttk.LabelFrame(self.container, text="설정"))
        info.pack(fill="x", padx=10, pady=4)
        moves = self.kwargs.get("moves_path", DEFAULT_MOVES)
        ttk.Label(info, text=f"moves: {moves}").pack(anchor="w", padx=10, pady=4)
        ttk.Label(info, text="모드: 디지털 트윈 (sim + 실물 동기화)").pack(anchor="w", padx=10, pady=4)

        hw = (ttk.LabelFrame(self.container, text="하드웨어 포트"))
        hw.pack(fill="x", padx=10, pady=4)
        row = ttk.Frame(hw); row.pack(fill="x", padx=8, pady=4)
        ttk.Label(row, text="Arduino port").pack(side="left")
        self.port = tk.StringVar(value="/dev/tty.usbmodem0001" if IS_MAC else "/dev/ttyACM0")
        ttk.Entry(row, textvariable=self.port, width=28).pack(side="left", padx=(4, 16))
        ttk.Label(row, text="CAN").pack(side="left")
        self.can = tk.StringVar(value="can0")
        ttk.Entry(row, textvariable=self.can, width=10).pack(side="left", padx=(4, 16))
        ttk.Label(row, text="Speed").pack(side="left")
        self.speed = tk.StringVar(value="1.0")
        ttk.Entry(row, textvariable=self.speed, width=6).pack(side="left", padx=(4, 0))

        btns = (ttk.Frame(self.container)); btns.pack(fill="x", padx=10, pady=8)
        ttk.Button(btns, text="▶ 적재 실행", width=20, command=self._run).pack(side="left", padx=4)
        ttk.Button(btns, text="■ Stop", width=10, command=self.runner.stop).pack(side="left", padx=4)

        self.make_log(height=18)
        self._moves = moves

    def _run(self):
        if self.runner.is_running():
            self.log_q.put(("note", "이전 작업 중지 후 새로 시작 ...\n"))
            self.runner.stop()
        release_link(self.log_q)
        py = python_cmd(need_viewer=True)
        argv = [py, "manager/main.py", "--mode", "dual",
                "--moves", str(self._moves),
                "--speed", self.speed.get(),
                "--port", self.port.get(),
                "--can", self.can.get()]
        ok = self.runner.start(argv)
        if ok:
            dock_launcher_and_viewer(self.root, self.log_q)


# ---------- 개발자 모드 ----------
class DeveloperMenuScreen(Screen):
    """개발자 모드 — 5 개 서브 메뉴."""

    def build(self):
        self.root.title("개발자 모드")
        self.header("개발자 모드", back_to=WelcomeScreen)

        grid = (ttk.Frame(self.container))
        grid.pack(expand=True, fill="both", padx=40, pady=20)

        items = [
            ("Algorithm Test",    AlgorithmDevScreen,  "입력 → 추론 → 완료 알림"),
            ("Arm",             ArmDevScreen,         "demo 녹화 + 함수 호출 테스트"),
            ("Suction",         SuctionDevScreen,     "ON / OFF 토글 테스트"),
            ("Camera",          CameraDevScreen,      "wrist + overview RGB 표시"),
            ("Rail",            RailDevScreen,        "INIT / MOVE / H / SPEED 직접 명령"),
        ]
        for i, (label, cls, desc) in enumerate(items):
            fr = ttk.Frame(grid); fr.pack(fill="x", pady=4)
            ttk.Button(fr, text=label, width=22,
                       command=lambda c=cls: self.nav.go(c)).pack(side="left")
            ttk.Label(fr, text=desc, foreground="#666").pack(side="left", padx=12)


class AlgorithmDevScreen(Screen):
    """알고리즘 개발 — 입력/출력 경로 + 추론 (완료 시 알림창)."""

    def build(self):
        self.root.title("개발자 — 알고리즘")
        self.header("알고리즘 개발", back_to=DeveloperMenuScreen)

        box = (ttk.LabelFrame(self.container, text="경로"))
        box.pack(fill="x", padx=10, pady=4)
        r1 = ttk.Frame(box); r1.pack(fill="x", padx=8, pady=6)
        ttk.Label(r1, text="입력:", width=8).pack(side="left")
        self.in_var = tk.StringVar(value=str(DEFAULT_INPUT))
        ttk.Entry(r1, textvariable=self.in_var).pack(side="left", fill="x", expand=True, padx=(4, 4))
        ttk.Button(r1, text="찾기…", command=self._pick_in, width=8).pack(side="left")
        r2 = ttk.Frame(box); r2.pack(fill="x", padx=8, pady=6)
        ttk.Label(r2, text="출력:", width=8).pack(side="left")
        self.out_var = tk.StringVar(value=str(DEFAULT_MOVES))
        ttk.Entry(r2, textvariable=self.out_var).pack(side="left", fill="x", expand=True, padx=(4, 4))

        opts = (ttk.Frame(self.container)); opts.pack(fill="x", padx=10, pady=4)
        self.skip = tk.BooleanVar(value=False)
        ttk.Checkbutton(opts, text="추론 스킵 (기존 출력 그대로)",
                        variable=self.skip).pack(side="left", padx=8)
        self.motion = tk.BooleanVar(value=False)
        ttk.Checkbutton(opts, text="--motion (Piper reach 검증)",
                        variable=self.motion).pack(side="left", padx=8)

        btns = (ttk.Frame(self.container)); btns.pack(fill="x", padx=10, pady=8)
        ttk.Button(btns, text="▶ 추론", width=12, command=self._run).pack(side="left", padx=4)
        ttk.Button(btns, text="■ Stop", width=10, command=self.runner.stop).pack(side="left", padx=4)

        self.make_log(height=16)

    def _pick_in(self):
        p = filedialog.askopenfilename(initialdir=str(SEQUENCE_DIR))
        if p: self.in_var.set(p)

    def _run(self):
        out_path = Path(self.out_var.get())
        if self.skip.get():
            if not out_path.exists():
                messagebox.showerror("출력 없음", f"스킵하려면 파일이 있어야: {out_path}")
                return
            self._notify_done(out_path)
            return
        in_path = Path(self.in_var.get())
        if not in_path.exists():
            messagebox.showerror("입력 없음", f"{in_path}")
            return
        self._pending_out = out_path
        argv = [sys.executable, "plan_moves.py", "--mode", "big",
                "--manifest", str(in_path), "--out", str(out_path),
                "--plans", "16", "--workers", "16"]
        if self.motion.get():
            argv.append("--motion")
        self.runner.start(argv, cwd=PACKING_DIR)

    def on_done(self):
        if hasattr(self, "_pending_out") and self._pending_out.exists():
            self._notify_done(self._pending_out)

    def _notify_done(self, out_path: Path):
        messagebox.showinfo("추론 완료", f"출력:\n{out_path}")


class ArmDevScreen(Screen):
    """arm 개발 — demo 녹화 + 함수 호출 테스트."""

    def build(self):
        self.root.title("개발자 — arm")
        self.header("arm 개발", back_to=DeveloperMenuScreen)

        rec = (ttk.LabelFrame(self.container, text="데모 녹화 (실물 Piper CAN)"))
        rec.pack(fill="x", padx=10, pady=4)
        r = ttk.Frame(rec); r.pack(fill="x", padx=8, pady=6)
        ttk.Label(r, text="name").pack(side="left")
        self.name = tk.StringVar(value="mv_pos")
        ttk.Combobox(r, textvariable=self.name, values=["mv_pos", "approach_to_buffer"],
                     width=20, state="readonly").pack(side="left", padx=(4, 16))
        ttk.Label(r, text="CAN").pack(side="left")
        self.can = tk.StringVar(value="can0")
        ttk.Entry(r, textvariable=self.can, width=8).pack(side="left", padx=(4, 16))
        ttk.Label(r, text="port (선택)").pack(side="left")
        self.port = tk.StringVar(value="")
        ttk.Entry(r, textvariable=self.port, width=24).pack(side="left", padx=(4, 16))
        ttk.Button(r, text="● 녹화 시작", command=self._record).pack(side="left")

        fn = (ttk.LabelFrame(self.container, text="함수 호출 테스트"))
        fn.pack(fill="x", padx=10, pady=4)
        row = ttk.Frame(fn); row.pack(fill="x", padx=8, pady=6)
        for label, func in (("reset", "reset"),
                            ("mv_pos", "mv_pos"),
                            ("approach_to_buffer", "approach_to_buffer"),
                            ("alignment(pack)", "alignment_pack"),
                            ("alignment(unpack)", "alignment_unpack")):
            ttk.Button(row, text=label, width=18,
                       command=lambda f=func: self._call(f)).pack(side="left", padx=4, pady=2)

        # 실물 연동 체크 — 체크 해제 시 sim 만, 체크 시 Robot.dual (실물+sim)
        row2 = ttk.Frame(fn); row2.pack(fill="x", padx=8, pady=(0, 6))
        self.use_real_call = tk.BooleanVar(value=False)
        ttk.Checkbutton(row2,
                        text="실물 연동 (체크 해제 시 sim 만)",
                        variable=self.use_real_call).pack(side="left")

        self.make_log(height=14)

    def _record(self):
        cmd = [python_cmd(need_viewer=False), "manager/demos/record.py",
               "--name", self.name.get(), "--backend", "real",
               "--can", self.can.get()]
        if self.port.get():
            release_link(self.log_q)
            cmd += ["--port", self.port.get()]
        shell = f'cd "{ROOT}" && {" ".join(cmd)}; echo; echo "[종료]"; read -n 1'
        if IS_MAC:
            subprocess.Popen(["osascript", "-e",
                              f'tell application "Terminal"\n  activate\n  do script "{shell}"\nend tell'])
            self.log_q.put(("note", "새 Terminal 창에서 녹화 중.\n"))
        elif sys.platform.startswith("linux"):
            for term in ("gnome-terminal", "konsole", "xterm"):
                if shutil.which(term):
                    argv = [term, "--", "bash", "-c", shell] if term == "gnome-terminal" else [term, "-e", f"bash -c '{shell}'"]
                    subprocess.Popen(argv)
                    self.log_q.put(("note", f"{term} 새 창에서 녹화 중.\n"))
                    return
            self.log_q.put(("err", "터미널을 못 찾음.\n"))

    def _call(self, func: str):
        """선택한 스킬 함수를 1 회 호출. 체크박스 OFF → sim 전용(뷰어 포함), ON → Robot.dual (실물+sim)."""
        if "alignment" in func:
            call = f'skills.alignment(robot, "{func.split("_")[1]}")'
        else:
            call = f"skills.{func}(robot)"
        if self.use_real_call.get():
            release_link(self.log_q)
            port = self.port.get() or "/dev/ttyACM0"
            robot_init = (
                f"robot = Robot.dual(s.m, s.d, "
                f"arduino_port='{port}', can='{self.can.get()}', on_step=on_step)"
            )
        else:
            robot_init = "robot = Robot.sim(s.m, s.d, on_step=on_step)"
        script = (
            "import sys, time\n"
            "sys.path.insert(0, 'modules/arm')\n"
            "sys.path.insert(0, 'manager')\n"
            "import mujoco\n"
            "import mujoco.viewer\n"
            "from stacking import Stacker, READY, set_view\n"
            "from robot import Robot\n"
            "import skills\n"
            "s = Stacker([1])\n"
            "s.q = READY.copy(); s.d.qpos[:] = s.q\n"
            "mujoco.mj_forward(s.m, s.d)\n"
            "with mujoco.viewer.launch_passive(s.m, s.d, show_left_ui=False, show_right_ui=True) as v:\n"
            "    set_view(v.cam)\n"
            "    def on_step():\n"
            "        mujoco.mj_forward(s.m, s.d)\n"
            "        if v.is_running():\n"
            "            v.sync()\n"
            f"    {robot_init}\n"
            "    try:\n"
            f"        {call}\n"
            "        print('OK')\n"
            "    except Exception as e:\n"
            "        print('[예외]', type(e).__name__, e)\n"
            "    # 함수 종료 후에도 뷰어 유지 — 창 닫으면 종료\n"
            "    print('뷰어 창을 닫으면 종료됩니다.')\n"
            "    while v.is_running():\n"
            "        time.sleep(0.1)\n"
        )
        # 뷰어 필요 → Mac 에서는 mjpython 자동 사용
        self.runner.start([python_cmd(need_viewer=True), "-c", script])


class SuctionDevScreen(Screen):
    """suction 개발 — ON/OFF. Rail 화면과 같은 시리얼 연결(LINK) 을 쓴다.
    grip() / un_grip() 은 RealSuction 에서 on() / off() 와 같은 "ON" / "OFF" 전송이라 따로 두지 않는다."""

    def build(self):
        self.root.title("개발자 — suction")
        self.header("suction 테스트", back_to=DeveloperMenuScreen)
        self.make_conn_bar()

        btns = (ttk.LabelFrame(self.container, text="흡착 (= grip() / un_grip())"))
        btns.pack(fill="x", padx=10, pady=4)
        row = ttk.Frame(btns); row.pack(fill="x", padx=8, pady=10)
        ttk.Button(row, text="ON  (grip)", width=14,
                   command=lambda: LINK.send("ON")).pack(side="left", padx=8)
        ttk.Button(row, text="OFF  (un_grip)", width=14,
                   command=lambda: LINK.send("OFF")).pack(side="left", padx=8)
        ttk.Button(row, text="STATUS", width=10,
                   command=lambda: LINK.send("STATUS")).pack(side="left", padx=8)

        self.make_log(height=12)


class CameraDevScreen(Screen):
    """camera 개발 — wrist + overview RGB 표시."""

    def build(self):
        self.root.title("개발자 — camera")
        self.header("camera (wrist + overview)", back_to=DeveloperMenuScreen)

        info = (ttk.LabelFrame(self.container, text="설정"))
        info.pack(fill="x", padx=10, pady=4)
        r = ttk.Frame(info); r.pack(fill="x", padx=8, pady=6)
        ttk.Label(r, text="wrist 장치 index").pack(side="left")
        self.wrist = tk.StringVar(value="0")
        ttk.Entry(r, textvariable=self.wrist, width=6).pack(side="left", padx=(4, 16))
        ttk.Label(r, text="overview 장치 index").pack(side="left")
        self.over = tk.StringVar(value="1")
        ttk.Entry(r, textvariable=self.over, width=6).pack(side="left", padx=(4, 16))
        ttk.Button(r, text="▶ 양쪽 RGB 표시", command=self._run).pack(side="left", padx=4)
        ttk.Button(r, text="■ Stop", command=self.runner.stop).pack(side="left", padx=4)

        note = (
            "wrist cam (lerobot SO101 arm 장착) 와 overview cam (Intel D435) 의 RGB 를 OpenCV 창 2 개로 띄웁니다.\n"
            "장치 index 는 ls /dev/video* (Linux) 또는 system_profiler SPCameraDataType (Mac) 로 확인."
        )
        (ttk.Label(self.container, text=note, justify="left", foreground="#555")).pack(anchor="w", padx=14, pady=4)

        self.make_log(height=10)

    def _run(self):
        """OpenCV 로 두 카메라 RGB 를 창 두 개에 띄우는 서브프로세스."""
        script = (
            "import cv2, sys\n"
            f"w = cv2.VideoCapture({self.wrist.get()}); "
            f"o = cv2.VideoCapture({self.over.get()})\n"
            "if not w.isOpened(): print('wrist 열기 실패'); sys.exit(1)\n"
            "if not o.isOpened(): print('overview 열기 실패'); sys.exit(1)\n"
            "print('창에서 q 또는 ESC 로 종료')\n"
            "try:\n"
            "  while True:\n"
            "    r1, f1 = w.read(); r2, f2 = o.read()\n"
            "    if r1: cv2.imshow('wrist', f1)\n"
            "    if r2: cv2.imshow('overview', f2)\n"
            "    k = cv2.waitKey(1) & 0xff\n"
            "    if k in (27, ord('q')): break\n"
            "finally:\n"
            "  w.release(); o.release(); cv2.destroyAllWindows()\n"
        )
        self.runner.start([sys.executable, "-c", script])


class RailDevScreen(Screen):
    """rail 개발 — 아두이노에 직접 INIT / MOVE / H / SPEED / ACCEL. 연결은 LINK (Suction 화면과 공유)."""

    SPEED_MAX_MM_S = 400.0      # 펌웨어: SPEED * stepsPerMM(2000/150) <= MAX_SPEED(5400)
    ACCEL_MAX_MM_S2 = 1500.0    # 펌웨어: ACCEL * stepsPerMM <= MAX_ACCEL(20000)

    def build(self):
        self.root.title("개발자 — rail")
        self.header("rail 테스트", back_to=DeveloperMenuScreen)
        self.make_conn_bar()

        ctl = (ttk.LabelFrame(self.container, text="레일 명령 (아두이노 좌표 mm, 0 = 모터 반대쪽 끝)"))
        ctl.pack(fill="x", padx=10, pady=4)

        r1 = ttk.Frame(ctl); r1.pack(fill="x", padx=8, pady=6)
        ttk.Button(r1, text="INIT (현재 위치 = 0)", width=20,
                   command=lambda: self._send("INIT")).pack(side="left", padx=4)
        ttk.Button(r1, text="H (홈 복귀)", width=14,
                   command=lambda: self._send("H")).pack(side="left", padx=4)
        ttk.Button(r1, text="■ STOP", width=10,
                   command=lambda: self._send("STOP")).pack(side="left", padx=4)
        ttk.Button(r1, text="POS", width=8,
                   command=lambda: self._send("POS")).pack(side="left", padx=4)

        r2 = ttk.Frame(ctl); r2.pack(fill="x", padx=8, pady=6)
        ttk.Label(r2, text="MOVE (mm, 0~1000)").pack(side="left")
        self.move_mm = tk.StringVar(value="100")
        e_move = ttk.Entry(r2, textvariable=self.move_mm, width=10)
        e_move.pack(side="left", padx=(4, 4))
        e_move.bind("<Return>", lambda _e: self._move())
        ttk.Button(r2, text="MOVE", width=10, command=self._move).pack(side="left", padx=(0, 24))

        ttk.Label(r2, text=f"SPEED (mm/s, ~{self.SPEED_MAX_MM_S:.0f})").pack(side="left")
        self.speed_mm = tk.StringVar(value="400")
        e_speed = ttk.Entry(r2, textvariable=self.speed_mm, width=10)
        e_speed.pack(side="left", padx=(4, 4))
        e_speed.bind("<Return>", lambda _e: self._speed())
        ttk.Button(r2, text="SPEED 설정", width=12, command=self._speed).pack(side="left")

        r3 = ttk.Frame(ctl); r3.pack(fill="x", padx=8, pady=6)
        ttk.Label(r3, text=f"ACCEL (mm/s², ~{self.ACCEL_MAX_MM_S2:.0f}) — 낮으면 출발·정지가 굼뜸").pack(side="left")
        self.accel_mm = tk.StringVar(value="1500")
        e_accel = ttk.Entry(r3, textvariable=self.accel_mm, width=10)
        e_accel.pack(side="left", padx=(4, 4))
        e_accel.bind("<Return>", lambda _e: self._accel())
        ttk.Button(r3, text="ACCEL 설정", width=12, command=self._accel).pack(side="left")

        tests = (ttk.LabelFrame(self.container, text="테스트 (test.py rail 전체)"))
        tests.pack(fill="x", padx=10, pady=4)
        row = ttk.Frame(tests); row.pack(fill="x", padx=8, pady=6)
        ttk.Button(row, text="전체 테스트 (test.py rail)", width=24,
                   command=self._full).pack(side="left", padx=4)
        ttk.Button(row, text="■ Stop", width=10,
                   command=self.runner.stop).pack(side="left", padx=4)
        ttk.Label(row, text="※ 실행 시 GUI 연결 자동 해제", foreground="#666").pack(side="left", padx=8)

        self.make_log(height=12)

    # ---------- 명령 ----------
    def _send(self, cmd: str):
        LINK.send(cmd)

    @staticmethod
    def _parse(text: str):
        try:
            return float(text)
        except ValueError:
            return None

    def _move(self):
        mm = self._parse(self.move_mm.get())
        if mm is None or not (0.0 <= mm <= 1000.0):
            self.log_q.put(("err", f"MOVE 값은 0 ~ 1000 mm: {self.move_mm.get()!r}\n"))
            return
        self._send(f"MOVE {mm:.3f}")

    def _speed(self):
        v = self._parse(self.speed_mm.get())
        if v is None or not (0.0 < v <= self.SPEED_MAX_MM_S):
            self.log_q.put(("err", f"SPEED 값은 0 < v <= {self.SPEED_MAX_MM_S:.0f} mm/s: "
                                   f"{self.speed_mm.get()!r}\n"))
            return
        self._send(f"SPEED {v:g}")

    def _accel(self):
        v = self._parse(self.accel_mm.get())
        if v is None or not (0.0 < v <= self.ACCEL_MAX_MM_S2):
            self.log_q.put(("err", f"ACCEL 값은 0 < v <= {self.ACCEL_MAX_MM_S2:.0f} mm/s²: "
                                   f"{self.accel_mm.get()!r}\n"))
            return
        self._send(f"ACCEL {v:g}")

    def _full(self):
        release_link(self.log_q)
        self._refresh_conn()
        argv = [sys.executable, "test.py", "rail", "--port", self.port.get()]
        self.runner.start(argv)


# ============================================================
#  엔트리
# ============================================================
def main():
    print(f"[gui] 시작  python={sys.executable}  platform={sys.platform}", flush=True)
    try:
        root = tk.Tk()
    except tk.TclError as e:
        print(f"[gui] tk.Tk() 실패: {e}  (conda install -c conda-forge tk)", flush=True)
        sys.exit(1)

    # 모든 화면이 이 크기 안에 들어가도록 고정. 화면 전환 시 크기 변화 없음.
    root.title("ManiBox")
    root.geometry("1000x720+120+60")
    root.minsize(1000, 720)
    _apply_fonts(root)
    _apply_icon(root)

    nav = Navigator(root)
    nav.go(WelcomeScreen)
    _front(root)
    print("[gui] 창 표시됨.", flush=True)
    try:
        root.mainloop()
    except KeyboardInterrupt:
        print("[gui] 종료 (Ctrl-C)", flush=True)
    finally:
        LINK.disconnect()


if __name__ == "__main__":
    main()
