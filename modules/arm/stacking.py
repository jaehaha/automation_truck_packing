"""상자 적재 모션 플래너.

흰 칸(투입칸)에 놓인 상자를 윗면으로 흡착해, 지금까지 쌓인 상자를 피해서 지정한 위치·방향에 놓는 경로를 만든다.
물리 시뮬레이션이 아니라 운동학 계획이다: 관절 각도 경로를 만들고, 매 단계 MuJoCo 로 충돌만 검사한다.

  from stacking import Stacker
  s = Stacker([6, 6, 5])                         # 이번 작업의 상자 종류 목록(투입 순서)
  plan = s.plan(0, center=(0.80, -0.12, -0.539), size_xyz=(0.12, 0.24, 0.18))
  if plan.ok: s.commit(plan)                     # 놓인 것으로 확정 -> 다음 상자의 장애물이 된다

좌표: 로봇 좌표계 = 레일 0 일 때의 팔 베이스(ORIGIN)가 원점, 축은 프레임과 같다(x = 레일 방향, z = 위).
center 는 놓인 상자의 중심, size_xyz 는 놓인 상자의 x, y, z 방향 길이(상자 치수의 순열)다.

놓는 방식 (흡착한 면 = 가로x세로 면)
  top  : 흡착면이 위(size_xyz 의 z 가 상자 높이). 공구가 수직이라 낮은 곳만 닿는다.
  side : 흡착면이 팔 쪽(-x)을 본다(size_xyz 의 x 가 상자 높이). 공구가 수평(+x).
"""
from __future__ import annotations

import hashlib
import pickle
import re
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parent
ORIGIN = np.array([0.700, 0.280, 0.672])          # 로봇 좌표계 원점의 월드 좌표 (레일 0 일 때 팔 베이스)
FLOOR = 0.04                                     # 바닥판 윗면 높이 (월드 z)
# 상자 6종 (가로 L, 세로 W, 높이 H) m. 8 cm 단위 치수.
BOXES = {1: (0.08, 0.08, 0.08), 2: (0.08, 0.08, 0.10), 3: (0.16, 0.08, 0.08),
         4: (0.16, 0.08, 0.10), 5: (0.16, 0.08, 0.12), 6: (0.16, 0.16, 0.10)}
COLORS = {1: "0.85 0.33 0.31", 2: "0.93 0.62 0.25", 3: "0.90 0.82 0.35", 4: "0.45 0.72 0.42", 5: "0.35 0.60 0.85", 6: "0.62 0.45 0.78"}
PICK_XY = (0.29, 0.28)                            # 투입 위치 (월드 x, y): 흰 칸 중앙. 상자는 높이가 위, 가로(L)가 y 방향으로 놓인다.
GAP = 0.0005                                      # 흡착컵 끝과 상자 사이 틈
TOL = 0.0015                                      # 이보다 깊이 파고들면 충돌로 본다 (맞닿음은 허용)
CLEAR = 0.003                                     # 옮기는 동안 상자(든 상자·쌓인 상자)가 지켜야 하는 최소 간격
LIFT_OFF = 0.005                                  # 놓기 직전·집은 직후에 받침에서 띄우는 높이
TILT_OFF, TILT_RISE = 5, 0.010                    # 수직으로 띄울 수 없을 때: 공구를 이만큼(도) 기울이면서 이 높이만큼 띄운다
APPROACH = 0.015                                  # 옆면 방식으로 놓을 때 자리보다 이만큼 앞(-x)에서 내려온 뒤 밀어 넣는다
MV_CUP_POS = 0.47                                       # 흡착컵 끝 최고 높이 (월드 z)
EX, EY, EZ = np.eye(3)
VIEW = dict(lookat=(0.800, 0.316, 0.273), distance=2.129, azimuth=-31.13, elevation=-1.67)   # 모델의 카메라 "view" 와 같은 시점


def set_view(cam):
    """뷰어의 자유 시점을 기본 시점(카메라 "view")에 맞춘다. 마우스로는 계속 움직일 수 있다."""
    cam.lookat[:] = VIEW["lookat"]
    cam.distance, cam.azimuth, cam.elevation = VIEW["distance"], VIEW["azimuth"], VIEW["elevation"]
READY = np.array([0, 0, 0.3, 0, 0, -1.1, 0.0])
SEEDS = [np.array(s, float) for s in (
    [0, 0, 0.3, -1.0, 0, -0.8, 0], [0, 0, 2.5, -2.8, 0, -1.2, 0], [0, 0, 0.0, -2.6, 0, 1.1, 0], [0, 0, 1.5, -1.5, 0, 0, 0],
    [0, 0.5, 1.2, -1.2, 0.5, 0.3, 0], [0, -0.5, 1.2, -1.2, -0.5, 0.3, 0], [0, 0, 0.4, -0.7, 0, -1.2, 0], [0, 2.4, 2.0, -2.0, 0, -0.5, 0])]


def tilt(v, deg):
    """공구를 수직 아래(-z)에서 전방(+x)으로 deg 만큼 세우는 회전(y축 둘레)을 벡터에 적용."""
    a = np.radians(deg); c, s = np.cos(a), np.sin(a)
    return np.array([c * v[0] - s * v[2], v[1], s * v[0] + c * v[2]])


def roll(v, deg):
    """+x 축 둘레 회전."""
    a = np.radians(deg); c, s = np.cos(a), np.sin(a)
    return np.array([v[0], c * v[1] - s * v[2], s * v[1] + c * v[2]])


def _fingerprint(sources):
    """모델과 계획 코드의 내용으로 만든 지문. 이 중 하나라도 바뀌면 저장해 둔 계획은 쓰지 않는다."""
    h = hashlib.md5()
    for path in [ROOT / "model" / "manibox_modified.xml", Path(__file__), *map(Path, sources)]:
        h.update(Path(path).read_bytes())
    return h.hexdigest()


def load_cache(name, sources=()):
    """outputs/<name>.pkl 에 저장해 둔 계획을 읽는다. 없거나, 모델·코드가 바뀌어 지문이 다르면 None."""
    path = ROOT / "outputs" / f"{name}.pkl"
    if not path.exists():
        return None
    try:
        saved = pickle.loads(path.read_bytes())
    except Exception:
        return None
    return saved["data"] if isinstance(saved, dict) and saved.get("fingerprint") == _fingerprint(sources) else None


def save_cache(name, data, sources=()):
    """계획을 outputs/<name>.pkl 에 저장한다. 반환: 저장한 경로."""
    path = ROOT / "outputs" / f"{name}.pkl"
    path.parent.mkdir(exist_ok=True)
    path.write_bytes(pickle.dumps(dict(fingerprint=_fingerprint(sources), data=data)))
    return path


@dataclass
class Plan:
    ok: bool
    index: int = -1
    reason: str = ""
    mode: str = ""
    center: np.ndarray | None = None               # 놓인 상자 중심 (월드)
    quat: np.ndarray | None = None                 # 놓인 상자 자세 (월드)
    held_geom: int = -1
    approach: list = field(default_factory=list)   # 빈 손: 현재 자세 -> 집는 자세
    carry: list = field(default_factory=list)      # 상자를 든 채: 집는 자세 -> 놓는 자세
    retreat: list = field(default_factory=list)    # 빈 손: 놓은 뒤 물러남


class Stacker:
    def __init__(self, types: list[int]):
        self.types = list(types)
        self.m = mujoco.MjModel.from_xml_string(self._xml())
        # 들고 있는 상자의 위치(geom_pos)를 실행 중에 바꾸므로, 컴파일 때 만든 몸체별 경계 트리를 쓰지 않게 한다.
        self.m.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_MIDPHASE
        self.d = mujoco.MjData(self.m)
        self.tcp = self.m.site("tcp").id
        self.lo, self.hi = self.m.jnt_range[:, 0].copy(), self.m.jnt_range[:, 1].copy()
        self.held_ids = {(k, g): self.m.geom(f"held_{k}_{g}").id for k in BOXES for g in (0, 1)}
        self.box_mocap = [self.m.body(f"box_{i}").mocapid[0] for i in range(len(types))]
        self.q = READY.copy()
        self.placed: list[int] = []
        self.clear = CLEAR        # 상자가 낀 접촉에 요구하는 간격. None 이면 맞닿음 허용 (집는 순간·놓는 순간에만)
        self._isbox = np.zeros(self.m.ngeom, bool)
        self._isbox[list(self.held_ids.values()) + [self.m.geom(f"box_{i}").id for i in range(len(types))]] = True
        self.m.geom_margin[self._isbox] = CLEAR                   # 간격 CLEAR 안쪽이면 접촉으로 보고되게 한다
        self._active = None       # 들고 있는 상자 geom 은 충돌을 켠 채로 컴파일해야 충돌 구조에 들어간다. 평소에는 실행 중에 꺼 둔다.
        self._hold(-1)
        self._jp, self._jr = np.zeros((3, self.m.nv)), np.zeros((3, self.m.nv))
        for i in range(len(types)):
            self._hide(i)
        self.d.qpos[:] = self.q
        mujoco.mj_forward(self.m, self.d)

    # ---------- 모델 ----------
    def _xml(self) -> str:
        x = (ROOT / "model" / "manibox_modified.xml").read_text().replace('meshdir="../assets"', f'meshdir="{ROOT / "assets"}"')
        tcp_z = float(re.search(r'<site name="tcp" pos="0 0 ([\d.]+)"', x).group(1))   # 흡착컵 끝면까지의 거리 (모델에서 읽는다)
        held = ""
        for k, (l, w, h) in BOXES.items():
            for g in (0, 1):                      # g=1: 공구 x축이 상자 세로(W) 방향이 되도록 90도 돌려 잡은 경우
                held += (f'<geom name="held_{k}_{g}" type="box" pos="0 0 {tcp_z + GAP + h / 2}" size="{l / 2} {w / 2} {h / 2}"'
                         f'{" quat=\"0.70710678 0 0 0.70710678\"" if g else ""} contype="4" conaffinity="7" mass="0" group="2" rgba="{COLORS[k]} 0"/>\n')
        x = x.replace('<site name="tcp"', held + '<site name="tcp"')
        boxes = ""
        for i, k in enumerate(self.types):
            l, w, h = BOXES[k]
            boxes += (f'<body name="box_{i}" mocap="true" pos="0 0 -5"><geom name="box_{i}" type="box" size="{l / 2} {w / 2} {h / 2}" '
                      f'contype="1" conaffinity="0" group="2" rgba="{COLORS[k]} 1"/></body>\n')
        return re.sub(r'(\s*<body name="carriage")', "\n" + boxes + r"\1", x, count=1)

    def _hide(self, i):
        self.d.mocap_pos[self.box_mocap[i]] = (0, 0, -5 - i)

    def set_box(self, i, center, quat=(1, 0, 0, 0)):
        self.d.mocap_pos[self.box_mocap[i]] = center
        self.d.mocap_quat[self.box_mocap[i]] = quat

    def spawn(self, i, xy=None):
        """상자 i 를 투입 위치(또는 흰 칸의 xy)에 놓는다 (높이가 위, 가로가 y 방향)."""
        h = BOXES[self.types[i]][2]
        x, y = PICK_XY if xy is None else xy
        self.set_box(i, (x, y, FLOOR + h / 2), (0.70710678, 0, 0, 0.70710678))

    def carry_box(self, i, gid):
        """표시용: 상자 i 를 지금 자세(self.d.qpos)에서 공구에 붙은 위치로 옮긴다. gid = 그 상자를 든 held geom.
        든 상자를 held geom 의 색(모델 값)을 바꿔 보여 주면 뷰어 창에는 반영되지 않아 상자가 사라졌다 나타나는 것처럼 보인다.
        그래서 상자 본체(mocap, 데이터 값)를 매 프레임 공구에 맞춰 옮겨서 보여 준다."""
        mujoco.mj_kinematics(self.m, self.d)
        quat = np.zeros(4); mujoco.mju_mat2Quat(quat, self.d.geom_xmat[gid])
        self.set_box(i, self.d.geom_xpos[gid], quat)

    # ---------- 충돌 ----------
    def _hold(self, gid):
        if gid == self._active:
            return
        for g in self.held_ids.values():
            self.m.geom_contype[g], self.m.geom_conaffinity[g] = (4, 7) if g == gid else (0, 0)
        self._active = gid

    def free(self, q, held=-1) -> bool:
        self._hold(held)
        self.d.qpos[:] = q
        mujoco.mj_kinematics(self.m, self.d)
        mujoco.mj_collision(self.m, self.d)
        for i in range(self.d.ncon):
            c = self.d.contact[i]
            need = self.clear if self.clear is not None and (self._isbox[c.geom1] or self._isbox[c.geom2]) else -TOL
            if c.dist < need:
                return False
        return True

    def path(self, a, b, held=-1) -> bool:
        diff = np.abs(b - a)
        n = int(max(2, np.ceil(diff[0] / 0.01), np.ceil(diff[1:].max() / 0.03))) + 1
        return all(self.free(a + (b - a) * t, held) for t in np.linspace(0, 1, n))

    # ---------- 역기구학 ----------
    def ik(self, pos, zdir, xdir, held=-1, near=None, all_=False):
        """tcp 위치, 공구축(zdir), 공구 x축(xdir)을 만족하는 충돌 없는 자세. near 가 있으면 그 자세에서 이어지는 해만."""
        pos, zdir, xdir = (np.asarray(v, float) for v in (pos, zdir, xdir))
        m, d, W = self.m, self.d, 0.3

        def fun(q, ref, reg):
            d.qpos[:] = q; mujoco.mj_kinematics(m, d)
            R = d.site_xmat[self.tcp].reshape(3, 3)
            r = [d.site_xpos[self.tcp] - pos, W * (R[:, 2] - zdir), W * (R[:, 0] - xdir)]
            return np.concatenate(r + ([reg * (q - ref)] if ref is not None else []))

        def jac(q, ref, reg):
            d.qpos[:] = q; mujoco.mj_kinematics(m, d); mujoco.mj_comPos(m, d)
            mujoco.mj_jacSite(m, d, self._jp, self._jr, self.tcp)
            R = d.site_xmat[self.tcp].reshape(3, 3)
            J = [self._jp, W * np.cross(self._jr.T, R[:, 2]).T, W * np.cross(self._jr.T, R[:, 0]).T]
            return np.vstack(J + ([reg * np.eye(m.nv)] if ref is not None else []))

        def run(seed, ref):
            r = least_squares(fun, np.clip(seed, self.lo + 1e-4, self.hi - 1e-4), jac=jac, bounds=(self.lo, self.hi),
                              args=(ref, 0.003), xtol=1e-12, ftol=1e-14, gtol=1e-12, max_nfev=60)
            f = r.fun[:9]
            return r.x if np.abs(f[:3]).max() < 4e-4 and np.abs(f[3:]).max() < 1e-3 else None

        if near is not None:
            q = run(near, near)
            if q is not None and abs(q[0] - near[0]) < 0.3 and np.abs(q[1:] - near[1:]).max() < 0.7 and self.free(q, held):
                return q
            return None
        out = []
        for off in (0.4, 0.15, -0.3, 0.6):
            for s in SEEDS:
                s = s.copy(); s[0] = np.clip(pos[0] - ORIGIN[0] - off, self.lo[0], self.hi[0])
                q = run(s, None)
                if q is None or not self.free(q, held) or any(np.abs(q - o).max() < 0.05 for o in out):
                    continue
                if not all_:
                    return q
                out.append(q)
        return out if all_ else None

    def tcp_pos(self, q):
        self.d.qpos[:] = q; mujoco.mj_kinematics(self.m, self.d)
        return self.d.site_xpos[self.tcp].copy()

    # ---------- 동작 조각 ----------
    def _lift(self, wps, pos, zdir, xdir, held, top=MV_CUP_POS, step=0.02):
        """자세를 유지한 채 수직으로 올릴 수 있는 데까지 올린다."""
        pos = np.array(pos, float)
        while pos[2] + step <= top + 1e-9:
            n = self.ik(pos + (0, 0, step), zdir, xdir, held, near=wps[-1])
            if n is None or not self.path(wps[-1], n, held):
                break
            wps.append(n); pos[2] += step
        return pos

    def depart(self, pos, x0, held, tilt_up: bool):
        """pos 에서 시작해 위쪽 이송 자세(공구 수평 +x, 가능한 한 높이)까지 가는 경로들.
        tilt_up=True: 공구 수직(-z)으로 시작해 들어올린 뒤 세운다.  False: 처음부터 수평이라 들어올리기만 한다.
        시작점에서는 상자가 받침·이웃과 맞닿아 있으므로 첫 구간(LIFT_OFF 만큼 띄우기)만 맞닿음을 허용하고,
        그 뒤로는 간격 CLEAR 를 지킨다. 수직으로 띄우는 경로가 없으면 조금 기울이면서 띄우는 경로를 찾는다. 수평으로 시작하면(옆면 방식) 띄운 뒤 -x 로 APPROACH 만큼 물러나서 올라간다.
        반환: (경유 자세 목록, 끝 tcp 위치, 끝 공구 x축) 또는 None. 경유 자세 0->1 이 맞닿음 허용 구간이다."""
        z0 = -EZ if tilt_up else EX
        self.clear = None
        starts = self.ik(pos, z0, x0, held, all_=True)
        self.clear = CLEAR
        # 띄우는 방법 두 가지: (기울기 도, 띄우는 높이). 먼저 그대로 수직으로 띄우고, 그걸로 안 되면 공구를 조금 기울이면서 띄운다.
        # 공구가 길어서(홀더 55 mm) 높은 상자를 수직으로 집으면 팔이 거의 다 접혀, 수직인 채로는 더 올리지 못하는 자세가 있다.
        for a0, rise in (((0, LIFT_OFF), (TILT_OFF, TILT_RISE)) if tilt_up else ((0, LIFT_OFF),)):
            z1, x1 = (tilt(z0, a0), tilt(x0, a0)) if a0 else (z0, x0)
            for q0 in starts:
                wps, p = [q0], np.array(pos, float) + (0, 0, rise)
                self.clear = None
                n = self.ik(p, z1, x1, held, near=q0)
                first = n is not None and self.path(q0, n, held)
                self.clear = CLEAR
                if not first or not self.free(n, held):
                    continue
                wps.append(n)
                if not tilt_up:
                    n = self.ik(p - (APPROACH, 0, 0), z0, x0, held, near=wps[-1])
                    if n is None or not self.path(wps[-1], n, held):
                        continue
                    wps.append(n); p = p - (APPROACH, 0, 0)
                if not a0:
                    p = self._lift(wps, p, z0, x0, held, top=MV_CUP_POS if not tilt_up else 0.30)
                ok = True
                if tilt_up:
                    for a in range(a0 + 5, 91, 5):
                        zd, xd = tilt(-EZ, a), tilt(x0, a)
                        toc = ORIGIN[1] - p[1]                       # 중심선(y)에서 벗어나 있으면 세우면서 중심선 쪽으로 옮긴다
                        dys = (0.0,) if abs(toc) < 1e-6 else (np.sign(toc) * min(abs(toc), 0.03), 0.0)
                        for dy, dz, dx in ((dy, dz, dx) for dy in dys for dz in (0.06, 0.04, 0.02, 0.0, -0.02, -0.04) for dx in (0.0, 0.04, 0.08, -0.04, 0.12)):
                            if p[2] + dz > MV_CUP_POS:
                                continue
                            n = self.ik(p + (dx, dy, dz), zd, xd, held, near=wps[-1])
                            if n is not None and self.path(wps[-1], n, held):
                                wps.append(n); p = p + (dx, dy, dz); break
                        else:
                            ok = False; break
                    if ok:
                        p = self._lift(wps, p, EX, tilt(x0, 90), held)
                if ok:
                    return wps, p, (tilt(x0, 90) if tilt_up else np.asarray(x0, float))
        return None

    def _roll(self, q, p, xa, xb, held):
        """이송 자세에서 공구축(+x) 둘레로 xa -> xb 까지 돌린다."""
        ang = np.degrees(np.arctan2(np.dot(np.cross(xa, xb), EX), np.dot(xa, xb)))
        if abs(ang) < 1:
            return [q]
        for total in (ang, ang - np.sign(ang) * 360):
            wps, n_steps = [q], int(np.ceil(abs(total) / 10))
            for i in range(1, n_steps + 1):
                n = self.ik(p, EX, roll(xa, total * i / n_steps), held, near=wps[-1])
                if n is None or not self.path(wps[-1], n, held):
                    break
                wps.append(n)
            else:
                return wps
        return None

    def _line(self, q, pa, pb, xdir, held):
        """공구 수평 자세로 tcp 를 직선 이동."""
        wps, n = [q], int(np.ceil(np.linalg.norm(pb - pa) / 0.03))
        for i in range(1, n + 1):
            nq = self.ik(pa + (pb - pa) * i / n, EX, xdir, held, near=wps[-1])
            if nq is None or not self.path(wps[-1], nq, held):
                return None
            wps.append(nq)
        return wps

    def _connect(self, qa, pa, qb, pb, xdir, held):
        """이송 자세 A -> B. 팔 자세를 B 와 같게 만든 뒤 레일만 민다. 안 되면 직선 이동."""
        qm = qb.copy(); qm[0] = np.clip(qb[0] - (pb[0] - pa[0]), self.lo[0], self.hi[0])
        if self.path(qa, qm, held) and self.path(qm, qb, held):
            return [qa, qm, qb]
        pm = self.tcp_pos(qm)
        for via in (np.array([pa[0], pm[1], pm[2]]), np.array([pa[0], pa[1], pm[2]])):
            w1 = self._line(qa, pa, via, xdir, held)
            if w1 is None:
                continue
            w2 = self._line(w1[-1], via, pm, xdir, held) if np.linalg.norm(via - pm) > 1e-6 else [w1[-1]]
            if w2 is None:
                continue
            w3 = self._line(w2[-1], pm, pb, xdir, held)
            if w3 is not None and self.path(w3[-1], qb, held):
                return w1 + w2[1:] + w3[1:] + [qb]
        return None

    # ---------- 계획 ----------
    def plan(self, i: int, center, size_xyz, mode: str | None = None, frame: str = "robot", grip=(0.0, 0.0), pick_xy=None) -> Plan:
        """상자 i(투입 위치에 있다고 가정)를 center, size_xyz 로 놓는 경로를 만든다. 지금까지 commit 한 상자는 장애물이다.
        grip: 흡착점이 흡착면 중심에서 벗어난 거리 (상자 가로 방향, 세로 방향) m. 투입 위치에서 가로 = +y, 세로 = -x 다.
        pick_xy: 집는 위치(월드 x, y). 없으면 PICK_XY. 윗면 집기는 팔 바로 아래로만 닿아서 중심선(y = ORIGIN[1]) 위의 자리만 된다."""
        k = self.types[i]; L, W, H = BOXES[k]
        c = np.asarray(center, float) + (ORIGIN if frame == "robot" else 0)
        s = np.asarray(size_xyz, float)
        if not np.allclose(sorted(s), sorted((L, W, H)), atol=1e-6):
            return Plan(False, i, f"size_xyz {tuple(s)} 가 상자 {k} 치수 {(L, W, H)} 의 순열이 아니다")
        modes = [mode] if mode else [mm for mm, ax in (("top", 2), ("side", 0)) if abs(s[ax] - H) < 1e-6]
        if not modes:
            return Plan(False, i, "흡착면(가로x세로 면)이 위나 팔 쪽(-x)을 보는 방향만 놓을 수 있다 (상자 높이가 y 방향인 배치는 미지원)")
        if c[2] - s[2] / 2 < FLOOR - 1e-6:
            return Plan(False, i, "바닥판 아래로 내려가는 위치")
        self._hide(i)             # 계획 중에는 들고 있는 상자(held geom)가 이 상자를 대신한다
        best = [0, "경로 없음"]     # 여러 시도 중 가장 멀리 간 단계의 실패 이유를 남긴다

        def fail(stage, msg):
            if stage >= best[0]:
                best[:] = [stage, msg]
        off_pick = grip[0] * EY - grip[1] * EX                       # 흡착점 오프셋 (월드, 투입 위치)
        if abs(grip[0]) > L / 2 or abs(grip[1]) > W / 2:
            return Plan(False, i, "grip 이 흡착면 밖이다")
        pxy = PICK_XY if pick_xy is None else pick_xy
        pick_pos = np.array([pxy[0], pxy[1], FLOOR + H + GAP]) + off_pick
        for mode in modes:
            if mode == "top":
                place_pos, la, lb = c + (0, 0, H / 2 + GAP), (EX if abs(s[0] - L) < 1e-6 else EY), None
                lb = EY if la is EX else EX
            else:
                place_pos, la = c - (H / 2 + GAP, 0, 0), (EY if abs(s[1] - L) < 1e-6 else EZ)
                lb = EZ if la is EY else EY
            # 공구 x축(u0)을 집을 때 어디로 둘지 4가지. 상자 가로(L)는 투입 위치에서 y 방향이다.
            # 공구 단면은 카메라 받침 쪽(-x)으로 82 mm 나온다. 옆면 방식에서는 받침이 위(빈 공간)를 보게 되는
            # 잡는 방향·부호부터 시도해, 받침이 바닥이나 이웃 상자 쪽으로 가지 않게 한다.
            up_g = (0 if la is EZ else 1) if mode == "side" else None
            u0s = sorted((EY, -EY, EX, -EX), key=lambda u: (0 if abs(u[1]) > 0.5 else 1) != up_g) if mode == "side" else (EY, -EY, EX, -EX)
            for u0 in u0s:
                g = 0 if abs(u0[1]) > 0.5 else 1
                held = self.held_ids[(k, g)]
                o_s = np.array([off_pick @ u0, off_pick @ np.cross(-EZ, u0)])   # 오프셋을 공구 좌표로
                self.m.geom_pos[held, :2] = -o_s                                 # 상자는 공구에서 반대쪽으로 그만큼 치우쳐 붙는다
                dp = self.depart(pick_pos, u0, held, tilt_up=True)
                if dp is None:
                    fail(1, "집은 뒤 이송 자세로 올리지 못함"); continue
                wa, pa, xa = dp
                for sign in ((-1, 1) if mode == "side" else (1, -1)):
                    v0 = sign * (la if g == 0 else lb)          # 놓는 순간의 공구 x축
                    zt = -EZ if mode == "top" else EX
                    dq = self.depart(place_pos + o_s[0] * v0 + o_s[1] * np.cross(zt, v0), v0, held, tilt_up=(mode == "top"))
                    if dq is None:
                        fail(2, "놓는 위치가 막혀 있거나(다른 상자·벽과 겹침) 팔이 닿지 못함"); continue
                    wb, pb, xb = dq
                    wr = self._roll(wa[-1], pa, xa, xb, held)
                    if wr is None:
                        fail(3, "이송 자세에서 상자를 돌리지 못함"); continue
                    wc = self._connect(wr[-1], pa, wb[-1], pb, xb, held)
                    if wc is None:
                        fail(4, "투입칸 위에서 놓는 위치 위까지 가는 길이 막힘"); continue
                    carry = wa + wr[1:] + wc[1:] + wb[::-1][1:]
                    # 빈 손 구간: 접근(현재 자세 -> 집는 자세), 후퇴(놓은 자세 -> 투입칸 위)
                    appr = self._link(self.q, wa[-1])
                    if appr is None:
                        fail(5, "현재 자세에서 투입칸 위로 가는 길이 막힘"); continue
                    Rm = np.column_stack([la, np.cross(self._hdir(mode), la), self._hdir(mode)])
                    quat = np.zeros(4); mujoco.mju_mat2Quat(quat, Rm.flatten())
                    self.set_box(i, c, quat)                    # 놓인 상자를 장애물로 두고 후퇴 경로 확인
                    back = wb + wc[::-1][1:]
                    ok = True
                    for n_seg, (a, b) in enumerate(zip(back[:-1], back[1:])):
                        self.clear = None if n_seg <= 1 else CLEAR          # 놓은 직후 공구는 상자에 맞닿아 있다 (띄우기·물러나기 구간)
                        ok = ok and self.path(a, b)
                    self.clear = CLEAR
                    self._hide(i)
                    if not ok:
                        fail(6, "놓은 뒤 팔이 빠져나오지 못함"); continue
                    return Plan(True, i, "", mode, c, quat, held, appr + wa[::-1][1:], carry, back)
        self._hide(i)
        return Plan(False, i, best[1])

    @staticmethod
    def _hdir(mode):
        return EZ if mode == "top" else EX

    def _link(self, a, b):
        """빈 손으로 a -> b. 직접 안 되면 대기 자세(READY)를 거친다."""
        if self.path(a, b):
            return [a, b]
        r1, r2 = READY.copy(), READY.copy(); r1[0], r2[0] = a[0], b[0]
        chain = [a, r1, r2, b]
        return chain if all(self.path(p, q) for p, q in zip(chain[:-1], chain[1:])) else None

    def commit(self, plan: Plan):
        """계획대로 놓였다고 확정한다."""
        self.set_box(plan.index, plan.center, plan.quat)
        self.placed.append(plan.index)
        self.q = plan.retreat[-1].copy()

    # ---------- 재생 ----------
    def frames(self, plan: Plan, step=0.04):
        """재생용 프레임: (관절값, 상자를 들고 있는지). 관절은 프레임당 step rad(레일은 step/3 m) 이하로 움직인다."""
        def dense(wps):
            for a, b in zip(wps[:-1], wps[1:]):
                n = int(max(1, np.ceil(abs(b[0] - a[0]) / (step / 3)), np.ceil(np.abs(b[1:] - a[1:]).max() / step)))
                for t in np.linspace(0, 1, n + 1)[1:]:
                    yield a + (b - a) * t
        for q in dense(plan.approach):
            yield q, False, "approach"
        for q in dense(plan.carry):
            yield q, True, "carry"
        for q in dense(plan.retreat):
            yield q, False, "retreat"
