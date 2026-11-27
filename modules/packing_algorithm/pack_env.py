"""적재 위치 학습용 MuJoCo 환경 (팔 없음).

슬라이드 1의 "Packing Model" 자리를 학습하기 위한 환경이다. 상자가 도착 순서대로 하나씩 들어오면
정책이 컨테이너 안의 자리(방향 + 격자 칸)를 고른다. 팔의 경로는 나중에 별도의 모션 플래너가
맡으므로, 여기서는 팔 대신 그 플래너가 요구하는 조건을 공간 규칙으로 검사한다.

컨테이너 좌표계 (mm)
  원점  정면 벽(가장 안쪽) 바닥의 y 끝 모서리 = 월드 (1.52, 0.52, 0.043) m
  x     출구 방향 (월드 -x),  y  세로 (월드 -y),  z  위
  공간  깊이 820 (월드 x 1.52 ~ 0.70, 그 앞은 팔 통로) x 세로 480 (바닥판) x 높이 480

공간 규칙 (팔·흡착 방식에서 온 것)
  방향      흡착면(가로 x 세로 면)이 출구(+x)를 본다. 그 면 안에서 90도 회전만 고른다 -> 2가지
  틈        상자를 gap(5 mm)만큼 키워서 격자에 놓는다
  받침      바닥 면적의 support_min 이상이 받쳐져야 한다. physics=True 면 MuJoCo 로 떨어뜨려 실제로 서는지도 본다
  진입 통로  놓는 순간 상자 앞(+x, 출구 쪽)이 상자 바닥 높이보다 낮아야 한다 (앞에서 내려와 밀어 넣는다)
  높이      상자 윗면 <= 480
  하차 순서  먼저 내릴 상자의 앞이나 위에 나중에 내릴 상자를 두면 위반 1건 (보상에서 뺀다)

행동  Discrete(2 * NX * NY): a = o * NX * NY + i * NY + j  (o 방향, i x 칸, j y 칸). 높이는 아래 받침으로 정해진다.
      action_masks() 가 지금 상자에 둘 수 있는 행동을 알려 준다 (sb3-contrib MaskablePPO 와 바로 쓸 수 있다).
관측  heightmap (NX, NY), 지금 상자, 다음 lookahead 개 상자, 진행률.

실행 예
  uv run python pack_env.py --check                 # gymnasium 규격 검사
  uv run python pack_env.py --episodes 5            # 기준 정책(안쪽·아래부터) 통계
  uv run python pack_env.py --render human          # 뷰어로 한 에피소드 보기
  uv run python pack_env.py --save-json out.json    # 마지막 에피소드 배치를 플래너 입력 형식으로 저장
"""
from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import gymnasium as gym
import mujoco
import mujoco.viewer
import numpy as np
from gymnasium import spaces
from numpy.lib.stride_tricks import sliding_window_view

C_ORIGIN = np.array([1.241, 0.52, 0.052])   # 컨테이너 원점의 월드 좌표 (m). 컨테이너 x = 월드 -x, y = 월드 -y
                                            # 경인 최종 도면 (handoff/ManiBox_Floor_Panels_Dimensions): 적재 X41 ~ X1241 (안쪽 끝 = 구획선 X1241),
                                            # 바닥판 윗면 z 0.052 (판 12 mm, 로봇 장착면은 바닥판 위 620 mm). 이전 값 (1.52, 0.52, 0.043)
ROBOT_ORIGIN = np.array([0.210, 0.280, 0.672])   # 로봇 좌표계 원점의 월드 좌표 (레일 0 = 투입 위치쪽 끝 일 때 팔 베이스). stacking.ORIGIN 과 일치해야 함

# 종류 -> (가로 L, 세로 W, 높이 H) mm. 흡착면 = L x W (투입 칸에서 윗면).
BOX_TYPES = {1: (80, 80, 80), 2: (80, 80, 100), 3: (160, 80, 80), 4: (160, 80, 100), 5: (160, 160, 100)}
STOP_COLORS = [(0.84, 0.15, 0.16), (1.00, 0.50, 0.05), (0.90, 0.80, 0.10), (0.17, 0.63, 0.17), (0.09, 0.75, 0.81),
               (0.12, 0.47, 0.71), (0.58, 0.40, 0.74), (0.89, 0.47, 0.76), (0.55, 0.34, 0.29), (0.50, 0.50, 0.50)]
TOL = 1.0   # mm


@dataclass
class Config:
    depth: int = 820
    width: int = 480
    height: int = 480
    grid: int = 10                      # 격자 한 칸 (mm)
    gap: int = 5                        # 상자 사이 최소 틈 (mm)
    support_min: float = 0.8            # 받침 비율 하한
    box_types: dict = field(default_factory=lambda: dict(BOX_TYPES))
    shares: dict | None = None          # 종류별 도착 비율. None 이면 균등
    stops: tuple = (6, 12)              # 한 에피소드의 하차 지점 수 범위
    per_stop: tuple = (3, 10)           # 지점당 상자 수 범위
    order: str = "sorted"               # "sorted": 나중에 내릴 지점부터 도착 / "random": 무작위 도착
    max_boxes: int = 120
    lookahead: int = 5
    physics: bool = True                # MuJoCo 로 떨어뜨려 안정성 확인
    settle_time: float = 0.3            # s
    drift_tol: float = 3.0              # mm. 놓은 뒤 이보다 많이 움직이면 불안정
    tilt_tol: float = 2.0               # deg
    w_util: float = 10.0                # 보상 = w_util * (상자 부피 / 공간 부피) - w_block * 하차 순서 위반 수
    w_block: float = 0.2
    w_fail: float = 1.0                 # 불안정으로 끝날 때 벌점


def to_world(center_mm, dims_xyz_mm):
    """컨테이너 좌표(mm) -> 월드 좌표(m). 상자는 축 정렬이라 치수는 순서만 그대로 둔다."""
    c = np.asarray(center_mm, float) / 1000
    return C_ORIGIN + np.array([-c[0], -c[1], c[2]]), np.asarray(dims_xyz_mm, float) / 1000


def to_planner(p: dict):
    """export() 한 항목 -> 모션 플래너 입력: 상자 중심(로봇 좌표, m), 놓은 뒤 x·y·z 방향 치수(m)."""
    w, s = to_world(p["center"], p["dims_xyz"])
    return w - ROBOT_ORIGIN, s


def _quat_x90():
    return np.array([math.cos(math.pi / 4), math.sin(math.pi / 4), 0, 0])


class PackEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(self, config: Config | None = None, render_mode: str | None = None):
        self.cfg = c = config or Config()
        self.render_mode = render_mode
        self.NX, self.NY = c.depth // c.grid, c.width // c.grid
        self.vol = c.depth * c.width * c.height
        self.dmax = max(max(v) for v in c.box_types.values())
        self.types = sorted(c.box_types)
        self.action_space = spaces.Discrete(2 * self.NX * self.NY)
        feat = len(self.types) + 4                          # 종류 원핫 + (L, W, H)/dmax + 지점 비율
        self.observation_space = spaces.Dict({
            "heightmap": spaces.Box(0, 1, (self.NX, self.NY), np.float32),
            "box": spaces.Box(0, 1, (feat,), np.float32),
            "next": spaces.Box(0, 1, (c.lookahead, feat), np.float32),
            "progress": spaces.Box(0, 1, (2,), np.float32),
        })
        self.m = self.d = None
        self._viewer = self._renderer = None

    # ---------- 에피소드 ----------
    def make_job(self, rng):
        c = self.cfg
        shares = np.array([(c.shares or {}).get(k, 1.0) for k in self.types], float)
        shares /= shares.sum()
        n_stop = int(rng.integers(c.stops[0], c.stops[1] + 1))
        job = []                                            # (지점, 종류). 지점 1 = 가장 먼저 내림
        for stop in range(n_stop, 0, -1):
            n = int(rng.integers(c.per_stop[0], c.per_stop[1] + 1))
            job += [(stop, int(rng.choice(self.types, p=shares))) for _ in range(n)]
        if c.order == "random":
            rng.shuffle(job)
        return job[:c.max_boxes]

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        self.job = [tuple(b) for b in options["job"]] if "job" in options else self.make_job(self.np_random)
        self.n_stop = max(s for s, _ in self.job)
        self.hm = np.zeros((self.NX, self.NY))
        self.k = 0
        self.placed = []                                    # dict(id, type, stop, o, center, dims_xyz, lo, hi)
        self.util = 0.0
        self.violations = 0
        self._build_scene()
        self._mask_cache = None
        if self.render_mode == "human":
            self.render()
        return self._obs(), {"action_mask": self.action_masks().copy()}

    def _dims(self, k, o):
        L, W, H = self.cfg.box_types[k]
        return (H, L, W) if o == 0 else (H, W, L)            # 높이 축은 항상 x (흡착면이 출구를 본다)

    def _feat(self, idx):
        f = np.zeros(len(self.types) + 4, np.float32)
        if idx < len(self.job):
            stop, k = self.job[idx]
            f[self.types.index(k)] = 1
            f[len(self.types):len(self.types) + 3] = np.array(self.cfg.box_types[k]) / self.dmax
            f[-1] = stop / self.n_stop
        return f

    def _obs(self):
        c = self.cfg
        return {
            "heightmap": (self.hm / c.height).astype(np.float32),
            "box": self._feat(self.k),
            "next": np.stack([self._feat(self.k + 1 + t) for t in range(c.lookahead)]),
            "progress": np.array([self.k / len(self.job), min(self.util, 1.0)], np.float32),
        }

    # ---------- 규칙 ----------
    def _cells(self, dims):
        g, gap = self.cfg.grid, self.cfg.gap
        return math.ceil((dims[0] + gap) / g), math.ceil((dims[1] + gap) / g)

    def _feasible(self, dims):
        """모든 (i, j) 에 대해 놓을 수 있는지와 바닥 높이를 한 번에 계산한다."""
        c = self.cfg
        nx, ny = self._cells(dims)
        ok = np.zeros((self.NX, self.NY), bool)
        base = np.zeros((self.NX, self.NY))
        if nx > self.NX or ny > self.NY:
            return ok, base
        win = sliding_window_view(self.hm, (nx, ny))
        b = win.max(axis=(2, 3))
        sup = (win >= b[..., None, None] - TOL).mean(axis=(2, 3))
        colmax = sliding_window_view(self.hm, ny, axis=1).max(-1)          # (NX, NY-ny+1)
        suf = np.maximum.accumulate(colmax[::-1], axis=0)[::-1]            # suf[x] = x 이후(출구 쪽) 최대 높이
        I, J = b.shape
        front = np.full((I, J), -np.inf)
        idx = np.arange(I) + nx
        front[idx < self.NX] = suf[idx[idx < self.NX]]
        good = (b + dims[2] <= c.height + TOL) & (sup >= c.support_min - 1e-9) & (front <= b + TOL)
        ok[:I, :J], base[:I, :J] = good, b
        return ok, base

    def action_masks(self):
        if self._mask_cache is not None:
            return self._mask_cache
        mask = np.zeros((2, self.NX, self.NY), bool)
        if self.k < len(self.job):
            k = self.job[self.k][1]
            L, W, _ = self.cfg.box_types[k]
            for o in (0, 1) if L != W else (0,):
                mask[o] = self._feasible(self._dims(k, o))[0]
        self._mask_cache = mask.reshape(-1)
        return self._mask_cache

    def decode(self, a):
        o, r = divmod(int(a), self.NX * self.NY)
        return o, *divmod(r, self.NY)

    def _blocks(self, lo, hi, stop):
        """새 상자가 먼저 내릴 상자의 앞(출구 쪽)이나 위를 막는 수."""
        n = 0
        for p in self.placed:
            if stop <= p["stop"]:
                continue
            ov = [min(hi[a], p["hi"][a]) - max(lo[a], p["lo"][a]) > TOL for a in range(3)]
            front = lo[0] >= p["hi"][0] - TOL and ov[1] and ov[2]
            above = lo[2] >= p["hi"][2] - TOL and ov[0] and ov[1]
            n += front or above
        return n

    # ---------- 진행 ----------
    def step(self, action):
        c = self.cfg
        stop, k = self.job[self.k]
        o, i, j = self.decode(action)
        info = {}
        if not self.action_masks()[action]:                 # 마스크를 쓰지 않는 정책: 벌점 후 종료
            return self._obs(), -self.cfg.w_fail, True, False, {
                "reason": "invalid_action", "util": self.util, "violations": self.violations, "placed": self.k,
                "total": len(self.job), "action_mask": self.action_masks().copy()}
        dims = self._dims(k, o)
        nx, ny = self._cells(dims)
        base = self.hm[i:i + nx, j:j + ny].max()
        center = np.array([(i + nx / 2) * c.grid, (j + ny / 2) * c.grid, base + dims[2] / 2])
        lo, hi = center - np.array(dims) / 2, center + np.array(dims) / 2

        if c.physics:
            drift, tilt = self._settle(k, o, center)
            info.update(drift_mm=drift, tilt_deg=tilt)
            if drift > c.drift_tol or tilt > c.tilt_tol:
                info.update(reason="unstable", util=self.util, violations=self.violations, placed=self.k, total=len(self.job))
                self._mask_cache = None
                return self._obs(), -c.w_fail, True, False, info

        blocks = self._blocks(lo, hi, stop)
        self.hm[i:i + nx, j:j + ny] = base + dims[2]
        p = dict(id=f"b{self.k + 1:03d}", type=k, stop=stop, o=o, center=center.round(2).tolist(),
                 dims_xyz=list(dims), lo=lo, hi=hi)
        self.placed.append(p)
        self._show(self.k, p)
        frac = np.prod(dims) / self.vol
        self.util += frac
        self.violations += blocks
        reward = c.w_util * frac - c.w_block * blocks
        self.k += 1
        self._mask_cache = None
        terminated = False
        if self.k == len(self.job):
            terminated, info["reason"] = True, "done"
        elif not self.action_masks().any():
            terminated, info["reason"] = True, "no_space"
        info.update(blocks=blocks, util=self.util, violations=self.violations, placed=self.k, total=len(self.job),
                    action_mask=self.action_masks().copy())
        if self.render_mode == "human":
            self.render()
        return self._obs(), reward, terminated, False, info

    def export(self):
        """배치 결과 (컨테이너 좌표, mm). to_planner() 로 플래너 입력이 된다."""
        return [{key: p[key] for key in ("id", "type", "stop", "center", "dims_xyz")} for p in self.placed]

    # ---------- MuJoCo 장면 ----------
    def _build_scene(self):
        c = self.cfg
        spec = mujoco.MjSpec.from_string(self._container_xml())
        wb = spec.worldbody
        self.box_geom, self.probe = [], {}
        for n, (stop, k) in enumerate(self.job):
            dims = np.array(self._dims(k, 0)) / 1000
            b = wb.add_body(name=f"box_{n}", mocap=True, pos=[5 + 0.3 * n, 5, -1])
            col = STOP_COLORS[(stop - 1) % len(STOP_COLORS)]
            b.add_geom(name=f"box_{n}", type=mujoco.mjtGeom.mjGEOM_BOX, size=dims / 2, rgba=[*col, 1],
                       contype=0, conaffinity=0)
        for k in sorted({k for _, k in self.job}):
            dims = np.array(self._dims(k, 0)) / 1000
            b = wb.add_body(name=f"probe_{k}", pos=[-5 - k, -5, 1])
            b.add_freejoint()
            b.add_geom(name=f"probe_{k}", type=mujoco.mjtGeom.mjGEOM_BOX, size=dims / 2, mass=0.5,
                       rgba=[1, 1, 1, 0], contype=0, conaffinity=0)
        self.m = spec.compile()
        self.d = mujoco.MjData(self.m)
        mujoco.mj_forward(self.m, self.d)
        self.box_geom = [self.m.geom(f"box_{n}").id for n in range(len(self.job))]
        self.box_mocap = [self.m.body_mocapid[self.m.body(f"box_{n}").id] for n in range(len(self.job))]
        self.probe = {k: (self.m.geom(f"probe_{k}").id, self.m.jnt_qposadr[self.m.body(f"probe_{k}").jntadr[0]],
                          self.m.jnt_dofadr[self.m.body(f"probe_{k}").jntadr[0]]) for k in {k for _, k in self.job}}
        self.park = self.d.qpos.copy()
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
        self._renderer = None

    def _container_xml(self):
        c = self.cfg
        D, W, H = c.depth / 1000, c.width / 1000, c.height / 1000
        ox, oy, oz = C_ORIGIN
        xm = ox - D / 2
        return f"""
<mujoco model="pack_container">
  <option timestep="0.002"/>
  <visual><global offwidth="1280" offheight="720"/></visual>
  <asset>
    <texture name="grid" type="2d" builtin="checker" rgb1=".85 .85 .85" rgb2=".75 .75 .75" width="300" height="300"/>
    <material name="grid" texture="grid" texrepeat="8 8"/>
  </asset>
  <worldbody>
    <light pos="{xm} {oy - W / 2} 2.5" dir="0 0 -1" directional="true"/>
    <geom name="ground" type="plane" size="3 3 0.1" material="grid" contype="0" conaffinity="0"/>
    <geom name="board" type="box" pos="{xm} {oy - W / 2} {oz - 0.0015}" size="{D / 2} {W / 2} 0.0015" rgba=".55 .56 .58 1"/>
    <geom name="wall_deep" type="box" pos="{ox + 0.0015} {oy - W / 2} {oz + H / 2}" size="0.0015 {W / 2} {H / 2}" rgba=".55 .56 .58 .4"/>
    <geom name="wall_side0" type="box" pos="{xm} {oy + 0.0015} {oz + H / 2}" size="{D / 2} 0.0015 {H / 2}" rgba=".55 .56 .58 .25"/>
    <geom name="wall_side1" type="box" pos="{xm} {oy - W - 0.0015} {oz + H / 2}" size="{D / 2} 0.0015 {H / 2}" rgba=".55 .56 .58 .25"/>
  </worldbody>
</mujoco>"""

    def _collide(self, gid, on: bool):
        """geom 하나의 충돌을 켜고 끈다. MuJoCo 는 몸체 단위 값(body_contype)으로 먼저 거르므로 둘 다 바꾼다."""
        m, v = self.m, int(on)
        b = m.geom_bodyid[gid]
        m.geom_contype[gid] = m.geom_conaffinity[gid] = v
        m.body_contype[b] = m.body_conaffinity[b] = v

    def _settle(self, k, o, center):
        """상자를 1 mm 위에서 놓고 settle_time 동안 굴려 본다. 반환: (이동 mm, 기울기 deg)."""
        m, d, c = self.m, self.d, self.cfg
        gid, qa, da = self.probe[k]
        d.qpos[:] = self.park
        d.qvel[:] = 0
        pos, _ = to_world(center + np.array([0, 0, 1.0]), (0, 0, 0))
        quat = np.array([1.0, 0, 0, 0]) if o == 0 else _quat_x90()
        d.qpos[qa:qa + 3], d.qpos[qa + 3:qa + 7] = pos, quat
        self._collide(gid, True)
        mujoco.mj_forward(m, d)
        for _ in range(int(c.settle_time / m.opt.timestep)):
            mujoco.mj_step(m, d)
        target, _ = to_world(center, (0, 0, 0))
        drift = float(np.linalg.norm(d.qpos[qa:qa + 3] - target) * 1000)
        q = d.qpos[qa + 3:qa + 7]
        tilt = float(np.degrees(2 * math.acos(min(1.0, abs(float(q @ quat))))))
        self._collide(gid, False)
        d.qpos[:] = self.park
        d.qvel[:] = 0
        mujoco.mj_forward(m, d)
        return drift, tilt

    def _show(self, n, p):
        """놓은 상자를 정적인(mocap) 상자로 남긴다. 이후 안정성 검사의 받침이 된다."""
        m, d = self.m, self.d
        pos, _ = to_world(p["center"], (0, 0, 0))
        d.mocap_pos[self.box_mocap[n]] = pos
        d.mocap_quat[self.box_mocap[n]] = [1, 0, 0, 0] if p["o"] == 0 else _quat_x90()
        self._collide(self.box_geom[n], True)
        mujoco.mj_forward(m, d)

    # ---------- 그리기 ----------
    def _set_cam(self, cam):
        cam.lookat[:], cam.distance, cam.azimuth, cam.elevation = (1.11, 0.28, 0.20), 1.7, -30.0, -35.0

    def render(self):
        if self.render_mode == "human":
            if self._viewer is None:
                self._viewer = mujoco.viewer.launch_passive(self.m, self.d)
                self._set_cam(self._viewer.cam)
            self._viewer.sync()
            return None
        if self.render_mode == "rgb_array":
            if self._renderer is None:
                self._renderer = mujoco.Renderer(self.m, 480, 640)
                self._cam = mujoco.MjvCamera()
                self._set_cam(self._cam)
            self._renderer.update_scene(self.d, self._cam)
            return self._renderer.render()
        return None

    def close(self):
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None


gym.register(id="ManiBoxPack-v0", entry_point=PackEnv)


# ---------- 기준 정책 ----------
def greedy(env: PackEnv):
    """안쪽(x 작은 쪽) -> 낮은 곳 -> y 작은 쪽 순으로 첫 자리. 학습 결과와 비교하는 기준선."""
    mask = env.action_masks().reshape(2, env.NX, env.NY)
    k = env.job[env.k][1]
    best = None
    for o in (0, 1):
        if not mask[o].any():
            continue
        _, base = env._feasible(env._dims(k, o))
        ii, jj = np.nonzero(mask[o])
        key = np.lexsort((jj, base[ii, jj], ii))[0]
        cand = (ii[key], base[ii[key], jj[key]], jj[key], o)
        best = cand if best is None or cand < best else best
    i, _, j, o = best
    return o * env.NX * env.NY + i * env.NY + j


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--policy", choices=["greedy", "random"], default="greedy")
    ap.add_argument("--render", choices=["none", "human"], default="none")
    ap.add_argument("--no-physics", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--check", action="store_true", help="gymnasium 규격 검사만 한다")
    ap.add_argument("--save-json", help="마지막 에피소드 배치를 저장 (컨테이너 mm + 로봇 좌표 m)")
    a = ap.parse_args()

    cfg = Config(physics=not a.no_physics)
    env = PackEnv(cfg, render_mode=None if a.render == "none" else a.render)
    if a.check:
        from gymnasium.utils.env_checker import check_env
        check_env(env, skip_render_check=True)
        print("gymnasium check_env 통과")
        return
    rng = np.random.default_rng(a.seed)
    for ep in range(a.episodes):
        obs, info = env.reset(seed=a.seed + ep)
        total, t0, done = 0.0, time.time(), False
        while not done:
            act = greedy(env) if a.policy == "greedy" else int(rng.choice(np.flatnonzero(info["action_mask"])))
            obs, r, done, _, info = env.step(act)
            total += r
            if a.render == "human":
                time.sleep(0.15)
        dt = time.time() - t0
        print(f"에피소드 {ep}: 지점 {env.n_stop}곳, 상자 {info.get('placed', env.k)}/{len(env.job)} 적재, "
              f"부피 사용률 {env.util * 100:.1f}%, 하차 순서 위반 {env.violations}, 보상 {total:.2f}, "
              f"종료 {info['reason']}, {dt / max(env.k, 1) * 1000:.0f} ms/상자")
    if a.save_json:
        out = []
        for p in env.export():
            c, s = to_planner(p)
            out.append({**p, "robot_center_m": c.round(4).tolist(), "robot_size_xyz_m": s.round(4).tolist()})
        Path(a.save_json).write_text(json.dumps({"frame": "container", "unit": "mm", "placements": out},
                                                ensure_ascii=False, indent=1))
        print(f"저장: {a.save_json}")
    if a.render == "human":
        print("창을 닫으면 끝납니다")
        while env._viewer is not None and env._viewer.is_running():
            time.sleep(0.1)
    env.close()


if __name__ == "__main__":
    main()
