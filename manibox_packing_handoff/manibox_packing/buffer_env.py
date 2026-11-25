"""버퍼가 있는 ManiBox 적재 환경 (Gymnasium) 과 로봇 이동 출력.

pack_env.PackEnv 를 넓힌 것이다. 장면·안정성 검사(_settle)·그리기는 PackEnv 것을 그대로 쓰고,
규칙과 진행은 buffer_core.Core 가 맡는다 (자리 규칙은 pack_env._feasible 과 같다).

  컨테이너 830 x 480 x 568 (mm) + 앞쪽 버퍼 2 x 2 칸 (투입 칸 1 + 재배치 버퍼 3)
  한 스텝 = 로봇 이동 하나 (투입 칸/버퍼 -> 컨테이너, 또는 투입 칸 -> 버퍼)

행동  Discrete(4 x 2 x NX x NY + 3)
  a < 4*2*NX*NY : 칸 s (0 투입 칸, 1~3 버퍼) 의 상자를 방향 o 로 격자 (i, j) 에 넣는다
                  a = s * (2*NX*NY) + o * (NX*NY) + i * NY + j
  그 뒤 3개    : 투입 칸 상자를 버퍼 칸 1~3 으로 옮긴다
관측  heightmap (NX, NY), slots (4, 9) 투입 칸·버퍼 상자, next (5, 9) 다음 상자, progress (2,)

robot_moves(core) 는 이동마다 시작점·끝점을 로봇 좌표(m)로 돌려준다 (pack_env.to_planner 와 같은 원점).
"""
from __future__ import annotations

import dataclasses
import json
import math

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

import pack_env
from pack_env import PackEnv, ROBOT_ORIGIN, STOP_COLORS, to_world
from buffer_core import BufferConfig, Core, PLACE, STAGE, RETRIEVE, SLOT_NAMES, make_job, buffer_cells, fast_feasible

# 상자 자세 (로봇 좌표 = 월드 축). 상자 축 a = L, b = W, c = H (흡착면 L x W 의 바깥 법선 = +c)
#   버퍼 칸: 흡착면이 위 (c = +z), L 이 x 축 방향
#   컨테이너: 흡착면이 출구 (c = 월드 -x). o=0: L 이 y 방향, W 가 위 / o=1: W 가 y 방향, L 이 위
R_FLAT = np.eye(3)
R_IN = {0: np.array([[0., 0., -1.], [-1., 0., 0.], [0., 1., 0.]]),
        1: np.array([[0., 0., -1.], [0., 1., 0.], [1., 0., 0.]])}
PRE_INSERT = 0.04     # 밀어 넣기 전 대기점: 끝점에서 출구 쪽으로 (m)
RAINBOW = [(0.86, 0.16, 0.16), (0.98, 0.55, 0.10), (0.97, 0.85, 0.15), (0.20, 0.66, 0.28), (0.15, 0.38, 0.88), (0.55, 0.25, 0.75)]


def _quat(R):
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.asarray(R, float).ravel())
    return q


def _r(v, n=4):
    return [round(float(x), n) for x in v]


def slot_pose(core: Core, slot: int, k: int) -> dict:
    """버퍼 칸에 흡착면을 위로 놓인 상자의 자세 (로봇 좌표, m)."""
    L, W, H = core.cfg.box_types[k]
    x, y = core.cells[slot]
    c, _ = to_world((x, y, H / 2), (0, 0, 0))
    return dict(slot=SLOT_NAMES[slot], center_m=_r(c - ROBOT_ORIGIN), size_xyz_m=_r(np.array([L, W, H]) / 1000),
                quat_wxyz=_r(_quat(R_FLAT)), grasp_m=_r(c + (0, 0, H / 2000) - ROBOT_ORIGIN), grasp_normal=[0, 0, 1])


def container_pose(center_mm, dims_xyz_mm, o: int) -> dict:
    """컨테이너 안 최종 자세. 흡착면 중심 = 출구 쪽 면의 중심, 바깥 법선 = 월드 -x."""
    c, s = to_world(center_mm, dims_xyz_mm)
    face = c + (-s[0] / 2, 0, 0)
    return dict(slot='container', center_m=_r(c - ROBOT_ORIGIN), size_xyz_m=_r(s), quat_wxyz=_r(_quat(R_IN[o])),
                grasp_m=_r(face - ROBOT_ORIGIN), grasp_normal=[-1, 0, 0],
                pre_insert_m=_r(face - (PRE_INSERT, 0, 0) - ROBOT_ORIGIN),
                container_mm=dict(center=_r(center_mm, 1), dims_xyz=list(dims_xyz_mm)))


def move_record(core: Core, mv: dict, step: int) -> dict:
    """Core.moves 의 한 항목 -> 모션 플래너 입력 (시작점·끝점, 로봇 좌표 m)."""
    k = mv['type']
    if mv['kind'] == 'retrieve':                    # 컨테이너에서 다시 꺼내 버퍼 칸으로 (넣기의 역순)
        rec = dict(step=step, box=f"b{mv['jid'] + 1:03d}", type=k, stop=mv['stop'], kind='retrieve',
                   **{'from': container_pose(mv['center'], mv['dims_xyz'], mv['o'])}, to=slot_pose(core, mv['dst'], k),
                   orientation=mv['o'], order_violations_removed=mv['unblocks'])
        if 'before' in mv:
            b = mv['before']
            rec['scene_before'] = dict(placed=b['placed'], slots={SLOT_NAMES[s]: f'b{j + 1:03d}' for s, j in enumerate(b['slots'])
                                                                  if j is not None})
        return rec
    rec = dict(step=step, box=f"b{mv['jid'] + 1:03d}", type=k, stop=mv['stop'], kind=mv['kind'],
               **{'from': slot_pose(core, mv['src'], k)})
    if mv['kind'] == 'stage':
        rec['to'] = slot_pose(core, mv['dst'], k)
    else:
        rec['to'] = container_pose(mv['center'], mv['dims_xyz'], mv['o'])
        rec['orientation'] = mv['o']
        rec['order_violations'] = mv['blocks']
    if 'before' in mv:      # 이 이동 직전 장면: 컨테이너에 실린 상자 수 (placements 앞에서부터) + 칸마다 있는 상자
        b = mv['before']
        rec['scene_before'] = dict(placed=b['placed'], slots={SLOT_NAMES[s]: f'b{j + 1:03d}' for s, j in enumerate(b['slots'])
                                                              if j is not None})
    return rec


def robot_moves(core: Core) -> list:
    """에피소드의 이동 목록 -> 모션 플래너 입력. (zone 에서 끝날 때 칸에 둔 상자 'keep' 은 이동이 아니라서 뺀다.)"""
    return [move_record(core, mv, n + 1) for n, mv in enumerate(m for m in core.moves if m['kind'] != 'keep')]


def kept_boxes(core: Core) -> list:
    """zone: 끝날 때 칸에 둔 채로 실은 상자 (로봇 좌표 자세)."""
    return [dict(box=f"b{m['jid'] + 1:03d}", type=m['type'], stop=m['stop'], **slot_pose(core, m['src'], m['type']))
            for m in core.moves if m['kind'] == 'keep']


def candidate_record(core: Core, k: int, step: int = 0) -> dict:
    """후보 k 를 실행하면 나올 이동 (move_record 형식). core 는 바꾸지 않는다."""
    c = core.cand
    before = dict(slots=list(core.slots), placed=len(core.placed))
    if int(c['kind'][k]) == RETRIEVE:
        p = core.placed[int(c['pidx'][k])]
        mv = dict(kind='retrieve', jid=p['jid'], type=p['type'], stop=p['stop'], o=p['o'], center=p['center'],
                  dims_xyz=list(p['dims_xyz']), dst=int(c['dst'][k]), unblocks=int(c['unblocks'][k]), before=before)
        return move_record(core, mv, step)
    src = int(c['src'][k])
    jid = core.slots[src]
    stop, typ = core.job[jid]
    if int(c['kind'][k]) == STAGE:
        mv = dict(kind='stage', jid=jid, type=typ, stop=stop, src=0, dst=int(c['dst'][k]), before=before)
    else:
        o, i, j = int(c['o'][k]), int(c['i'][k]), int(c['j'][k])
        info = core.place_info(src, o, i, j)
        mv = dict(kind='place', jid=jid, type=typ, stop=stop, src=src, o=o, center=np.round(info['center'], 2).tolist(),
                  dims_xyz=list(info['dims']), blocks=int(c['blocks'][k]), before=before)
    return move_record(core, mv, step)


def obstacles_world(core: Core, src: int | None = None, skip_placed: int | None = None) -> list:
    """모션 검사의 장애물: 실은 상자 (skip_placed 말고) + src 말고 칸에 있는 상자. [(중심 월드 m, 크기 m)]."""
    out = [to_world(p['center'], p['dims_xyz']) for n, p in enumerate(core.placed) if n != skip_placed]
    for s, jid in enumerate(core.slots):
        if jid is None or s == src:
            continue
        L, W, H = core.cfg.box_types[core.job[jid][1]]
        x, y = core.cells[s]
        out.append((to_world((x, y, H / 2), (0, 0, 0))[0], np.array([L, W, H], float) / 1000))
    return out


def check_motion(core: Core, k: int, motion) -> dict:
    """후보 k 의 모션 검사 (rig.motion.MotionChecker). 돌려줌 dict(ok, why, poses, record)."""
    rec = candidate_record(core, k)
    if int(core.cand['kind'][k]) == RETRIEVE:
        obs = obstacles_world(core, None, skip_placed=int(core.cand['pidx'][k]))
    else:
        obs = obstacles_world(core, int(core.cand['src'][k]))
    res = motion.check(rec, obs, ROBOT_ORIGIN)
    res['record'] = rec
    return res


class BufferPackEnv(PackEnv):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, config: BufferConfig | None = None, render_mode: str | None = None, job_kind: str = 'full'):
        cfg = config or BufferConfig()
        super().__init__(cfg, render_mode)
        self.core = Core(cfg)
        self.NX, self.NY = self.core.NX, self.core.NY      # zone 이면 깊이 1200 의 높이 지도
        self.job_kind = job_kind
        self.n_cells = self.NX * self.NY
        self.n_stage_actions = cfg.n_stage
        self.action_space = spaces.Discrete(len(SLOT_NAMES) * 2 * self.n_cells + cfg.n_stage + (cfg.max_boxes if cfg.retrieve else 0))
        feat = len(self.types) + 4
        self.observation_space = spaces.Dict({
            "heightmap": spaces.Box(0, 1, (self.NX, self.NY), np.float32),
            "slots": spaces.Box(0, 1, (len(SLOT_NAMES), feat), np.float32),
            "next": spaces.Box(0, 1, (cfg.lookahead, feat), np.float32),
            "progress": spaces.Box(0, 1, (2,), np.float32),
        })

    # ---------- 에피소드 ----------
    def reset(self, *, seed=None, options=None):
        gym.Env.reset(self, seed=seed)
        options = options or {}
        job = options.get("job") or make_job(self.np_random, self.cfg, options.get("kind", self.job_kind))
        self.core.reset(job)
        self.job, self.n_stop = self.core.job, self.core.n_stop
        self.k = 0
        self.placed, self.util, self.violations = self.core.placed, 0.0, 0
        self.hm = self.core.hm
        self._mask_cache = None
        if self.cfg.physics or self.render_mode is not None:
            self._build_scene()
            self._show_slots()
        if self.render_mode == "human":
            self.render()
        return self._obs(), {"action_mask": self.action_masks().copy()}

    def _feat_job(self, jid):
        f = np.zeros(len(self.types) + 4, np.float32)
        if jid is not None and jid < len(self.core.job):
            stop, k = self.core.job[jid]
            f[self.types.index(k)] = 1
            f[len(self.types):len(self.types) + 3] = np.array(self.cfg.box_types[k]) / self.dmax
            f[-1] = stop / self.core.n_stop
        return f

    def _obs(self):
        c = self.core
        return {
            "heightmap": (c.hm / self.cfg.height).astype(np.float32),
            "slots": np.stack([self._feat_job(x) for x in c.slots]),
            "next": np.stack([self._feat_job(c.next + t) for t in range(self.cfg.lookahead)]),
            "progress": np.array([min(1.0, c.next / len(c.job)), min(c.util, 1.0)], np.float32),
        }

    # ---------- 행동 ----------
    def decode(self, a):
        a = int(a)
        base = len(SLOT_NAMES) * 2 * self.n_cells
        if a >= base + self.n_stage_actions:          # 다시 꺼내기: placed 번호, 빈 칸 중 첫 칸으로
            cells = self.core.staging_slots()
            return RETRIEVE, a - base - self.n_stage_actions, 0, 0, 0, cells[0] if cells else -1
        if a >= base:
            return STAGE, 0, 0, 0, 0, a - base + 1
        s, r = divmod(a, 2 * self.n_cells)
        o, r = divmod(r, self.n_cells)
        i, j = divmod(r, self.NY)
        return PLACE, s, o, i, j, -1

    def encode(self, kind, src, o=0, i=0, j=0, dst=-1):
        if kind == RETRIEVE:                        # src = placed 번호
            return len(SLOT_NAMES) * 2 * self.n_cells + self.n_stage_actions + src
        if kind == STAGE:
            return len(SLOT_NAMES) * 2 * self.n_cells + dst - 1
        return src * 2 * self.n_cells + o * self.n_cells + i * self.NY + j

    def encode_cand(self, cand, k):
        """Core 후보 k 의 행동 번호."""
        kind = int(cand['kind'][k])
        if kind == RETRIEVE:
            return self.encode(RETRIEVE, int(cand['pidx'][k]))
        return self.encode(kind, int(cand['src'][k]), int(cand['o'][k]), int(cand['i'][k]), int(cand['j'][k]), int(cand['dst'][k]))

    def action_masks(self):
        """모든 칸에 대한 마스크 (pack_env 규칙 그대로; 하차 순서 위반은 벌점이라 막지 않는다)."""
        if self._mask_cache is not None:
            return self._mask_cache
        c = self.core
        mask = np.zeros(self.action_space.n, bool)
        for s, jid in enumerate(c.slots):
            if jid is None:
                continue
            _, k = c.job[jid]
            for o in c.orients(k):
                ok = c.slot_feasible(s, k, o)[0]
                start = s * 2 * self.n_cells + o * self.n_cells
                mask[start:start + self.n_cells] = ok.ravel()
        if c.slots[0] is not None:
            for d in c.staging_slots():
                mask[len(SLOT_NAMES) * 2 * self.n_cells + d - 1] = True
        for r, _, _ in c.retrievable():
            mask[self.encode(RETRIEVE, r)] = True
        self._mask_cache = mask
        return mask

    def step(self, action):
        c, cfg = self.core, self.cfg
        if not self.action_masks()[action]:
            return self._obs(), -cfg.w_fail, True, False, {**c.metrics(), "reason": "invalid_action",
                                                               "action_mask": self.action_masks().copy()}
        kind, s, o, i, j, dst = self.decode(action)
        info = {}
        if kind == PLACE and cfg.physics:
            p = c.place_info(s, o, i, j)
            drift, tilt = self._settle(p['type'], o, p['center'])
            info.update(drift_mm=drift, tilt_deg=tilt)
            if drift > cfg.drift_tol or tilt > cfg.tilt_tol:
                self._mask_cache = None
                return self._obs(), -cfg.w_fail, True, False, {**c.metrics(), **info, "reason": "unstable"}
        jid = c.placed[s]['jid'] if kind == RETRIEVE else c.slots[s]
        reward = c.do(kind, s, o, i, j, dst)
        self._mask_cache = None
        self.util, self.violations, self.k = c.util, c.violations, len(c.placed)
        if self.m is not None:
            if kind == RETRIEVE:                    # 컨테이너에서 빠졌다: 물리 검사에서 빼고 칸에 보인다
                self._collide(self.box_geom[jid], False)
            if kind == PLACE:
                self._show(jid, next(p for p in reversed(c.placed) if p['jid'] == jid))
            self._show_slots()
        terminated = c.done
        info.update(c.metrics(), action_mask=self.action_masks().copy())
        if self.render_mode == "human":
            self.render()
        return self._obs(), reward, terminated, False, info

    def skip_arrival(self):
        """투입 칸 상자를 빼고 (작업자가 꺼낸다, 다음 트럭으로) 이어 간다. 돌려줌: 끝났는지."""
        jid = self.core.slots[0]
        self.core.skip_arrival()
        self._mask_cache = None
        if self.m is not None:
            self.d.mocap_pos[self.box_mocap[jid]] = (5 + .3 * jid, 5, -1)
            self._show_slots()
        return self.core.done

    def finish(self):
        """zone: 적재를 마감한다 (Core.finish). 돌려줌: 끝났는지."""
        self.core.finish()
        self._mask_cache = None
        if self.m is not None:
            self._show_slots()
        return self.core.done

    def check_stable(self, src, o, i, j):
        """그 자리에 놓았을 때 MuJoCo 안정성 검사 (PackEnv._settle). (통과 여부, 이동 mm, 기울기 deg)."""
        p = self.core.place_info(src, o, i, j)
        drift, tilt = self._settle(p['type'], o, p['center'])
        return drift <= self.cfg.drift_tol and tilt <= self.cfg.tilt_tol, drift, tilt

    def export(self):
        return [{key: p[key] for key in ("id", "type", "stop", "center", "dims_xyz", "kept")} for p in self.core.placed]

    def robot_moves(self):
        return robot_moves(self.core)

    # ---------- 장면 ----------
    def _container_xml(self):
        c = self.cfg
        if c.zone:                                  # 바닥판·옆벽을 버퍼 구역까지 (1200) 늘린다
            self.cfg = dataclasses.replace(c, depth=self.core.D)
            try:
                xml = super()._container_xml()
            finally:
                self.cfg = c
        else:
            xml = super()._container_xml()
        geoms = []
        x0, x1 = c.depth, c.depth + 2 * c.cell + 3 * c.wall
        cells = buffer_cells(c)
        y0 = (c.width - (2 * c.cell + 3 * c.wall)) / 2
        wx = lambda x: pack_env.C_ORIGIN[0] - x / 1000  # noqa: E731
        wy = lambda y: pack_env.C_ORIGIN[1] - y / 1000  # noqa: E731
        z = pack_env.C_ORIGIN[2]
        zt = 0.0002 if c.zone else 0.0015           # zone: 흰 판은 바닥판 위 얇은 표시 (칸막이도 바닥 표시)
        geoms.append(f'<geom name="buffer_floor" type="box" pos="{wx((x0 + x1) / 2)} {wy(c.width / 2)} {z + zt if c.zone else z - zt}" '
                     f'size="{(x1 - x0) / 2000} {c.width / 2000} {zt}" rgba=".95 .95 .93 1" contype="0" conaffinity="0"/>')
        h = 0.001 if c.zone else 0.02
        used = sorted({0, *[s for s in c.stage_order if 0 < s < len(cells)]}) if c.stage_order else list(range(len(cells)))

        def wall(cx, cy, sx, sy):
            geoms.append(f'<geom type="box" pos="{wx(cx)} {wy(cy)} {z + h / 2}" size="{sx / 2000} {sy / 2000} {h / 2}" '
                         f'rgba=".35 .37 .4 1" contype="0" conaffinity="0"/>')
        if len(used) == len(cells):                 # 2 x 2 칸
            for t in range(3):
                wall(x0 + t * (c.cell + c.wall) + c.wall / 2, c.width / 2, c.wall, x1 - x0)
                wall((x0 + x1) / 2, y0 + t * (c.cell + c.wall) + c.wall / 2, x1 - x0, c.wall)
        else:                                       # 쓰는 칸만 (투입 칸 + 재배치 칸). zone 이면 나머지는 적재 공간이라 흰 판도 그 칸 아래만
            if c.zone:
                geoms.pop(0)
                for s in used:
                    cx, cy = cells[s]
                    geoms.append(f'<geom type="box" pos="{wx(cx)} {wy(cy)} {z + zt}" size="{(c.cell + 2 * c.wall) / 2000} '
                                 f'{(c.cell + 2 * c.wall) / 2000} {zt}" rgba=".95 .95 .93 1" contype="0" conaffinity="0"/>')
            for s in used:
                cx, cy = cells[s]
                d, n = c.cell / 2 + c.wall / 2, c.cell + 2 * c.wall
                wall(cx - d, cy, c.wall, n)
                wall(cx + d, cy, c.wall, n)
                wall(cx, cy - d, n, c.wall)
                wall(cx, cy + d, n, c.wall)
        ax, ay = cells[0]
        geoms.append(f'<geom name="arrival_mark" type="box" pos="{wx(ax)} {wy(ay)} {z + 0.0005}" size="{c.cell / 2000} {c.cell / 2000} 0.0005" '
                     f'rgba=".95 .85 .3 1" contype="0" conaffinity="0"/>')
        return xml.replace("</worldbody>", "".join(geoms) + "</worldbody>")

    def _build_scene(self):
        super()._build_scene()
        if self.core.n_stop <= len(RAINBOW):       # 지점 6곳 이하면 빨주노초파보
            for n, (stop, _) in enumerate(self.core.job):
                self.m.geom_rgba[self.box_geom[n], :3] = RAINBOW[stop - 1]

    def _show_slots(self):
        """투입 칸·버퍼 칸의 상자를 그 칸에 (흡착면 위로) 보이게 둔다."""
        if self.m is None:
            return
        c = self.core
        R = np.array([[0., 1., 0.], [0., 0., 1.], [1., 0., 0.]])   # 몸체 축 (H, L, W) -> 월드 (z, x, y)
        q = _quat(R)
        keep = [(m['src'], m['jid']) for m in c.moves if m['kind'] == 'keep']     # zone: 끝날 때 칸에 둔 채로 실은 상자
        for s, jid in [*enumerate(c.slots), *keep]:
            if jid is None:
                continue
            _, k = c.job[jid]
            L, W, H = self.cfg.box_types[k]
            x, y = c.cells[s]
            pos, _ = to_world((x, y, H / 2), (0, 0, 0))
            mid = self.box_mocap[jid]
            self.d.mocap_pos[mid] = pos
            self.d.mocap_quat[mid] = q
        mujoco.mj_forward(self.m, self.d)

    def _set_cam(self, cam):
        cam.lookat[:], cam.distance, cam.azimuth, cam.elevation = (1.02, 0.28, 0.16), 1.75, 22.0, -30.0


gym.register(id="ManiBoxBuffer-v0", entry_point=BufferPackEnv)


def main():
    import argparse
    ap = argparse.ArgumentParser(description="버퍼 환경 확인: 무작위(마스크 안) 정책으로 한 에피소드")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    env = BufferPackEnv(BufferConfig(physics=False))
    if a.check:
        from gymnasium.utils.env_checker import check_env
        check_env(env, skip_render_check=True)
        print("gymnasium check_env 통과")
        return
    obs, info = env.reset(seed=a.seed)
    rng = np.random.default_rng(a.seed)
    done = False
    while not done:
        act = int(rng.choice(np.flatnonzero(info["action_mask"])))
        obs, r, done, _, info = env.step(act)
    print({k: info[k] for k in ("util", "placed", "total", "violations", "stage_moves", "reason")})
    print(json.dumps(env.robot_moves()[:2], ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
