"""계획 파일(plan_moves.py 결과)의 이동마다 정해진 순서의 경유 자세(레일 + 관절 6개)를 만든다.

  uv run python heuristic_motion.py examples/minbuf/moves_big_104.json --out motion.json

place 이동 하나의 템플릿
  ready         운반 자세 (툴 앞 +x, 투입 칸 뒤쪽 위). 앞 이동이 끝난 자리에서 온다
  approach      "돌리기 경로"를 거꾸로 따라 내려가 윗면에 닿는다        -> pick 에서 흡착 켬
  flip          돌리기 경로: 레일을 뒤로 빼면서 들어 올리고 툴을 아래(-z)에서 앞(+x)으로 90° 돌린다
  transit       운반 자세 그대로 레일만 움직여 끝 자리 뒤쪽으로
  align         끝 자리 관절 자세로 보간 (아직 적재된 상자보다 출구 쪽)
  insert        레일만 전진해 밀어 넣는다 (관절 고정 = x 직선)          -> 흡착 끔
  retreat, unalign, ready   같은 길을 빈손으로 되짚어 운반 자세로
stage 이동 (투입 칸 -> 버퍼 칸): 투입 칸의 돌리기 경로로 들어 올린 뒤, 버퍼 칸의 돌리기 경로를 거꾸로 따라 내려놓는다.

왜 수직으로 들지 않나: 이 팔(거꾸로 단 Piper)은 툴을 아래로 둔 채로는 바닥에서 약 16 cm 위까지만 닿는다.
그래서 들어 올리려면 레일을 빼면서 툴을 같이 기울여야 한다. 돌리기 경로는 상자 종류마다 한 번만 찾아 두고 다시 쓴다
(장면이 바뀌어 막히면 그 장면에서 다시 찾는다). 찾는 방법은 7.5° 씩 기울이며 높이·앞뒤 위치·레일을 조금씩 바꿔 보는 것이다.

경유점 사이는 관절 공간 직선 보간(레일 포함 7축)으로 가정하고, 그 보간을 촘촘히 잘라
rig/ 모델로 팔·들고 가는 상자의 충돌을 본다. 상자는 흡착 순간의 툴 좌표에 붙은 것으로 본다.
돌리는 동안에는 상자가 월드 y 축 둘레로만 돌도록 관절 6 을 맞추고, 끝 자리에서는 계획(to.quat_wxyz)과 같게 맞춘다.

실행할 때 (ArUco 로 잰 상자 위치 반영): Templater.adjust_pick() 이 approach·flip 의 아래쪽 경유점을 잰 위치에 맞게 다시 푼다.
rig/ 모델(레일 0 = 베이스 x 0.2 m, 툴 80 mm)은 가정값이다. 실물 치수가 다르면 결과도 다시 만들어야 한다.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import mujoco
import numpy as np

from pack_env import BOX_TYPES, to_world
from rig import model as M
from rig.motion import MotionChecker

DOWN, FWD = np.array([0., 0., -1.]), np.array([1., 0., 0.])
PUSH = .04                # 밀어 넣는 거리 (계획의 pre_insert 와 같다)
BACKOFF = (.20, .14, .08, .04, .0, .30, .45, .65, .90)   # 끝 자리 관절 자세를 잡을 때 대기점보다 더 뒤로 빼는 거리 후보 (앞에서부터 시도)
Q_STEP, RAIL_STEP = .03, .008          # 충돌 검사 간격 (rad, m)
INSERT_SEEDS = 10         # 끝 자리 IK 를 풀 때 더 써 보는 무작위 씨앗 수
FLIP_STEPS = 12           # 돌리기 90° 를 나누는 수 (7.5°)
FLIP_JUMP = .45           # 돌리기 한 단계에서 관절이 움직여도 되는 최대 (rad). 넘으면 다른 해로 건너뛴 것
FLIP_BEAM, FLIP_TRY = 30, 6   # 단계마다 남기는 갈래 수, 그중 운반 자세로 넘어가 볼 갈래 수
FLIP_MIN_TILT = 20.       # 이만큼(도) 기운 뒤부터 운반 자세로 넘어가 본다
ADJUST_BELOW = 30.        # adjust_pick: 툴 기울기가 이 각도(도) 아래인 경유점만 고친다


def _mat(quat):
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, np.asarray(quat, float))
    return R.reshape(3, 3)


def _quat(R):
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.asarray(R, float).ravel())
    return q


def _rotz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])


def _tilt(th):
    """월드 y 축 둘레 회전: 툴 축 -z 를 +x 쪽으로 th 만큼 기울인다."""
    c, s = math.cos(th), math.sin(th)
    return np.array([[c, 0., -s], [0., 1., 0.], [s, 0., c]])


def _wrap(a, period):
    return a - period * math.ceil(a / period - .5)


class Templater:
    def __init__(self, plan: dict, seed: int = 0):
        self.plan = plan
        self.origin = np.array(plan['robot_origin_world_m'])
        self.mc = MotionChecker(n_boxes=len(plan['placements']) + 10, seed=seed)
        self.lo, self.hi = self.mc.lo, self.mc.hi
        self.cells = {k: np.array(v) + self.origin for k, v in plan['buffer_cells_robot_m'].items()}
        self.types = {f'b{n + 1:03d}': int(k) for n, (_, k) in enumerate(plan['job'])}
        self.box_types = {int(k): tuple(v) for k, v in plan['box_types'].items()} if 'box_types' in plan else BOX_TYPES
        self.placed = [to_world(p['center'], p['dims_xyz']) for p in plan['placements'] if not p.get('kept')]
        self.floor = float(self.cells['arrival'][2])
        self.tall = max(t[2] for t in self.box_types.values()) / 1000
        self.wide = max(max(t[0], t[1]) for t in self.box_types.values()) / 1000
        self.flips = {}           # (칸, 상자 치수) -> 돌리기 경로
        self.n_flip_search = 0

    # ---------- 기본 ----------
    def fk(self, rail, q):
        self.mc.d.qpos[self.mc.rail] = rail
        return self.mc._fk(q)

    def frame0(self, rail, q):
        """관절 6 = 0 일 때 툴 자세 A. 실제 툴 자세 = A · Rz(q6)."""
        q0 = np.array(q, float)
        q0[5] = 0
        return self.fk(rail, q0)[1]

    def hit(self, rail, q, carry):
        mc = self.mc
        if carry is None:
            mc.clear_carried()
        else:
            p, R = self.fk(rail, q)
            mc.set_carried(p + R[:, 2] * carry['size'][2] / 2, carry['size'], _quat(R @ carry['R_rel']))
        return mc.collides(rail, q)

    def seg(self, a, b, carry):
        """(rail, q) a -> b 관절 공간 직선 보간이 충돌 없으면 None, 있으면 (보간 위치, geom 쌍)."""
        (ra, qa), (rb, qb) = a, b
        n = max(1, math.ceil(max(float(np.abs(qb - qa).max()) / Q_STEP, abs(rb - ra) / RAIL_STEP)))
        for t in np.linspace(0, 1, n + 1):
            bad, pair = self.hit(ra + t * (rb - ra), qa + t * (qb - qa), carry)
            if bad:
                return round(float(t), 2), pair
        return None

    def line_back(self, rail, q, p, dist, step=.01):
        """툴 끝을 p 에서 -x 로 dist 만큼 뺀 직선 (레일 고정, 툴 +x 유지). [가장 뒤 q, ..., p 바로 앞 q] 또는 None."""
        out = []
        for d in np.arange(step, dist + 1e-9, step):
            q2 = self.mc.ik(p - np.array([d, 0., 0.]), FWD, rail, q, iters=60)
            if q2 is None or float(np.abs(q2[:5] - q[:5]).max()) > .3:
                return None
            q2[5] = q[5]
            out.append(q2)
            q = q2
        return out[::-1]

    def align_vias(self, T, P, target):
        """운반 자세 T -> 끝 자리 자세 P 사이에 넣어 볼 중간 자세들 (같은 레일). 앞에서부터 시도한다."""
        ra, qt = T
        qp = P[1]
        yield []                                             # 바로 간다
        yield [(ra, np.r_[qp[:3], qt[3:]])]                  # 팔 먼저, 손목 나중
        yield [(ra, np.r_[qt[:3], qp[3:]])]                  # 손목 먼저, 팔 나중
        p_t, p_p = self.fk(ra, qt)[0], self.fk(ra, qp)[0]
        for p in (np.array([p_t[0], p_p[1], p_p[2]]),        # 뒤쪽에서 끝 자리의 폭·높이를 먼저 맞춘다
                  np.array([p_t[0], p_t[1], p_p[2]]),        # 높이만 먼저
                  np.array([p_t[0], p_p[1], p_t[2]])):       # 폭만 먼저
            for seed in (qp, qt):
                qm = self.mc.ik(p, FWD, ra, seed, iters=60)
                if qm is not None:
                    yield [(ra, np.r_[qm[:5], qp[5]])]

    def path_hit(self, pts, carry):
        """경유점 열 [(rail, q), ...] 을 따라가며 검사. 통과면 None."""
        for x, y in zip(pts, pts[1:]):
            bad = self.seg(x, y, carry)
            if bad:
                return bad
        return None

    def scene(self, mv, extra=(), skip=()):
        """이동 직전 장면: 실린 상자 + 다른 칸 상자 (+ extra). skip 칸의 상자는 뺀다."""
        sb = mv['scene_before']
        obs = list(self.placed[:sb['placed']])
        for slot, box in sb['slots'].items():
            if slot == mv['from']['slot'] or slot in skip:
                continue
            L, W, H = self.box_types[self.types[box]]
            c = self.cells[slot].copy()
            c[2] = self.floor + H / 2000
            obs.append((c, np.array([L, W, H], float) / 1000))
        obs.extend(extra)
        self.mc.set_obstacles(obs)

    def target_box(self, mv):
        """집을 상자 (칸에 놓인 상태)."""
        return np.array(mv['from']['center_m']) + self.origin, np.array(mv['from']['size_xyz_m'], float)

    # ---------- 돌리기 경로 ----------
    def flip_search(self, grasp, size):
        """지금 장면에서 grasp(윗면 중심, 월드) 에 놓인 상자(size = 칸에서의 x·y·z)를 집어 돌리는 경로를 찾는다.
        1) 집는 자세에서 7.5° 씩 기울이며 들어 올린다 (레일·높이·앞뒤를 조금씩 바꿔 보며 여러 갈래를 같이 키운다)
        2) 충분히 뜬 갈래에서 운반 자세(툴 +x)로 관절 보간해 넘어갈 수 있는지 본다 (이 팔은 끝까지 기울이면 손목 한계에 걸린다)
        돌려줌: [(rail, q)] (0 번 = 집는 자세 툴 아래, 마지막 = 운반 자세). 관절 6 은 집을 때 0 기준 보정값. 없으면 None."""
        mc = self.mc
        self.n_flip_search += 1
        ths = np.linspace(0, math.pi / 2, FLIP_STEPS + 1)
        rng = np.random.default_rng(3)

        def wrist_fix(rail, q, th, A0):
            """상자가 월드 y 둘레로만 돈 자세가 되게 관절 6 을 맞춘다."""
            Mz = self.frame0(rail, q).T @ _tilt(th) @ A0
            c = math.atan2(Mz[1, 0], Mz[0, 0])
            if Mz[2, 2] < .999 or not (self.lo[5] + .3 <= c <= self.hi[5] - .3):
                return None
            q = q.copy()
            q[5] = c
            return q

        # 운반 자세 후보: 툴 +x, 베이스 앞 dx, 높이 z
        mc.clear_carried()
        cands = []
        for rail in (0., .06, .12, .215):
            for dx in (.22, .30, .16):
                for z in (.36, .28, .44):
                    for y in (M.BASE_Y, float(grasp[1])):
                        for _ in range(3):
                            q = mc.ik(np.array([M.BASE_X0 + rail + dx, y, z]), FWD, rail, rng.uniform(self.lo, self.hi))
                            if q is not None and all(abs(rail - r) > 1e-6 or float(np.abs(q[:5] - o[:5]).max()) > .2 for r, o in cands):
                                cands.append((rail, q))

        # 갈래: (rail, q, x, z, 부모 번호). 뿌리 = 집는 자세
        nodes, level = [], []
        for r0 in (.215, .15, .28, .10, .35):
            for _ in range(6):
                q0 = mc.ik(grasp, DOWN, r0, rng.uniform(self.lo, self.hi))
                if q0 is None:
                    continue
                q0[5] = 0
                if any(abs(r0 - nodes[i][0]) < 1e-6 and float(np.abs(q0 - nodes[i][1]).max()) < .1 for i in level):
                    continue
                carry = dict(size=size, R_rel=self.frame0(r0, q0).T)     # 칸에서 상자 축 = 월드 축
                if not self.hit(r0, q0, carry)[0]:
                    nodes.append((r0, q0, float(grasp[0]), float(grasp[2]), -1, carry, self.frame0(r0, q0)))
                    level.append(len(nodes) - 1)

        def trace(i):
            out = []
            while i >= 0:
                out.append((nodes[i][0], nodes[i][1]))
                i = nodes[i][4]
            return out[::-1]

        for k in range(FLIP_STEPS - 1):
            th = float(ths[k + 1])
            axis = np.array([math.sin(th), 0., -math.cos(th)])
            zmin = self.floor + size[2] * math.cos(th) + size[0] / 2 * math.sin(th) + .006   # 기운 상자 모서리가 바닥에서 뜨는 높이
            seen, nxt = set(), []
            for i in level:
                rail, q, x, z, _, carry, A0 = nodes[i]
                for dz in (0., .02, .04, .07):
                    for dx in (0., -.02, -.04, .02):
                        for dr in (-.03, 0., -.06, .03):
                            r2 = float(np.clip(rail + dr, 0, M.TRAVEL))
                            x2, z2 = x + dx, max(z, zmin) + dz
                            key = (round(r2, 3), round(x2, 3), round(z2, 3), i if len(level) < 8 else 0)
                            if key in seen:
                                continue
                            seen.add(key)
                            q2 = mc.ik(np.array([x2, grasp[1], z2]), axis, r2, q, iters=60)
                            if q2 is None or float(np.abs(q2[:5] - q[:5]).max()) > FLIP_JUMP:
                                continue
                            q2 = wrist_fix(r2, q2, th, A0)
                            if q2 is None or self.seg((rail, q), (r2, q2), carry):
                                continue
                            nodes.append((r2, q2, x2, z2, i, carry, A0))
                            nxt.append(len(nodes) - 1)
            if not nxt:
                break
            nxt.sort(key=lambda j: -nodes[j][3])                  # 높이 뜬 갈래부터
            level = nxt[:FLIP_BEAM]
            if th < math.radians(FLIP_MIN_TILT):
                continue
            for j in level[:FLIP_TRY]:
                rail, q, x, z, _, carry, A0 = nodes[j]
                for rc, qc in cands:
                    qc = wrist_fix(rc, qc, math.pi / 2, A0)
                    if qc is None or self.hit(rc, qc, carry)[0] or self.seg((rail, q), (rc, qc), carry):
                        continue
                    return trace(j) + [(rc, qc)]
        return None

    def flip_for(self, slot, grasp, size):
        """캐시한 돌리기 경로가 지금 장면에서도 통하면 그대로, 아니면 다시 찾는다."""
        key = (slot, tuple(np.round(size, 4)))
        path = self.flips.get(key)
        if path is not None:
            carry = dict(size=size, R_rel=self.frame0(*path[0]).T)
            if self.path_hit(path, carry) is None:
                return path
        path = self.flip_search(grasp, size)
        if path is not None:
            self.flips[key] = path
        return path

    # ---------- 손목 각도 ----------
    def wrist_delta(self, A_pick, A_here, R_from, R_to):
        """(끝 자리 관절 6) - (집을 때 관절 6). 툴 축이 안 맞으면 None."""
        Mz = A_here.T @ R_to @ R_from.T @ A_pick
        return math.atan2(Mz[1, 0], Mz[0, 0]) if Mz[2, 2] > .999 else None

    def split_wrist(self, d, period, comp):
        """집을 때 관절 6 = a, 끝 자리 = b = a + d (+ period 의 배수). 돌리기 보정값 comp 를 더해도 범위 안이고 0 에 가깝게."""
        lo, hi = self.lo[5] + .05, self.hi[5] - .05
        best = None
        for a in np.arange(-2., 2.001, .05):
            if not all(lo <= a + c <= hi for c in comp):
                continue
            for n in (-2, -1, 0, 1, 2):
                b = a + d + n * period
                if lo <= b <= hi:
                    cost = max(abs(b), max(abs(a + c) for c in comp))
                    if best is None or cost < best[0]:
                        best = (cost, float(a), float(b))
        return (best[1], best[2]) if best else None

    # ---------- 이동 ----------
    def plan_move(self, mv: dict, prev=None) -> dict:
        """prev: 앞 이동이 끝난 (rail, q) (없으면 이 이동의 운반 자세에서 시작)."""
        if mv['kind'] not in ('place', 'stage'):
            return dict(ok=False, why=f"kind {mv['kind']} 는 지원하지 않는다", waypoints=[])
        o = self.origin
        fr, to = mv['from'], mv['to']
        size = np.array(fr['size_xyz_m'], float)
        R_from, R_to = _mat(fr['quat_wxyz']), _mat(to['quat_wxyz'])
        period = math.pi / 2 if abs(size[0] - size[1]) < 1e-6 else math.pi
        grasp = np.array(fr['grasp_m']) + o
        self.scene(mv)
        flip = self.flip_for(fr['slot'], grasp, size)
        if flip is None:
            return dict(ok=False, why='flip: 경로 없음', waypoints=[])
        A_pick = self.frame0(*flip[0])
        comp = [float(q[5]) for _, q in flip]
        if mv['kind'] == 'stage':
            return self._stage(mv, flip, prev, size)

        mc = self.mc
        target = np.array(to['grasp_m']) + o
        rc, qc = flip[-1]
        ref = mv.get('ik', {}).get('insert')
        rails = []
        for r in ([float(ref['rail_m'])] if ref else []) + [target[0] - dx - M.BASE_X0 for dx in (.30, .22, .38, .15, .45, .52, .08)]:
            r = float(np.clip(r, 0, M.TRAVEL))
            if all(abs(r - x) > .02 for x in rails):
                rails.append(r)
        rng = np.random.default_rng(mv['step'])
        seeds = [qc] + ([np.array(ref['q_rad'], float)] if ref else []) + [rng.uniform(self.lo, self.hi) for _ in range(INSERT_SEEDS)]
        why, tried = 'insert: ik', {}
        for r in rails:
            for s in seeds:
                qi = mc.ik(target, FWD, r, s)
                if qi is None or any(float(np.abs(qi[:5] - t).max()) < .05 for t in tried.setdefault(r, [])):
                    continue
                tried[r].append(qi[:5].copy())
                d = self.wrist_delta(A_pick, self.frame0(r, qi), R_from, R_to)
                ab = self.split_wrist(d, period, comp) if d is not None else None
                if ab is None:
                    why = 'insert: wrist'
                    continue
                a, b = ab
                qi = np.r_[qi[:5], b]
                fl = [(rr, np.r_[q[:5], q[5] + a]) for rr, q in flip]
                carry = dict(size=size, R_rel=(A_pick @ _rotz(a)).T @ R_from)
                self.scene(mv)
                bad = self.hit(r, qi, carry)
                if bad[0]:
                    why = f'insert: collision {bad[1][0]}-{bad[1][1]}'
                    continue
                for back in BACKOFF if r >= PUSH else (None,):
                    if back is None:
                        push = self.line_back(r, qi, target, PUSH)
                        if push is None:
                            why = 'insert: 레일 여유 없음 + 팔로 미는 직선 ik 실패'
                            continue
                        ra, pre = r, push[0]
                    else:
                        ra, pre, push = r - PUSH - back, qi, []
                        if ra < 0:
                            continue
                    C, T, P, I = fl[-1], (ra, fl[-1][1]), (ra, pre), (r, qi)
                    ins = [P] + [(r, q) for q in push[1:]] + [I]
                    new_box = (np.array(to['center_m']) + o, np.array(to['size_xyz_m'], float))
                    legs = None
                    for via in self.align_vias(T, P, target):
                        al = [T] + via + [P]
                        cand = ([('transit', C, T)] + [('align', x, y) for x, y in zip(al, al[1:])]
                                + [('insert', x, y) for x, y in zip(ins, ins[1:])])
                        self.scene(mv)
                        fail = next(((n, h) for n, x, y in cand if (h := self.seg(x, y, carry))), None)
                        if fail is None:
                            self.scene(mv, extra=[new_box])        # 다음 상자는 로봇이 ready 로 돌아온 뒤에 투입한다
                            names = dict(insert='retreat', align='unalign', transit='ready')
                            back_legs = [(names[n], y, x) for n, x, y in reversed(cand)]
                            fail = next(((n, h) for n, x, y in back_legs if (h := self.seg(x, y, None))), None)
                        if fail is None:
                            legs = cand
                            break
                        why = f'{fail[0]}: collision {fail[1][1][0]}-{fail[1][1][1]}'
                    if legs is None:
                        continue
                    wp = self._pick_waypoints(mv, fl, prev)
                    if wp is None:
                        why = 'ready: 앞 이동 끝 자세에서 오는 길이 막힘'
                        continue
                    wp += [dict(name=n, rail=y[0], q=y[1], suction=1) for n, x, y in legs]
                    wp[-1]['suction'] = 0              # 다 밀어 넣은 뒤 흡착 끔
                    wp += [dict(name=n, rail=y[0], q=y[1], suction=0) for n, x, y in back_legs]
                    return dict(ok=True, why='', waypoints=wp, backoff_m=back, end=C)
        return dict(ok=False, why=why, waypoints=[])

    def _pick_waypoints(self, mv, fl, prev):
        """ready -> approach(돌리기 경로 거꾸로, 빈손) -> pick(흡착 켬) -> flip(돌리기 경로, 상자 들고)."""
        C = fl[-1]
        wp = []
        if prev is not None and (abs(prev[0] - C[0]) > 1e-6 or float(np.abs(prev[1] - C[1]).max()) > 1e-6):
            self.scene(mv, extra=[self.target_box(mv)])
            if self.seg(prev, C, None):
                return None
            wp.append(dict(name='start', rail=prev[0], q=prev[1], suction=0))
        wp.append(dict(name='ready', rail=C[0], q=C[1], suction=0))
        wp += [dict(name='approach', rail=r, q=q, suction=0) for r, q in reversed(fl[:-1])]
        wp[-1].update(name='pick', suction=1)          # 닿은 뒤 흡착 켬
        wp += [dict(name='flip', rail=r, q=q, suction=1) for r, q in fl[1:]]
        return wp

    def _stage(self, mv, flip, prev, size):
        """투입 칸 -> 버퍼 칸. 투입 칸의 돌리기 경로로 올리고, 버퍼 칸의 돌리기 경로를 거꾸로 따라 내려놓는다."""
        to = mv['to']
        drop = np.array(to['grasp_m']) + self.origin
        self.scene(mv)
        flip2 = self.flip_for(to['slot'], drop, size)
        if flip2 is None:
            return dict(ok=False, why='stage: 버퍼 칸 돌리기 경로 없음', waypoints=[])
        carry = dict(size=size, R_rel=self.frame0(*flip[0]).T)
        wp = self._pick_waypoints(mv, flip, prev)
        if wp is None:
            return dict(ok=False, why='ready: 앞 이동 끝 자세에서 오는 길이 막힘', waypoints=[])
        self.scene(mv)
        if self.seg(flip[-1], flip2[-1], carry) or self.path_hit(flip2, carry):
            return dict(ok=False, why='stage: 옆 칸으로 옮기는 길이 막힘', waypoints=[])
        wp += [dict(name='over', rail=flip2[-1][0], q=flip2[-1][1], suction=1)]
        wp += [dict(name='lower', rail=r, q=q, suction=1) for r, q in reversed(flip2[:-1])]
        wp[-1].update(name='drop', suction=0)
        wp += [dict(name='rise', rail=r, q=q, suction=0) for r, q in flip2[1:]]
        wp[-1]['name'] = 'ready'
        return dict(ok=True, why='', waypoints=wp, end=flip2[-1])

    # ---------- 실행할 때: 잰 상자 위치 반영 ----------
    def adjust_pick(self, waypoints, dx: float, dy: float, yaw: float = 0.):
        """ArUco 로 잰 상자가 계획 위치에서 (dx, dy) m, yaw rad (월드 z 둘레) 만큼 어긋나 있을 때의 경유점을 돌려준다.
        approach·pick·flip 중 툴 기울기가 ADJUST_BELOW 도보다 작은 점만 고친다 (기울기에 따라 0 -> 1 로 섞는다).
        집는 점은 잰 위치 그대로이고, 올라가면서 계획 경로로 돌아오므로 그 뒤 경유점은 그대로 쓴다.
        waypoints: plan_move 가 준 목록 (dict 의 rail/q 또는 rail_m/q_rad). 못 풀면 None."""
        out = []
        lim = math.radians(ADJUST_BELOW)
        for w in waypoints:
            rail = float(w.get('rail', w.get('rail_m')))
            q = np.array(w.get('q', w.get('q_rad')), float)
            w2 = dict(w)
            if w['name'] in ('approach', 'pick', 'flip'):
                p, R = self.fk(rail, q)
                th = math.acos(float(np.clip(-R[2, 2], -1, 1)))      # 툴 축이 아래에서 기운 각
                k = max(0., 1 - th / lim)
                if k > 0:
                    q2 = self.mc.ik(p + k * np.array([dx, dy, 0.]), R[:, 2], rail, q, iters=60)
                    if q2 is None:
                        return None
                    # 팔이 옮겨 가며 생긴 툴 회전까지 합쳐, 툴이 원래 자세에서 월드 z 둘레로 k·yaw 만큼만 돈 자세가 되게 한다
                    Mz = self.frame0(rail, q2).T @ _rotz(k * yaw) @ R
                    q2[5] = np.clip(math.atan2(Mz[1, 0], Mz[0, 0]), self.lo[5], self.hi[5])
                    key = 'q' if 'q' in w else 'q_rad'
                    w2[key] = q2 if key == 'q' else [round(float(v), 4) for v in q2]
            out.append(w2)
        return out

    # ---------- 검사 ----------
    def check(self, mv, wp):
        """저장할 경유점 목록 전체를 처음부터 다시 따라가며 검사한다. 통과면 None, 아니면 이유."""
        k = next(i for i, w in enumerate(wp) if w['name'] == 'pick')
        R = self.fk(wp[k]['rail'], np.asarray(wp[k]['q'], float))[1]
        carry = dict(size=np.array(mv['from']['size_xyz_m'], float), R_rel=R.T @ _mat(mv['from']['quat_wxyz']))
        to = mv['to']
        new_box = (np.array(to['center_m']) + self.origin, np.array(to['size_xyz_m'], float))
        self.scene(mv)
        released = False
        for i, (x, y) in enumerate(zip(wp, wp[1:])):
            holding = bool(x['suction'])               # x 에 도착한 뒤의 흡착 상태로 x -> y 를 간다
            if i >= k and not holding and not released:
                self.scene(mv, extra=[new_box])
                released = True
            bad = self.seg((x['rail'], np.asarray(x['q'], float)), (y['rail'], np.asarray(y['q'], float)),
                           carry if holding else None)
            if bad:
                return f"{y['name']}: collision {bad[1][0]}-{bad[1][1]}", None
        # 놓는 순간 상자 중심·치수 방향이 계획과 맞는지
        j = max(i for i, w in enumerate(wp) if w['suction']) + 1
        p, R = self.fk(wp[j]['rail'], np.asarray(wp[j]['q'], float))
        ec = float(np.linalg.norm(p + R[:, 2] * carry['size'][2] / 2 - new_box[0]))
        es = float(np.abs(np.abs(R @ carry['R_rel']) @ carry['size'] - new_box[1]).max())
        return (None if ec < .002 and es < .002 else f'placed error {ec * 1000:.1f} / {es * 1000:.1f} mm'), (ec, es)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('plan', help='plan_moves.py 가 만든 moves.json')
    ap.add_argument('--out', default='motion.json')
    a = ap.parse_args()
    plan = json.loads(Path(a.plan).read_text())
    tp = Templater(plan)
    out, fails, backs = [], [], Counter()
    worst = np.zeros(2)
    prev = None
    for mv in plan['moves']:
        res = tp.plan_move(mv, prev)
        if res['ok']:
            bad, err = tp.check(mv, res['waypoints'])
            if err:
                worst = np.maximum(worst, err)
            if bad:
                res = dict(ok=False, why='check ' + bad, waypoints=[])
        if res['ok']:
            prev = res['end']
            backs[res.get('backoff_m')] += 1
        else:
            fails.append((mv['step'], mv['kind'], res['why']))
        out.append(dict(step=mv['step'], box_id=mv.get('box_id'), kind=mv['kind'], ok=res['ok'], why=res['why'],
                        waypoints=[dict(name=w['name'], rail_m=round(float(w['rail']), 4),
                                        q_rad=[round(float(v), 4) for v in w['q']], suction=int(w['suction']))
                                   for w in res['waypoints']]))
    doc = dict(source=str(a.plan), model=dict(base_x0_m=M.BASE_X0, tool_length_m=M.TOOL_LENGTH),
               note='waypoints 를 순서대로 간다. 사이는 관절 공간 직선 보간 (레일 + 관절 6). '
                    'suction = 그 점에 도착한 뒤의 흡착 상태 (1 켬, 0 끔)',
               moves=out)
    Path(a.out).write_text(json.dumps(doc, ensure_ascii=False))
    n = len(out)
    print(f'{n - len(fails)} / {n} 이동에 경로를 만들었다 -> {a.out}')
    print(f'놓는 순간 오차 최대: 중심 {worst[0] * 1000:.2f} mm, 치수 방향 {worst[1] * 1000:.2f} mm')
    print(f'돌리기 경로를 찾은 횟수 {tp.n_flip_search} (종류 {len(tp.flips)}), 뒤로 뺀 거리별 이동 수 {dict(backs)}')
    if fails:
        print('실패 이유:', dict(Counter(w.split(' collision')[0] for _, _, w in fails).most_common(8)))
        for f in fails[:12]:
            print('  ', f)


if __name__ == '__main__':
    main()
