"""모션 검사: 이동 하나 (시작점·끝점) 를 R17 + Piper 모델로 IK 와 충돌 검사한다 (운동학만, 경로는 보지 않는다).

이동마다 보는 자세 (move_record 의 점 그대로)
  pick    시작 칸 상자 윗면 중심, 툴 아래 (-z)                 들고 갈 상자는 칸에 있다
  flip    (선택, flip=True) 집은 자리에서 들어 올리며 툴을 -z -> +x 로 90° 돌리는 자세들 (check_flip)
  pre     끝 자리 앞면 중심에서 출구 쪽 40 mm, 툴 월드 +x         들고 가는 상자도 출구 쪽으로 40 mm
  insert  끝 자리 앞면 중심, 툴 월드 +x                         들고 가는 상자는 끝 자리
  drop    (stage) 버퍼 칸 상자 윗면 중심, 툴 아래
충돌: 팔(링크·툴) 이나 들고 가는 상자가 프레임·바닥·다른 상자를 1.5 mm 넘게 파고들면 실패.
다른 상자 = 컨테이너에 실은 상자 + 다른 칸 (투입 칸·버퍼) 의 상자. 팔과 들고 가는 상자끼리는 보지 않는다.
컵은 둥글어서 툴 축 둘레 회전은 자유로 둔다.

IK 는 numpy 감쇠 최소제곱 (관절 범위 안). 레일 위치를 몇 개 바꿔 가며 푼다.
이 모델은 이 저장소 기준이다 (레일 0 = 베이스 x 0.2 m, 툴 80 mm). 실제 판단은 모션 플래너 모델로 하고,
안 되는 이동은 OnlinePlanner.reject() 로 다음 순위를 받는다.
"""
from __future__ import annotations

import numpy as np
import mujoco

from . import model as M

TEAM_RAIL0_X = .700              # 팀 레일 0 = 베이스 x 0.70 (pack_env ROBOT_ORIGIN). 출력 ik 의 rail_team_m = 베이스 x - 0.70

W_ROT = .15


class MotionChecker:
    def __init__(self, n_boxes: int = 420, tool: float = M.TOOL_LENGTH, tol: float = .0015, clearance: float = 0.0, seed: int = 0,
                 flip: bool = False):
        """tol: 이만큼 파고드는 접촉은 봐준다 (맞닿은 면). clearance > 0: 팔이 이만큼 떨어져 있어야 한다 (도달 표용, 보수적).
        flip: check() 에서 집은 뒤 뒤집기 자세들(check_flip, 단계마다 따로 푼 필요조건)도 본다. 기본은 끔."""
        self.flip = flip
        self.m = mujoco.MjModel.from_xml_string(M.build(n_boxes, tool))
        self.d = mujoco.MjData(self.m)
        m = self.m
        self.tool, self.tol, self.clearance = tool, tol, clearance
        self.rail = m.joint('rail').qposadr[0]
        self.arm = np.array([m.joint(j).qposadr[0] for j in M.ARM_JOINTS])
        self.arm_dof = np.array([m.joint(j).dofadr[0] for j in M.ARM_JOINTS])
        self.lo, self.hi = m.jnt_range[[m.joint(j).id for j in M.ARM_JOINTS]].T.copy()
        self.tcp = m.site('tcp').id
        root = m.body('carriage').id
        robot = {b for b in range(m.nbody) if m.body_rootid[b] == root}
        self.robot_geoms = np.array([m.geom_bodyid[g] in robot for g in range(m.ngeom)])
        if clearance > 0:
            m.geom_margin[self.robot_geoms] = clearance       # 이 거리 안이면 접촉이 생긴다
        self.box_geom = [m.geom(f'box{i}').id for i in range(n_boxes)]
        self.box_mocap = [m.body_mocapid[m.body(f'box{i}').id] for i in range(n_boxes)]
        self.carried = m.geom('carried').id
        self.carried_q = m.jnt_qposadr[m.body('carried').jntadr[0]]
        self.n_active = 0
        self.rng = np.random.default_rng(seed)
        self.seeds = {True: [], False: []}         # 툴 +x (앞면) / 툴 아래 (윗면) 별로 최근 해
        self._jp, self._jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        self.n_ik = 0

    # ---------- 장면 ----------
    def _set_geom(self, g, half, on):
        m = self.m
        m.geom_size[g] = half
        m.geom_rbound[g] = float(np.linalg.norm(half))
        m.geom_aabb[g] = (0, 0, 0, *half)
        v = int(on)
        m.geom_contype[g] = m.geom_conaffinity[g] = v
        b = m.geom_bodyid[g]
        m.body_contype[b] = m.body_conaffinity[b] = v

    def set_obstacles(self, boxes) -> None:
        """boxes: [(중심 월드 m, x·y·z 크기 m)]. 축 정렬 상자. 0.5 mm 씩 작게 둔다 (맞닿은 이웃은 충돌이 아니다)."""
        assert len(boxes) <= len(self.box_geom), 'n_boxes 를 늘린다'
        for i, (c, s) in enumerate(boxes):
            self._set_geom(self.box_geom[i], np.asarray(s, float) / 2 - .0005, True)
            self.d.mocap_pos[self.box_mocap[i]] = c
        for i in range(len(boxes), self.n_active):
            self._set_geom(self.box_geom[i], np.full(3, .04), False)
            self.d.mocap_pos[self.box_mocap[i]] = (M.PARK[0] - .2 * (i % 20), M.PARK[1] + .2 * (i // 20), M.PARK[2])
        self.n_active = len(boxes)

    def clear_carried(self) -> None:
        self._set_geom(self.carried, np.full(3, .04), False)
        self.d.qpos[self.carried_q:self.carried_q + 7] = (M.PARK[0], M.PARK[1] - .5, M.PARK[2], 1, 0, 0, 0)

    def set_carried(self, center, size, quat=(1, 0, 0, 0)) -> None:
        """들고 가는 상자 (size 는 몸체 축 기준, quat 는 월드 자세). 사방 2 mm 작게 둔다 (놓는 순간 아래 상자·바닥에 닿는 것은 충돌이 아니다)."""
        self._set_geom(self.carried, np.asarray(size, float) / 2 - .002, True)
        self.d.qpos[self.carried_q:self.carried_q + 7] = (*center, *quat)

    # ---------- IK ----------
    def _fk(self, q):
        m, d = self.m, self.d
        d.qpos[self.arm] = q
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)
        return d.site_xpos[self.tcp].copy(), d.site_xmat[self.tcp].reshape(3, 3).copy()

    def ik(self, pos, axis, rail, seed, iters: int = 80):
        """툴 끝 = pos, 툴 축 = axis 인 관절 값 (없으면 None)."""
        m, d = self.m, self.d
        self.n_ik += 1
        d.qpos[self.rail] = rail
        pos, axis = np.asarray(pos, float), np.asarray(axis, float)
        q = np.clip(np.asarray(seed, float), self.lo + 1e-4, self.hi - 1e-4)

        def err(q):
            p, R = self._fk(q)
            return np.concatenate([p - pos, W_ROT * (R[:, 2] - axis)]), R
        e, R = err(q)
        f = float(e @ e)
        lam = 1e-2
        for _ in range(iters):
            if np.linalg.norm(e[:3]) < 5e-4 and float(R[:, 2] @ axis) > 1 - 5e-5:
                return q
            self._fk(q)
            mujoco.mj_jacSite(m, d, self._jp, self._jr, self.tcp)
            Jp, Jr = self._jp[:, self.arm_dof], self._jr[:, self.arm_dof]
            J = np.vstack([Jp, W_ROT * np.cross(Jr.T, R[:, 2]).T])
            H, g = J.T @ J, J.T @ e
            while True:
                dq = -np.linalg.solve(H + lam * (np.diag(np.diag(H)) + 1e-6 * np.eye(6)), g)
                qn = np.clip(q + dq, self.lo, self.hi)
                en, Rn = err(qn)
                fn = float(en @ en)
                if fn < f:
                    q, e, R, f = qn, en, Rn, fn
                    lam = max(lam / 3, 1e-7)
                    break
                lam *= 4
                if lam > 1e5:
                    break
            if lam > 1e5:
                break
        e, R = err(q)
        if np.linalg.norm(e[:3]) < 5e-4 and float(R[:, 2] @ axis) > 1 - 5e-5:
            return q
        return None

    def collides(self, rail, q):
        """(충돌 여부, 충돌한 geom 이름 쌍)."""
        m, d = self.m, self.d
        d.qpos[self.rail] = rail
        d.qpos[self.arm] = q
        mujoco.mj_fwdPosition(m, d)
        limit = self.clearance if self.clearance > 0 else -self.tol
        for c in d.contact[:d.ncon]:
            if c.dist >= limit:
                continue
            g1, g2 = int(c.geom1), int(c.geom2)
            r1, r2 = self.robot_geoms[g1], self.robot_geoms[g2]
            k1, k2 = g1 == self.carried, g2 == self.carried
            if (r1 and r2) or (r1 and k2) or (r2 and k1):
                continue
            if r1 or r2 or k1 or k2:
                return True, (m.geom(g1).name, m.geom(g2).name)
        return False, None

    def solve(self, pos, axis):
        """레일 위치와 IK 씨앗을 바꿔 가며 충돌 없는 해를 찾는다. (ok, rail, q, 실패 이유)."""
        front = axis[0] > .5
        dxs = (.30, .22, .38, .15, .45, .52, .60, .08, .68) if front else (0., .1, -.1, .2, -.2, .3, -.3)
        rails = []
        for dx in dxs:
            r = float(np.clip(pos[0] - dx - M.BASE_X0, 0, M.TRAVEL))
            if all(abs(r - o) > .02 for o in rails):
                rails.append(r)
        bank = self.seeds[front]
        why = 'ik'
        for extra in (4, 16):                          # 못 찾으면 씨앗을 늘려 한 번 더 (앞쪽 레일 3곳)
            for r in (rails if extra == 4 else rails[:3]):
                for s in (bank[:4] if extra == 4 else []) + [self.rng.uniform(self.lo, self.hi) for _ in range(extra)]:
                    q = self.ik(pos, axis, r, s)
                    if q is None:
                        continue
                    hit, pair = self.collides(r, q)
                    if hit:
                        why = f'collision {pair[0]}-{pair[1]}'
                        continue
                    bank.insert(0, q)
                    del bank[8:]
                    return True, r, q, ''
        return False, None, None, why

    def _follow(self, pos, axis, rail, q):
        """앞 단계 해 (rail, q) 에서 이어서 푼다 (경로가 매끄럽게). 안 되면 레일을 조금 옮겨 본다."""
        for dr in (0., .04, -.04, .08, -.08):
            r = float(np.clip(rail + dr, 0, M.TRAVEL))
            q2 = self.ik(pos, axis, r, q, iters=40)
            if q2 is not None and not self.collides(r, q2)[0]:
                return r, q2
        return None

    def check_flip(self, rec: dict, obstacles, origin, start=None, lifts=(.1, .06, .15, .2, .03), steps: int = 6,
                   continuous: bool = False) -> dict:
        """집은 뒤 뒤집기: 칸에서 윗면을 잡은 상자를 들어 올리면서 툴을 아래(-z)에서 월드 +x 로 90° 돌린다
        (흡착면이 출구를 보게). 툴 끝은 집은 점에서 위로 dz 만큼 올라간다 (각도에 비례, 또는 많이 기운 뒤에).
        상자는 툴에 붙어 같이 돈다.
          continuous=False  단계마다 IK 를 따로 푼다 (관절이 다른 해로 건너뛰어도 됨). 필요조건: 실패하면 그 자리에서는
                            돌릴 수 없다 (예: 돌리는 상자가 옆 칸 상자에 닿음)
          continuous=True   앞 단계 해에서 이어서 푼다 (매끄러운 경로). 이 모델에서는 제자리 뒤집기가 거의 다 실패한다
                            (집은 뒤 빈 곳으로 옮겨 돌리는 경로가 필요하다. 모션 플래너 몫)
        start: 집기 자세의 (rail, q) (continuous 에서만). 돌려줌 dict(ok, lift_m, why)."""
        origin = np.asarray(origin, float)
        self.set_obstacles(obstacles)
        fr = rec['from']
        g = np.array(fr['grasp_m']) + origin
        size = np.array(fr['size_xyz_m'], float)            # 칸에서: L = x, W = y, H = z
        if start is None and continuous:
            self.set_carried(g - (0, 0, size[2] / 2), size)
            ok, rail, q, _ = self.solve(g, (0, 0, -1))
            if not ok:
                self.clear_carried()
                return dict(ok=False, lift_m=None, why='pick')
            start = (rail, q)
        why = ''
        for late in (False, True):
            for dz in lifts:
                rail, q = start if start is not None else (0.0, None)
                ok = True
                for th in np.linspace(0, np.pi / 2, steps)[1:]:
                    a = np.array([np.sin(th), 0, -np.cos(th)])
                    rise = (1 - np.cos(th)) if late else th / (np.pi / 2)      # late: 많이 기운 뒤에 올린다
                    p = g + (0, 0, dz * rise)
                    self.set_carried(p + a * size[2] / 2, size, (np.cos(th / 2), 0, -np.sin(th / 2), 0))
                    if continuous:
                        nxt = self._follow(p, a, rail, q)
                    else:
                        res = self.solve(p, a)
                        nxt = (res[1], res[2]) if res[0] else None
                    if nxt is None:
                        ok, why = False, f'lift {dz:.2f}{" late" if late else ""} angle {np.degrees(th):.0f}'
                        break
                    rail, q = nxt
                if ok:
                    self.clear_carried()
                    return dict(ok=True, lift_m=dz, why='')
        self.clear_carried()
        return dict(ok=False, lift_m=None, why=why)

    # ---------- 이동 ----------
    def check(self, rec: dict, obstacles, origin) -> dict:
        """rec: buffer_env.move_record 형식 (로봇 좌표 m). obstacles: [(중심 월드 m, 크기 m)] (실은 상자 + 다른 칸 상자).
        origin: ROBOT_ORIGIN (로봇 좌표 -> 월드). 돌려줌 dict(ok, why, poses={이름: dict(rail_m, base_x_m, q_rad)})."""
        origin = np.asarray(origin, float)
        self.set_obstacles(obstacles)
        fr, to = rec['from'], rec['to']
        if rec['kind'] == 'retrieve':               # 넣기의 역순: 컨테이너 안에서 잡고 -> 40 mm 빼고 -> 버퍼 칸에 내려놓기
            c = np.array(fr['center_m']) + origin
            steps = [('grasp', np.array(fr['grasp_m']) + origin, (1, 0, 0), c, fr['size_xyz_m']),
                     ('pull', np.array(fr['pre_insert_m']) + origin, (1, 0, 0), c - (.04, 0, 0), fr['size_xyz_m']),
                     ('drop', np.array(to['grasp_m']) + origin, (0, 0, -1), np.array(to['center_m']) + origin, to['size_xyz_m'])]
        else:
            steps = [('pick', np.array(fr['grasp_m']) + origin, (0, 0, -1), np.array(fr['center_m']) + origin, fr['size_xyz_m'])]
        if rec['kind'] == 'retrieve':
            pass
        elif rec['kind'] == 'place' and rec.get('approach') == 'top':   # 위에서 내려놓기: 40 mm 위 대기점 -> 수직으로 내려놓기
            c = np.array(to['center_m']) + origin
            steps.append(('pre', np.array(to['pre_insert_m']) + origin, (0, 0, -1), c + (0, 0, .04), to['size_xyz_m']))
            steps.append(('insert', np.array(to['grasp_m']) + origin, (0, 0, -1), c, to['size_xyz_m']))
        elif rec['kind'] == 'place':
            c = np.array(to['center_m']) + origin
            steps.append(('pre', np.array(to['pre_insert_m']) + origin, (1, 0, 0), c - (.04, 0, 0), to['size_xyz_m']))
            steps.append(('insert', np.array(to['grasp_m']) + origin, (1, 0, 0), c, to['size_xyz_m']))
        else:
            steps.append(('drop', np.array(to['grasp_m']) + origin, (0, 0, -1), np.array(to['center_m']) + origin, to['size_xyz_m']))
        poses = {}
        for name, pos, axis, bc, bs in steps:
            self.set_carried(bc, bs)
            ok, rail, q, why = self.solve(pos, axis)
            if not ok:
                return dict(ok=False, why=f'{name}: {why}', poses=poses)
            poses[name] = dict(rail_m=round(rail, 4), base_x_m=round(M.BASE_X0 + rail, 4),
                               rail_team_m=round(M.BASE_X0 + rail - TEAM_RAIL0_X, 4), q_rad=[round(float(v), 4) for v in q])
            if name == 'pick' and self.flip and rec['kind'] == 'place':
                f = self.check_flip(rec, obstacles, origin, start=(rail, q))
                if not f['ok']:
                    return dict(ok=False, why=f"flip: {f['why']}", poses=poses)
                poses['flip'] = dict(lift_m=f['lift_m'])
        return dict(ok=True, why='', poses=poses)
