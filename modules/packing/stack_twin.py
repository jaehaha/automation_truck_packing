"""동적 적재 검사 (실물 대비): 지금까지 놓은 상자를 모두 움직일 수 있는 물체로 둔 MuJoCo 사본.

pack_env._settle 은 이미 놓은 상자를 고정한 채 새 상자만 떨어뜨려 본다. 그래서 좁은 기둥이 높아지며
아래 상자가 함께 기우는 경우를 못 본다. 여기서는 새 상자를 몇 mm 어긋나게 놓아 보면서 쌓인 상자 전체가
버티는지 본다. 하나라도 tol 보다 밀리거나 기울면 그 자리를 쓰지 않는다 (플래너가 다음 순위로 넘어간다).

좌표는 컨테이너 좌표 (m). 상자는 가로·세로 1 mm 작게 (실제 상자 오차), 무게 0.5 kg, 마찰 0.5.
"""
from __future__ import annotations

import math

import mujoco
import numpy as np


class StackTwin:
    def __init__(self, job, box_types, depth=830, width=480, height=568, friction=0.5, mass=0.5,
                 offsets=((2, 0), (-2, 0), (0, 2), (0, -2)), settle=0.3, disp_tol=3.0, tilt_tol=2.0, jitter=1.0, seed=0):
        D, W, H = depth / 1000, width / 1000, height / 1000
        g = [f'<geom type="plane" size="3 3 .1" friction="{friction} .005 .0001"/>',
             f'<geom type="box" pos="-.005 {W / 2} {H / 2}" size=".005 {W / 2 + .01} {H / 2}"/>',
             f'<geom type="box" pos="{D / 2} -.005 {H / 2}" size="{D / 2 + .01} .005 {H / 2}"/>',
             f'<geom type="box" pos="{D / 2} {W + .005} {H / 2}" size="{D / 2 + .01} .005 {H / 2}"/>']
        b = []
        for n, (_, k) in enumerate(job):
            L, Wd, Hh = box_types[k]
            s = max(L, Wd, Hh) / 2000
            b.append(f'<body name="b{n}" pos="{5 + .4 * n} 5 -2" gravcomp="1"><freejoint/>'
                     f'<geom type="box" size="{s} {s} {s}" mass="{mass}" friction="{friction} .005 .0001" contype="0" conaffinity="0"/></body>')
        xml = (f'<mujoco><option timestep="0.002" cone="elliptic" impratio="10"/><worldbody>{"".join(g)}{"".join(b)}'
               f'</worldbody></mujoco>')
        self.m = mujoco.MjModel.from_xml_string(xml)
        self.d = mujoco.MjData(self.m)
        m = self.m
        self.geom = [m.body(f'b{n}').geomadr[0] for n in range(len(job))]
        self.qa = [m.jnt_qposadr[m.body(f'b{n}').jntadr[0]] for n in range(len(job))]
        self.va = [m.jnt_dofadr[m.body(f'b{n}').jntadr[0]] for n in range(len(job))]
        self.nominal = {}          # jid -> (center m, quat)
        self.offsets, self.settle, self.disp_tol, self.tilt_tol, self.jitter = offsets, settle, disp_tol, tilt_tol, jitter
        self.rng = np.random.default_rng(seed)
        self.tests = 0

    def _activate(self, jid, dims_mm, on=True):
        m = self.m
        g = self.geom[jid]
        half = np.array(dims_mm, float) / 2000
        half[:2] -= 0.0005                                     # 가로·세로 1 mm 작게
        m.geom_size[g] = half
        m.geom_rbound[g] = float(np.linalg.norm(half))
        m.geom_aabb[g] = (0, 0, 0, *half)
        b = m.geom_bodyid[g]
        v = int(on)
        m.geom_contype[g] = m.geom_conaffinity[g] = v
        m.body_contype[b] = m.body_conaffinity[b] = v
        m.body_gravcomp[b] = 0.0 if on else 1.0
        if not on:
            self.d.qpos[self.qa[jid]:self.qa[jid] + 7] = (5 + .4 * jid, 5, -2, 1, 0, 0, 0)
            self.d.qvel[self.va[jid]:self.va[jid] + 6] = 0

    def place(self, jid, center_mm, dims_mm):
        """실제로 놓은 상자를 사본에 더한다."""
        self._activate(jid, dims_mm)
        self.nominal[jid] = np.array(center_mm, float) / 1000

    def _reset(self):
        d = self.d
        for jid, c in self.nominal.items():
            p = c.copy()
            p[:2] += self.rng.uniform(-self.jitter, self.jitter, 2) / 1000
            d.qpos[self.qa[jid]:self.qa[jid] + 7] = (*p, 1, 0, 0, 0)
            d.qvel[self.va[jid]:self.va[jid] + 6] = 0

    def test(self, jid, center_mm, dims_mm):
        """(통과 여부, 최대 수평 밀림 mm, 최대 기울기 deg). 사본 상태는 바뀌지 않는다."""
        self.tests += 1
        m, d = self.m, self.d
        self._activate(jid, dims_mm)
        worst_d, worst_t = 0.0, 0.0
        ids = list(self.nominal) + [jid]
        for ox, oy in self.offsets:
            self._reset()
            c = np.array(center_mm, float) / 1000 + (ox / 1000, oy / 1000, 0.001)
            d.qpos[self.qa[jid]:self.qa[jid] + 7] = (*c, 1, 0, 0, 0)
            d.qvel[self.va[jid]:self.va[jid] + 6] = 0
            mujoco.mj_forward(m, d)
            start = {i: d.qpos[self.qa[i]:self.qa[i] + 3].copy() for i in ids}
            for _ in range(int(self.settle / m.opt.timestep)):
                mujoco.mj_step(m, d)
            for i in ids:
                p = d.qpos[self.qa[i]:self.qa[i] + 3]
                q = d.qpos[self.qa[i] + 3:self.qa[i] + 7]
                worst_d = max(worst_d, float(np.linalg.norm(p[:2] - start[i][:2]) * 1000))
                worst_t = max(worst_t, float(np.degrees(2 * math.acos(min(1.0, math.sqrt(q[0] ** 2 + q[3] ** 2))))))
            if worst_d > self.disp_tol or worst_t > self.tilt_tol:
                break
        self._activate(jid, dims_mm, on=False)
        self._reset()
        return worst_d <= self.disp_tol and worst_t <= self.tilt_tol, worst_d, worst_t
