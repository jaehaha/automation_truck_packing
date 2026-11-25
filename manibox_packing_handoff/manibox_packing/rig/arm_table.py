"""앞면 밀어 넣기 팔 공간 표를 만든다 (buffer_core 의 arm 규칙이 쓴다).

상자 앞면 중심 (x, y, z) 마다 빈 장면 (프레임·바닥) 에서 충돌 없는 IK 해를 K 개까지 찾고, 해마다 팔이 차지하는 공간을 적는다.
팔 공간 = 대기점 자세 (앞면 + 40 mm) 와 밀어 넣은 자세 (레일만 40 mm 더) 의 팔 메시 (베이스 ~ link4) 꼭짓점을 바닥 20 mm 칸으로 모은
칸마다 가장 낮은 z (옆 칸까지 넓혀 보수적으로). 실린 상자 (높이 지도) 가 어느 해의 팔 공간에도 닿지 않아야 그 자리를 쓴다.
큰 상자를 맨 윗줄 옆벽 쪽에 넣을 때 팔꿈치 (link3) 가 옆 상자 기둥에 닿는 경우를 학습 때부터 뺀다.
손목 (link5·6) 과 흡착 컵은 넣지 않는다: 앞면에 붙어 있어 20 mm 칸으로는 옆 상자와 겹쳐 보인다. 진입 통로 규칙과 실행 전 모션 검사가 본다.

격자: x = 상자 앞면 (컨테이너 mm) 0~1200 / 40, y = 앞면 중심 40~440 / 20, z = 앞면 중심 40~560 / 40.
해: motion.MotionChecker.solve 와 같은 레일 위치 (앞면에서 0.08~0.68 m) 마다 씨앗 몇 개로 푼 해, 레일 위치가 다른 것부터 K 개.
칸 창: 앞면 중심 기준 x -120 ~ +680, y -240 ~ +240 mm (20 mm 칸). 값 = z (5 mm 단위, uint8), 255 = 팔 없음.

  uv run python -m rig.arm_table --out rig/arm_front.npz
"""
from __future__ import annotations

import argparse
import os
import time
from multiprocessing import Pool

os.environ.setdefault('OMP_NUM_THREADS', '1')
import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from pack_env import C_ORIGIN  # noqa: E402
from . import model as M  # noqa: E402
from .motion import MotionChecker  # noqa: E402

XS, YS, ZS = np.arange(0, 1201, 40), np.arange(40, 441, 20), np.arange(40, 561, 40)
WX0, WY0, CELL, NWX, NWY = -120, -240, 20, 40, 24
K = 6
PRE = 40
DXS = (.30, .22, .38, .15, .45, .52, .60, .08, .68)
_MC = None
_PTS = None


ARM_LINKS = ('base_link', 'link1', 'link2', 'link3', 'link4')


def _robot_points(mc):
    """팔 geom (베이스 ~ link4) 마다 몸체 좌표 점들 (메시 꼭짓점, 원기둥은 둘레 점)."""
    m = mc.m
    out = []
    for g in np.nonzero(mc.robot_geoms)[0]:
        if m.body(m.geom_bodyid[g]).name not in ARM_LINKS:
            continue
        if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
            mid = m.geom_dataid[g]
            v = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
            out.append((g, np.asarray(v, float)))
        elif m.geom_type[g] == mujoco.mjtGeom.mjGEOM_CYLINDER:
            r, h = m.geom_size[g][0], m.geom_size[g][1]
            t = np.linspace(0, 2 * np.pi, 12, endpoint=False)
            ring = np.stack([r * np.cos(t), r * np.sin(t)], 1)
            out.append((g, np.concatenate([np.c_[ring, np.full(12, z)] for z in (-h, 0, h)])))
    return out


def _footprint(mc, rail, q, xf, y):
    """(rail, q) 자세의 로봇 점들 -> 앞면 중심 기준 칸 창의 칸마다 가장 낮은 z (mm)."""
    m, d = mc.m, mc.d
    d.qpos[mc.rail] = rail
    d.qpos[mc.arm] = q
    mujoco.mj_kinematics(m, d)
    P = np.concatenate([d.geom_xpos[g] + v @ d.geom_xmat[g].reshape(3, 3).T for g, v in _PTS])
    cx = (C_ORIGIN[0] - P[:, 0]) * 1000 - xf
    cy = (C_ORIGIN[1] - P[:, 1]) * 1000 - y
    cz = (P[:, 2] - C_ORIGIN[2]) * 1000
    a = np.floor((cx - WX0) / CELL).astype(int)
    b = np.floor((cy - WY0) / CELL).astype(int)
    keep = (a >= 0) & (a < NWX) & (b >= 0) & (b < NWY) & (cz < 600)
    Z = np.full((NWX, NWY), np.inf)
    np.minimum.at(Z, (a[keep], b[keep]), cz[keep])
    return Z


def _job(args):
    global _MC, _PTS
    ix, iy, iz = args
    if _MC is None:
        _MC = MotionChecker(n_boxes=1, seed=os.getpid())
        _MC.clear_carried()
        _PTS = _robot_points(_MC)
    mc = _MC
    xf, y, z = XS[ix], YS[iy], ZS[iz]
    pos = C_ORIGIN + np.array([-(xf + PRE), -y, z]) / 1000          # 대기점: 툴 끝 = 앞면 + 40 mm
    axis = np.array([1., 0, 0])
    rails = []
    for dx in DXS:
        r = float(np.clip(pos[0] - dx - M.BASE_X0, 0, M.TRAVEL))
        if all(abs(r - o) > .02 for o in rails):
            rails.append(r)
    sols = []
    for r in rails:
        for s in mc.seeds[True][:3] + [mc.rng.uniform(mc.lo, mc.hi) for _ in range(4)]:
            q = mc.ik(pos, axis, r, s)
            if q is None or mc.collides(r, q)[0]:
                continue
            r2 = r + PRE / 1000                     # 밀어 넣기: 레일만 40 mm (끝이면 이어서 푼다)
            if r2 <= M.TRAVEL:
                q2 = q
            else:
                r2 = M.TRAVEL
                q2 = mc.ik(pos + axis * PRE / 1000, axis, r2, q, iters=40)
                if q2 is None:
                    continue
            if mc.collides(r2, q2)[0]:
                continue
            mc.seeds[True].insert(0, q)
            del mc.seeds[True][8:]
            Z = np.minimum(_footprint(mc, r, q, xf, y), _footprint(mc, r2, q2, xf, y))
            sols.append(Z)
            break                                   # 레일 위치마다 해 하나
        if len(sols) >= K:
            break
    out = np.full((K, NWX, NWY), 255, np.uint8)
    for n, Z in enumerate(sols):
        P = np.pad(Z, 1, constant_values=np.inf)    # 옆 칸까지 넓혀 (꼭짓점 사이 빈틈) 보수적으로
        Zd = np.min([P[i:i + NWX, j:j + NWY] for i in range(3) for j in range(3)], 0)
        out[n] = np.where(np.isfinite(Zd), np.clip(np.floor(Zd / 5), 0, 254), 255).astype(np.uint8)
    return ix, iy, iz, len(sols), out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', default='rig/arm_front.npz')
    ap.add_argument('--workers', type=int, default=200)
    a = ap.parse_args()
    t0 = time.time()
    grid = [(ix, iy, iz) for ix in range(len(XS)) for iy in range(len(YS)) for iz in range(len(ZS))]
    Z = np.full((len(XS), len(YS), len(ZS), K, NWX, NWY), 255, np.uint8)
    N = np.zeros((len(XS), len(YS), len(ZS)), np.uint8)
    with Pool(a.workers) as pool:
        for ix, iy, iz, n, out in pool.imap_unordered(_job, grid, chunksize=8):
            Z[ix, iy, iz], N[ix, iy, iz] = out, n
    np.savez_compressed(a.out, z=Z, n=N, x0=XS[0], dx=XS[1] - XS[0], y0=YS[0], dy=YS[1] - YS[0], z0=ZS[0], dz=ZS[1] - ZS[0],
                        wx0=WX0, wy0=WY0, cell=CELL, unit=5)
    print(f'{a.out}: 격자 {N.shape}, 해가 있는 점 {np.mean(N > 0) * 100:.1f}%, 점마다 해 {N[N > 0].mean():.1f}개, '
          f'{time.time() - t0:.0f} s')


if __name__ == '__main__':
    main()
