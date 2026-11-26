"""앞면 밀어 넣기 도달 표를 만든다 (buffer_core 의 reach 규칙이 쓴다).

격자: x = 상자 앞면 (컨테이너 mm) 0~1240 / 20, y = 앞면 중심 40~440 / 20, z = 앞면 중심 40~520 / 40.
빈 장면 (프레임·바닥만) 에서 툴 끝이 그 점, 툴 축이 월드 +x 이고 충돌 없는 IK 해가 있으면 참 (실행 검사와 같은 기준).
--clearance 를 주면 팔이 프레임·바닥에서 그만큼 떨어져야 참 (보수적. 3 mm 면 높이 80 상자를 바닥에 넣는 자리가 대부분 빠진다).
상자 사이에서 손목이 옆 상자에 닿는 경우는 표가 아니라 실행 때 모션 검사 (motion.py) 가 본다.

  uv run python -m rig.reach_table --tool 0.08 --out rig/reach_front.npz
  uv run python -m rig.reach_table --top --tool 0.08 --out rig/reach_top.npz     # 위에서 내려놓기: 툴 축 -z, x·y = 상자 윗면 중심, z = 윗면 높이
"""
from __future__ import annotations

import argparse
import os
import time
from multiprocessing import Pool

os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np  # noqa: E402

from pack_env import C_ORIGIN  # noqa: E402
from .motion import MotionChecker  # noqa: E402

XS, YS, ZS = np.arange(0, 1241, 20), np.arange(40, 441, 20), np.arange(40, 521, 40)
TOP_ZS = np.arange(80, 641, 40)                 # --top: 상자 윗면 높이 (대기점 = 40 mm 위까지 보려고 640 까지)
CLEAR = 0.0
TOP = False
_MC = None


def _job(a):
    global _MC
    ix, iy, iz, tool = a
    if _MC is None:
        _MC = MotionChecker(n_boxes=1, tool=tool, clearance=CLEAR, seed=os.getpid())
    axis = (0, 0, -1) if TOP else (1, 0, 0)
    pos = C_ORIGIN + np.array([-XS[ix], -YS[iy], (TOP_ZS if TOP else ZS)[iz]]) / 1000
    _MC.clear_carried()
    ok = _MC.solve(pos, axis)[0]
    if not ok:                                  # 씨앗을 비우고 한 번 더
        _MC.seeds[not TOP] = []
        ok = _MC.solve(pos, axis)[0]
    return ix, iy, iz, bool(ok)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--tool', type=float, default=.08)
    ap.add_argument('--out', default='rig/reach_front.npz')
    ap.add_argument('--workers', type=int, default=64)
    ap.add_argument('--clearance', type=float, default=0.0, help='팔과 프레임·바닥 사이 최소 거리 (m). 0 이면 실행 검사와 같다 (1.5 mm 파고듦까지 봐줌)')
    ap.add_argument('--top', action='store_true', help='위에서 내려놓기 표 (툴 축 -z, 점 = 상자 윗면 중심)')
    a = ap.parse_args()
    global CLEAR, TOP
    CLEAR, TOP = a.clearance, a.top
    zs = TOP_ZS if a.top else ZS
    t0 = time.time()
    tasks = [(ix, iy, iz, a.tool) for iz in range(len(zs)) for iy in range(len(YS)) for ix in range(len(XS))]
    with Pool(a.workers) as p:
        res = p.map(_job, tasks, chunksize=8)
    T = np.zeros((len(XS), len(YS), len(zs)), bool)
    for ix, iy, iz, ok in res:
        T[ix, iy, iz] = ok
    note = ('top-down placement (tool -z, point = box top-face centre)' if a.top else 'front insertion (tool +x world)')
    np.savez(a.out, ok=T, x0=XS[0], dx=20, y0=YS[0], dy=20, z0=zs[0], dz=40, tool=a.tool,
             clearance=a.clearance, note=note + ', R17 + inverted Piper (rig/model.py), empty scene')
    print(f'{T.mean() * 100:.1f}% reachable, {time.time() - t0:.0f}s -> {a.out}')


if __name__ == '__main__':
    main()
