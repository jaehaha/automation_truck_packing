"""실물 대비 검사: 완성된 적재를 처음부터 다시 쌓되 모든 상자를 움직일 수 있는 물체로 두고
(pack_env 의 검사는 이미 놓은 상자를 고정해 둔다), 놓는 위치 오차·상자 치수 오차·무게·마찰을 넣어
끝까지 쌓은 뒤 상자가 놓인 자리에서 수평으로 얼마나 밀렸고 얼마나 기울었는지 본다.

  uv run python verify_stack.py --moves moves.json
  uv run python verify_stack.py --eval eval.json --method policy --set full
"""
from __future__ import annotations

import argparse
import json
import math
import os
from multiprocessing import Pool
from pathlib import Path

os.environ.setdefault('OMP_NUM_THREADS', '1')
import mujoco  # noqa: E402
import numpy as np  # noqa: E402


def build(placements, depth, width, height, friction):
    D, W, H = depth / 1000, width / 1000, height / 1000
    g = [f'<geom type="plane" size="3 3 .1" friction="{friction} .005 .0001"/>',
         f'<geom type="box" pos="{-0.005} {W / 2} {H / 2}" size=".005 {W / 2 + .01} {H / 2}"/>',
         f'<geom type="box" pos="{D / 2} {-0.005} {H / 2}" size="{D / 2 + .01} .005 {H / 2}"/>',
         f'<geom type="box" pos="{D / 2} {W + 0.005} {H / 2}" size="{D / 2 + .01} .005 {H / 2}"/>']
    b = []
    for n, p in enumerate(placements):
        s = np.array(p['dims_xyz'], float) / 2000
        b.append(f'<body name="b{n}" pos="{5 + 0.4 * n} 5 -2" gravcomp="1"><freejoint/>'
                 f'<geom type="box" size="{s[0]} {s[1]} {s[2]}" mass="{p.get("mass", 0.4)}" friction="{friction} .005 .0001" '
                 f'contype="0" conaffinity="0"/></body>')
    return f'''<mujoco><option timestep="0.002" cone="elliptic" impratio="10"/>
<worldbody>{"".join(g)}{"".join(b)}</worldbody></mujoco>'''


def check(placements, depth=830, width=480, height=568, pos_err=2.0, size_err=1.0, yaw_err=1.0, friction=0.5,
          settle=0.3, final=1.0, seed=0):
    rng = np.random.default_rng(seed)
    ps = []
    for p in placements:
        q = dict(p)
        q['dims_xyz'] = [max(10.0, d - 1.0 + rng.uniform(-size_err, size_err)) for d in p['dims_xyz']]   # 1 mm 작게 +- 오차
        q['mass'] = float(rng.uniform(0.15, 1.5))
        ps.append(q)
    m = mujoco.MjModel.from_xml_string(build(ps, depth, width, height, friction))
    d = mujoco.MjData(m)
    n = len(ps)
    geoms = [m.body(f'b{i}').geomadr[0] for i in range(n)]
    qadr = [m.jnt_qposadr[m.body(f'b{i}').jntadr[0]] for i in range(n)]
    vadr = [m.jnt_dofadr[m.body(f'b{i}').jntadr[0]] for i in range(n)]
    targets = []
    for i, p in enumerate(ps):
        c = np.array(p['center'], float) / 1000
        c[:2] += rng.uniform(-pos_err, pos_err, 2) / 1000
        targets.append(c.copy())                     # 놓은 위치 (오차 포함). 그 뒤 수평으로 얼마나 밀렸는지 본다
        c[2] += 0.001
        yaw = math.radians(rng.uniform(-yaw_err, yaw_err))
        d.qpos[qadr[i]:qadr[i] + 7] = (*c, math.cos(yaw / 2), 0, 0, math.sin(yaw / 2))
        d.qvel[vadr[i]:vadr[i] + 6] = 0
        m.body_gravcomp[m.geom_bodyid[geoms[i]]] = 0
        m.geom_contype[geoms[i]] = m.geom_conaffinity[geoms[i]] = 1
        b = m.geom_bodyid[geoms[i]]
        m.body_contype[b] = m.body_conaffinity[b] = 1          # 몸체 단위 필터도 켠다 (pack_env._collide 와 같음)
        for _ in range(int(settle / m.opt.timestep)):
            mujoco.mj_step(m, d)
    for _ in range(int(final / m.opt.timestep)):
        mujoco.mj_step(m, d)
    disp, tilt = [], []
    for i in range(n):
        pos = d.qpos[qadr[i]:qadr[i] + 3]
        disp.append(float(np.linalg.norm(pos[:2] - targets[i][:2]) * 1000))
        q = d.qpos[qadr[i] + 3:qadr[i] + 7]
        tilt.append(float(np.degrees(2 * math.acos(min(1.0, math.sqrt(q[0] ** 2 + q[3] ** 2))))))
    disp, tilt = np.array(disp), np.array(tilt)
    bad = [dict(i=int(i), dims_xyz=placements[i]['dims_xyz'], center=placements[i]['center'], disp_mm=round(float(disp[i]), 1),
                tilt_deg=round(float(tilt[i]), 1)) for i in np.flatnonzero((disp > 5) | (tilt > 3))]
    return dict(bad=bad, boxes=n, max_disp_mm=float(disp.max()), mean_disp_mm=float(disp.mean()), moved_5mm=int((disp > 5).sum()),
                moved_10mm=int((disp > 10).sum()), max_tilt_deg=float(tilt.max()), tilted_3deg=int((tilt > 3).sum()),
                fell=int(((disp > 30) | (tilt > 15)).sum()))


def _job(a):
    placements, kw = a
    return check(placements, **kw)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--moves', help='plan_moves.py 결과')
    ap.add_argument('--eval', help='eval_buffer.py 결과')
    ap.add_argument('--method', default='policy')
    ap.add_argument('--set', default='full')
    ap.add_argument('--n', type=int, default=32)
    ap.add_argument('--pos-err', type=float, default=2.0, help='놓는 위치 오차 (mm, 균등 +-)')
    ap.add_argument('--size-err', type=float, default=1.0, help='상자 치수 오차 (mm, 균등 +-, 기본 1 mm 작게)')
    ap.add_argument('--friction', type=float, default=0.5)
    ap.add_argument('--workers', type=int, default=64)
    ap.add_argument('--out')
    a = ap.parse_args()
    if a.moves:
        layouts = [json.loads(Path(a.moves).read_text())['placements']]
    else:
        eps = json.loads(Path(a.eval).read_text())['episodes']
        layouts = [e['placements'] for e in eps if e['method'] == a.method and e['set'] == a.set][: a.n]
    kw = dict(pos_err=a.pos_err, size_err=a.size_err, friction=a.friction)

    def depth_of(L):                                # zone 이면 옆벽을 버퍼 구역 끝까지 (1200)
        return max(830, int(math.ceil(max(p['center'][0] + p['dims_xyz'][0] / 2 for p in L))))
    with Pool(min(a.workers, len(layouts))) as pool:
        res = pool.map(_job, [(L, dict(kw, seed=i, depth=depth_of(L))) for i, L in enumerate(layouts)], chunksize=1)
    agg = {k: float(np.mean([r[k] for r in res])) for k in res[0] if k != 'bad'}
    worst = max(res, key=lambda r: r['max_disp_mm'])
    print(f"적재 {len(res)}개, 상자 평균 {agg['boxes']:.0f}개: 최대 밀림 평균 {agg['max_disp_mm']:.1f} mm (최악 {worst['max_disp_mm']:.1f}), "
          f"5 mm 넘게 밀린 상자 {agg['moved_5mm']:.2f}개/적재, 10 mm 넘게 {agg['moved_10mm']:.2f}개, "
          f"3도 넘게 기운 상자 {agg['tilted_3deg']:.2f}개, 넘어짐 {agg['fell']:.2f}개, 넘어진 적재 {sum(r['fell'] > 0 for r in res)}/{len(res)}")
    if a.out:
        Path(a.out).write_text(json.dumps(dict(summary=agg, runs=res), indent=1))


if __name__ == '__main__':
    main()
