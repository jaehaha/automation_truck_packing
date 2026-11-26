"""계획 파일(plan_moves.py 결과) 의 이동을 처음부터 다시 따라가며 R17 + Piper 모델 (rig/) 로 모션 검사한다.

이동마다 그 직전 장면 (scene_before: 실린 상자 + 칸마다 있는 상자) 을 장애물로 두고
집기·대기점·밀어 넣기 (버퍼로 옮기기는 집기·내려놓기) 자세의 IK 와 팔·들고 가는 상자의 충돌을 본다.

  uv run python check_plan.py moves.json [more.json ...]
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np  # noqa: E402

from pack_env import BOX_TYPES, to_world  # noqa: E402
from buffer_core import SLOT_NAMES  # noqa: E402
from rig.motion import MotionChecker  # noqa: E402


def check_file(args):
    path, flip = args
    plan = json.loads(Path(path).read_text())
    origin = np.array(plan['robot_origin_world_m'])
    cells = {name: np.array(v) + origin for name, v in plan['buffer_cells_robot_m'].items()}
    types = {f'b{n + 1:03d}': int(k) for n, (_, k) in enumerate(plan['job'])}
    box_types = {int(k): tuple(v) for k, v in plan['box_types'].items()} if 'box_types' in plan else BOX_TYPES
    placed = [to_world(p['center'], p['dims_xyz']) for p in plan['placements'] if not p.get('kept')]
    mc = MotionChecker(n_boxes=len(plan['placements']) + 8)
    rows = []
    for mv in plan['moves']:
        sb = mv['scene_before']
        obs = list(placed[:sb['placed']])
        for slot, box in sb['slots'].items():
            if slot == mv['from']['slot']:
                continue                                  # 들고 갈 상자
            L, W, H = box_types[types[box]]
            c = cells[slot].copy()
            c[2] = to_world((0, 0, H / 2), (0, 0, 0))[0][2]
            obs.append((c, np.array([L, W, H], float) / 1000))
        res = mc.check(mv, obs, origin)
        row = dict(step=mv['step'], kind=mv['kind'], ok=res['ok'], why=res['why'], src=mv['from']['slot'],
                   neighbors=len(sb['slots']) - 1, to=mv['to'].get('container_mm', mv['to'].get('slot')))
        if flip and mv['kind'] == 'place':
            f = mc.check_flip(mv, obs, origin)
            row.update(flip=f['ok'], flip_lift_m=f['lift_m'], flip_why=f['why'])
        rows.append(row)
    return path, rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('plans', nargs='+')
    ap.add_argument('--workers', type=int, default=16)
    ap.add_argument('--flip', action='store_true', help='집은 뒤 뒤집기(툴 -z -> +x, 들어 올리며) 경로도 검사한다')
    ap.add_argument('--out')
    a = ap.parse_args()
    with Pool(min(a.workers, len(a.plans))) as pool:
        res = pool.map(check_file, [(p, a.flip) for p in a.plans], chunksize=1)
    rows = [r for _, rs in res for r in rs]
    for kind in ('place', 'stage'):
        rk = [r for r in rows if r['kind'] == kind]
        if rk:
            print(f"{kind:6s} {len(rk):5d} 이동, 모션 검사 통과 {np.mean([r['ok'] for r in rk]) * 100:.2f}%")
    fl = [r for r in rows if 'flip' in r]
    if fl:
        for lab, sub in (('버퍼 칸에 다른 상자 없음', [r for r in fl if r['neighbors'] == 0]),
                         ('버퍼 칸에 다른 상자 있음', [r for r in fl if r['neighbors'] > 0])):
            if sub:
                lifts = [r['flip_lift_m'] for r in sub if r['flip']]
                print(f"뒤집기 ({lab}) {len(sub):4d} 이동: 통과 {np.mean([r['flip'] for r in sub]) * 100:.1f}%, "
                      f"필요한 들어올림 중앙 {np.median(lifts) * 1000 if lifts else float('nan'):.0f} mm")
        for r in [r for r in fl if not r['flip']][:6]:
            print('   뒤집기 실패', r['step'], r['src'], r['neighbors'], r['flip_why'])
    fails = [r for r in rows if not r['ok']]
    if fails:
        print('실패 이유:', dict(Counter(r['why'].split(' ')[0] + ' ' + r['why'].split(' ')[1] for r in fails).most_common(6)))
        for r in fails[:8]:
            print('  ', r)
    if a.out:
        Path(a.out).write_text(json.dumps(dict(rows=rows), ensure_ascii=False))


if __name__ == '__main__':
    main()
