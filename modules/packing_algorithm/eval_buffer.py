"""평가: 같은 작업에서 기준선(버퍼 없음 / 버퍼를 막혔을 때만) 과 학습 정책(탐색 없음 / 탐색) 을 비교한다.
MuJoCo 안정성 확인을 켠 상태(physics=True)이고, 모든 행동이 공식 마스크(pack_env 규칙) 안인지 확인한다.

  uv run python eval_buffer.py --out eval.json            # 기본: team / full / full_near 각 64개
  uv run python eval_buffer.py --search 6 --sets full --n 32
"""
from __future__ import annotations

import argparse
import json
import os
import time
from multiprocessing import Pool
from pathlib import Path

os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np  # noqa: E402

from buffer_core import BufferConfig, make_job, unload_check  # noqa: E402
from buffer_planner import DigPlanner, GreedyPlanner, OrderPlanner, Planner, SearchPlanner, feed_search, sync_twin  # noqa: E402
from stack_twin import StackTwin  # noqa: E402
from buffer_policy import Policy  # noqa: E402
from buffer_env import BufferPackEnv  # noqa: E402

HERE = Path(__file__).resolve().parent
SETS = ['team', 'full', 'full_near', 'full_random', 'team_random', 'fit', 'fit44']     # 작업 묶음 (씨앗이 이 순서에 묶여 있다. 새 묶음은 끝에)


_MOTION = {}


def episode(args):
    method, kind, idx, policy_path, search, samples, support, opt = args
    n_stage = 0 if method == 'greedy_nobuf' else opt.get('n_stage', 3)
    rules = dict(zone=opt.get('zone', False), zone_late=opt.get('zone_late', False), slot_clear=opt.get('slot_clear', False),
                 reach=opt.get('reach'), stage_order=tuple(opt.get('stage_order', ())), keep_stage=opt.get('keep_stage', False),
                 arm=opt.get('arm'), pool=opt.get('pool', False), center_cell=opt.get('center_cell', False),
                 front_cell=opt.get('front_cell', False), top_down=opt.get('top_down', False), reach_top=opt.get('reach_top'))
    if opt.get('box_types'):
        rules['box_types'] = {int(k): tuple(v) for k, v in opt['box_types'].items()}
    if opt.get('shares'):
        rules['shares'] = {int(k): float(v) for k, v in opt['shares'].items()}
    if opt.get('retrieve') or method == 'policy_dig' or method.startswith('policy_rearrange'):
        rules.update(retrieve=True, max_retrieve_per_box=3)
    cfg = BufferConfig(physics=True, n_stage=n_stage, support_min=support, **rules)
    job_cfg = BufferConfig(zone=opt.get('job_zone', rules['zone']), **{k: rules[k] for k in ('box_types', 'shares') if k in rules})
    job = make_job(np.random.default_rng(777000 + 1000 * SETS.index(kind) + idx), job_cfg, kind)
    if opt.get('feed_tries'):                       # 오프라인: 목록을 다 알고 투입 순서를 정한다 (나중 지점부터 + 같은 지점 안 순서 탐색)
        order, _ = feed_search(job, Policy.load(policy_path), cfg, opt['feed_tries'], seed=idx)
        job = [job[n] for n in order]
    env = BufferPackEnv(cfg)
    obs, info = env.reset(options={'job': job})
    if method.startswith('greedy'):
        planner = GreedyPlanner()
    elif method == 'policy_order':                  # 버퍼로 순서 바로잡기 + 정책 자리
        planner = OrderPlanner(Policy.load(policy_path), seed=idx)
    elif method == 'policy_dig':                    # 파내고 넣기 (실은 상자 다시 꺼내기)
        planner = DigPlanner(Policy.load(policy_path), seed=idx)
    elif method.startswith('policy_rearrange'):     # 적재하면서 버퍼로 계속 재배치 (빔 탐색), 예: policy_rearrange3 = 3수
        planner = SearchPlanner(Policy.load(policy_path), seed=idx, depth=int(method[len('policy_rearrange'):] or 2))
    elif method.startswith('policy_sample'):       # 정책 확률로 뽑은 계획 (예: policy_sample3 = 씨앗 3). 여러 번 계획해 가장 좋은 것 고르기용
        planner = Planner(Policy.load(policy_path), seed=1000 * int(method[len('policy_sample'):] or 0) + idx,
                          temperature=opt.get('temperature', 1.0))
    else:
        planner = Planner(Policy.load(policy_path), search=search if method == 'policy_search' else 0, samples=samples, seed=idx)
    twin = StackTwin(env.core.job, cfg.box_types, cfg.depth, cfg.width, cfg.height, seed=idx) if method.endswith('robust') else None
    motion = audit = None
    if opt.get('motion') or opt.get('audit'):
        from rig.motion import MotionChecker
        if 'mc' not in _MOTION:
            _MOTION['mc'] = MotionChecker(n_boxes=cfg.max_boxes + 8, seed=idx)
        motion, audit = (_MOTION['mc'], None) if opt.get('motion') else (None, _MOTION['mc'])
    from buffer_env import check_motion
    t0, done, n, illegal, stuck, audit_fail, skips, consec = time.time(), False, 0, 0, False, [], 0, 0
    skip_stuck = opt.get('skip_stuck', 0)
    motion_closed = False

    def can_skip():                                 # 막혔을 때 투입 칸 상자를 빼고 이어 갈 수 있나 (연속 skip_stuck 개까지)
        cc = env.core
        return skip_stuck > consec and cc.slots[0] is not None and cc.next < len(cc.job)
    while not done:
        k = planner.choose(env.core, env, twin, motion=motion)
        if k < 0:                                   # 모션 검사를 통과하는 이동이 하나도 없다
            if can_skip():
                done = env.skip_arrival()
                skips, consec = skips + 1, consec + 1
                continue
            if cfg.zone:                            # 더 받지 않고 마감한다 (칸 자리까지 쓴다). 그래도 없으면 칸 상자를 둔 채로 끝
                done, motion_closed = env.finish(), True
                continue
            stuck = True
            break
        c = env.core.cand
        if audit is not None:                       # 고른 이동을 걸러내지 않고 모션 검사만 해 본다
            res = check_motion(env.core, k, audit)
            if not res['ok']:
                audit_fail.append(dict(kind=res['record']['kind'], why=res['why'],
                                       to=res['record']['to'].get('container_mm', res['record']['to'].get('slot'))))
        if cfg.pool and env.core.slots[0] is None and int(c['kind'][k]) == 0:   # 고른 상자를 투입 칸에 넣는다
            env.feed(int(c['jid'][k]))
            a = env.encode(0, 0, int(c['o'][k]), int(c['i'][k]), int(c['j'][k]))
        else:
            a = env.encode_cand(c, k)
        illegal += int(not env.action_masks()[a])
        obs, r, done, _, info = env.step(a)
        sync_twin(env.core, twin)
        n += 1
        if int(c['kind'][k]) == 0:
            consec = 0
        if done and env.core.reason == 'no_space' and can_skip():
            done = env.skip_arrival()
            skips, consec = skips + 1, consec + 1
    info = {**info, **env.core.metrics()}
    m = {k: info[k] for k in ('util', 'util_rig', 'placed', 'kept', 'total', 'left', 'violations', 'stage_moves', 'moves', 'reason',
                              'retrieves', 'abandoned')}
    if stuck:
        m['reason'] = 'motion_stuck'
    m.update(unload_check(env.core.placed))
    m.update(unstable_skipped=planner.stats['unstable_skipped'], twin_skipped=planner.stats['twin_skipped'],
             motion_skipped=planner.stats['motion_skipped'], audit_moves=n if audit is not None else 0, audit_fail=audit_fail,
             boxes_skipped=skips, motion_closed=motion_closed,
             illegal=illegal, sec_per_move=(time.time() - t0) / max(1, n),
             in_zone=int(sum(p['hi'][0] > cfg.depth + 1 for p in env.core.placed)),
             placements=env.export())
    return method, kind, idx, m


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--policy', default=str(HERE / 'policy.npz'))
    ap.add_argument('--sets', nargs='+', default=['team', 'full', 'full_near'])
    ap.add_argument('--methods', nargs='+', default=['greedy_nobuf', 'greedy', 'policy'])
    ap.add_argument('--n', type=int, default=64)
    ap.add_argument('--search', type=int, default=6)
    ap.add_argument('--samples', type=int, default=4)
    ap.add_argument('--support', type=float, default=0.9, help='받침 비율 하한 (pack_env 기본 0.8, 올리는 것만 허용)')
    ap.add_argument('--workers', type=int, default=96)
    ap.add_argument('--zone', action='store_true', help='버퍼 구역(830~1200)도 적재 공간 (칸 간섭 검사 포함)')
    ap.add_argument('--zone-late', action='store_true', help='830 안에 놓을 자리가 없을 때 버퍼 구역을 연다')
    ap.add_argument('--slot-clear', action='store_true', help='버퍼 칸 상자와 부딪히는 자리를 뺀다')
    ap.add_argument('--reach', help='앞면 밀어 넣기 도달 표 (.npz). 주면 닿지 않는 자리는 뺀다')
    ap.add_argument('--arm', help='팔 공간 표 (.npz, rig/arm_table.py). 주면 실린 상자가 팔꿈치 등에 닿는 자리는 뺀다')
    ap.add_argument('--n-stage', type=int, default=3)
    ap.add_argument('--stage-order', help='버퍼로 옮길 때 쓰는 칸과 순서, 예: 1 (문 쪽 칸만). 기본 1,2,3')
    ap.add_argument('--keep-stage', action='store_true', help='zone: 재배치 칸도 늘 비워 둔다 (버퍼 = 투입 칸 + 재배치 칸)')
    ap.add_argument('--box-types', help='상자 종류 JSON, 예: \'{"3": [160, 80, 80], "6": [160, 160, 120]}\' (L, W, H mm). 기본 pack_env 5종')
    ap.add_argument('--shares', help='종류별 도착 비율 JSON. 없으면 균등')
    ap.add_argument('--job-zone', action='store_true', help='작업 크기를 1200 공간 기준으로 (830 모드와 zone 모드를 같은 작업으로 비교)')
    ap.add_argument('--motion', action='store_true', help='실행 전 모션 검사 (rig/): 안 되는 이동은 건너뛴다')
    ap.add_argument('--skip-stuck', type=int, default=0, help='막히면 투입 칸 상자를 빼고 이어 간다 (연속 N 개까지)')
    ap.add_argument('--retrieve', action='store_true', help='실은 상자를 다시 꺼내는 이동도 쓴다')
    ap.add_argument('--audit', action='store_true', help='모션 검사로 고른 이동을 세기만 한다 (걸러내지 않음)')
    ap.add_argument('--feed-tries', type=int, default=0, help='오프라인: 나중 지점부터 넣고 같은 지점 안 투입 순서를 N 가지 미리 굴려 고른다')
    ap.add_argument('--pool', action='store_true', help='오프라인: 목록 전체를 알고 다음 상자와 자리를 같이 고른다 (BufferConfig.pool)')
    ap.add_argument('--center-cell', action='store_true', help='투입 칸 하나를 문 쪽 줄 가운데에 (--n-stage 0 과 같이)')
    ap.add_argument('--front-cell', action='store_true', help='투입 칸 하나를 문 앞 공간 가운데 (컨테이너 밖) 에 (--n-stage 0)')
    ap.add_argument('--top-down', action='store_true', help='위에서 내려놓기도 쓴다 (--reach-top 표와 같이)')
    ap.add_argument('--reach-top', help='위에서 내려놓기 도달 표 (.npz)')
    ap.add_argument('--temperature', type=float, default=1.0, help='policy_sampleN: 순위를 뽑는 온도')
    ap.add_argument('--out', default='eval.json')
    a = ap.parse_args()
    a.zone = a.zone or a.zone_late
    opt = dict(zone=a.zone, zone_late=a.zone_late, slot_clear=a.slot_clear or a.zone, reach=a.reach, n_stage=a.n_stage,
               job_zone=a.job_zone or a.zone,
               motion=a.motion, audit=a.audit, box_types=json.loads(a.box_types) if a.box_types else None,
               shares=json.loads(a.shares) if a.shares else None, skip_stuck=a.skip_stuck, retrieve=a.retrieve,
               stage_order=tuple(int(x) for x in a.stage_order.split(',')) if a.stage_order else (), feed_tries=a.feed_tries,
               keep_stage=a.keep_stage, arm=a.arm, pool=a.pool, center_cell=a.center_cell, front_cell=a.front_cell,
               top_down=a.top_down, reach_top=a.reach_top,
               temperature=a.temperature)
    tasks = [(m, s, i, a.policy, a.search, a.samples, a.support, opt) for s in a.sets for m in a.methods for i in range(a.n)]
    with Pool(a.workers) as pool:
        res = pool.map(episode, tasks, chunksize=1)
    table = {}
    for s in a.sets:
        for m in a.methods:
            r = [x[3] for x in res if x[0] == m and x[1] == s]
            f = lambda key: float(np.mean([x[key] for x in r]))  # noqa: E731
            table[f'{s}/{m}'] = dict(util=f('util'), util_rig=f('util_rig'), kept=f('kept'), in_zone=f('in_zone'),
                                     boxes_skipped=f('boxes_skipped'),
                                     util_min=float(min(x['util'] for x in r)), util_max=float(max(x['util'] for x in r)),
                                     placed=f('placed'), total=f('total'), left=f('left'), violations=f('violations'),
                                     rework=f('rework'), rework_boxes=f('rework_boxes'), rework_stops=f('rework_stops'),
                                     motion_skipped=f('motion_skipped'), motion_stuck=float(np.mean([x['reason'] == 'motion_stuck' for x in r])),
                                     motion_closed=float(np.mean([x['motion_closed'] for x in r])),
                                     audit_fail_rate=float(sum(len(x['audit_fail']) for x in r) / max(1, sum(x['audit_moves'] for x in r))),
                                     rework_jobs=float(np.mean([x['rework'] > 0 for x in r])),
                                     stage_moves=f('stage_moves'), moves_per_box=f('moves') / max(1e-9, f('placed')),
                                     unstable_skipped=f('unstable_skipped'), twin_skipped=f('twin_skipped'),
                                     illegal=int(sum(x['illegal'] for x in r)),
                                     sec_per_move=f('sec_per_move'), n=len(r))
            t = table[f'{s}/{m}']
            print(f"{s:10s} {m:14s} 사용률 {t['util'] * 100:5.1f}% ({t['util_min'] * 100:.1f}~{t['util_max'] * 100:.1f}, 1200 기준 {t['util_rig'] * 100:.1f}%)  "
                  f"버퍼구역 상자 {t['in_zone']:.1f} 칸에둠 {t['kept']:.1f}  모션건너뜀 {t['motion_skipped']:.1f} 모션막힘 {t['motion_stuck'] * 100:.0f}% 모션마감 {t['motion_closed'] * 100:.0f}% "
                  f"모션실패(감사) {t['audit_fail_rate'] * 100:.2f}%  재작업상자 {t['rework_boxes']:.1f}  "
                  f"적재 {t['placed']:5.1f}/{t['total']:5.1f}  위반 {t['violations']:.2f}  하차재작업 {t['rework']:.1f}개 "
                  f"(지점 {t['rework_stops']:.1f}곳, 작업 {t['rework_jobs'] * 100:.0f}%)  버퍼이동 {t['stage_moves']:.1f}  "
                  f"이동/상자 {t['moves_per_box']:.3f}  불안정건너뜀 {t['unstable_skipped']:.2f}  동적검사건너뜀 {t['twin_skipped']:.1f}  "
                  f"마스크밖 {t['illegal']}  "
                  f"{t['sec_per_move'] * 1000:.0f} ms/이동", flush=True)
    Path(a.out).write_text(json.dumps(dict(table=table, episodes=[dict(method=m, set=s, idx=i, **x) for m, s, i, x in res]),
                                      ensure_ascii=False, default=float))
    print('->', a.out)


if __name__ == '__main__':
    main()
