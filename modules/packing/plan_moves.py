"""작업 하나를 처음부터 끝까지 계획하고 로봇 이동 목록(시작점·끝점)을 저장한다.

  uv run python plan_moves.py --kind full --seed 0 --out moves.json
  uv run python plan_moves.py --job my_job.json --out moves.json          # [[지점, 종류], ...] 도착 순서
  uv run --with "imageio[ffmpeg]" python plan_moves.py --kind full --video plan.mp4
  uv run python plan_moves.py --mode big --manifest boxes.json --motion --out moves.json   # 적재 목록 -> 투입 순서 + 전체 이동 (오프라인)

기본 (stable, big) = 목록 방식: 적재 목록 전체를 정책에 주고, 정책이 매번 다음에 넣을 상자 (지금 지점의 남은 상자 중) 와
자리를 같이 고른다. 지점은 나중에 내릴 곳부터. 버퍼는 문 쪽 줄 가운데 투입 칸 하나뿐이라 상자마다 로봇 이동이 하나다.
--plans N 이면 N 번 계획해 (첫 번째 = 정책 그대로, 나머지 = 정책 확률로 뽑음) 가장 많이 실은 것을 쓴다.

이동마다 (moves.json 의 "moves")
  kind        "place" (컨테이너에 넣기) / "stage" (투입 칸 -> 버퍼 칸, 이전 모드만)
  from / to   center_m (상자 중심), quat_wxyz (상자 자세), size_xyz_m (놓인 상태의 x·y·z 치수),
              grasp_m (흡착면 중심 = 툴 끝이 닿는 점), grasp_normal (흡착면 바깥 법선)
              to 가 컨테이너면 pre_insert_m (밀어 넣기 전 대기점, 출구 쪽 40 mm) 과 container_mm 도 있다
좌표는 모두 로봇 좌표 (m) = 월드 - ROBOT_ORIGIN (pack_env.to_planner 와 같다). 축은 월드와 같다.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import mujoco
import numpy as np

from pack_env import C_ORIGIN, ROBOT_ORIGIN, _quat_x90, to_world
from buffer_core import BufferConfig, PLACE, STAGE, RETRIEVE, SLOT_NAMES, make_job, buffer_cells, unload_check
from buffer_env import BufferPackEnv, body_quat, kept_boxes
from buffer_planner import DigPlanner, OrderPlanner, Planner, SearchPlanner, feed_search, sync_twin
from stack_twin import StackTwin
from buffer_policy import Policy

HERE = Path(__file__).resolve().parent
REACH = str(HERE / 'rig' / 'reach_front.npz')
MOTION_RULES = dict(slot_clear=True, reach=REACH)      # 버퍼 칸 상자와 부딪히는 자리, 팔이 닿지 않는 자리를 뺀다
BIG_TYPES = {1: (80, 80, 100), 2: (160, 80, 80), 3: (160, 80, 100), 4: (160, 160, 80), 5: (160, 160, 100)}   # 1·2번 상자를 붙인 큰 5종
BIG_SHARES = {1: 1, 2: 1, 3: 2, 4: 2, 5: 3}                                                                  # 도착 비율 (11/11/22/22/33%)
DOOR_CELL = dict(stage_order=(1,))      # 재배치 칸은 문 쪽 buffer1 하나만. 컨테이너 쪽 칸(buffer2·3)에 상자를 두면 팔이 컨테이너 앞쪽으로 들어가는 길을 막는다
ARM = str(HERE / 'rig' / 'arm_front.npz')               # 팔 공간 표: 실린 상자가 팔꿈치 등 (베이스 ~ link4) 에 닿는 자리를 뺀다
REACH_TOP = str(HERE / 'rig' / 'reach_top.npz')         # 위에서 내려놓기 도달 표 (상자 윗면 중심에서 툴 -z)
MIN_BUFFER = dict(zone=True, keep_stage=True, arm=ARM, **DOOR_CELL)   # 버퍼 = 문 쪽 줄 2칸 (투입 칸 + buffer1) 뿐. 나머지 바닥 (깊이 ~1020) 은 처음부터 적재 공간
LIST_ONE = dict(zone=True, n_stage=0, center_cell=True, pool=True, arm=ARM)   # 목록 방식: 버퍼 = 입구 쪽 맨 앞줄 가운데 투입 칸 하나 (경인 최종 도면:
                                                                            #   적재 X41 ~ X1241, 입구 앞 여유 41 mm 라 칸은 적재 구역 안)
BIG = dict(box_types=BIG_TYPES, shares=BIG_SHARES)
MODES = {'stable': (HERE / 'policy_list.npz', 0.9, dict(**LIST_ONE, **MOTION_RULES)),         # 목록 방식 (기본). 받침 90%
         'big': (HERE / 'policy_big_list.npz', 0.9, dict(**BIG, **LIST_ONE, **MOTION_RULES, top_down=True, reach_top=REACH_TOP)),
                                                                            # 큰 상자 5종, 목록 방식 + 입구 쪽 위에서 내려놓기
         'stable_minbuf': (HERE / 'policy_minbuf.npz', 0.9, dict(**MIN_BUFFER, **MOTION_RULES)),      # 이전 기본: 문 쪽 2칸, 들어오는 순서대로
         'big_minbuf': (HERE / 'policy_big_minbuf.npz', 0.9, dict(**BIG, **MIN_BUFFER, **MOTION_RULES)),
         'stable830': (HERE / 'policy_door1.npz', 0.9, dict(**DOOR_CELL, **MOTION_RULES)),   # 적재 830 + 투입 칸 + 문 쪽 버퍼 1칸
         'big830': (HERE / 'policy_big_door1.npz', 0.9, dict(**BIG, **DOOR_CELL, **MOTION_RULES)),  # 큰 상자 5종 (약 80개), 830 + 문 쪽 버퍼 1칸
         'stable3': (HERE / 'policy.npz', 0.9, MOTION_RULES),           # 적재 830 + 재배치 버퍼 3칸. 실물 오차 시험에서 넘어짐 0
         'big3': (HERE / 'policy_big.npz', 0.9, dict(**BIG, **MOTION_RULES)),   # 큰 상자 5종, 재배치 버퍼 3칸
         'max': (HERE / 'policy_s08.npz', 0.8, MOTION_RULES),           # pack_env 기본 받침 80%, 사용률 최대 (버퍼 3칸)
         'zone': (HERE / 'policy_zone.npz', 0.9, dict(zone=True, **MOTION_RULES))}  # 버퍼 구역(830~1200)까지 채운다


def _slerp(q0, q1, t):
    q0, q1 = np.asarray(q0, float), np.asarray(q1, float)
    d = float(q0 @ q1)
    if d < 0:
        q1, d = -q1, -d
    if d > 0.9995:
        q = q0 + t * (q1 - q0)
    else:
        th = np.arccos(d)
        q = (np.sin((1 - t) * th) * q0 + np.sin(t * th) * q1) / np.sin(th)
    return q / np.linalg.norm(q)


class Recorder:
    """이동을 영상으로: 칸에서 들어 올림 -> 돌리며 출구 앞 대기점으로 -> 밀어 넣기."""

    def __init__(self, env: BufferPackEnv, path: str, fps: int = 20, frames: int = 18, size=(720, 1280)):
        import imageio
        self.env, self.fps, self.frames = env, fps, frames
        self.writer = imageio.get_writer(path, fps=fps, quality=8, macro_block_size=8)
        self.r = mujoco.Renderer(env.m, *size)
        self.cam = mujoco.MjvCamera()
        env._set_cam(self.cam)
        R = np.array([[0., 1., 0.], [0., 0., 1.], [1., 0., 0.]])
        self.q_flat = np.zeros(4)
        mujoco.mju_mat2Quat(self.q_flat, R.ravel())

    def shot(self, n=1):
        self.r.update_scene(self.env.d, self.cam)
        img = self.r.render()
        for _ in range(n):
            self.writer.append_data(img)

    def move(self, kind, jid, src, dst, o=0, center=None):
        env, core = self.env, self.env.core
        stop, k = core.job[jid]
        L, W, H = env.cfg.box_types[k]
        q0 = self.q_flat
        if kind == RETRIEVE:                        # 컨테이너에서 앞으로 빼고 -> 눕혀서 -> 버퍼 칸으로
            c0, _ = to_world(center, (0, 0, 0))
            qc = np.array([1.0, 0, 0, 0]) if o == 0 else _quat_x90()
            x1, y1 = core.cells[dst]
            p1, _ = to_world((x1, y1, H / 2), (0, 0, 0))
            out = c0 + (-0.12, 0, 0.002)
            way = [c0, out, out + (0, 0, 0.1), p1 + (0, 0, 0.12), p1]
            qs = [qc, qc, q0, q0, q0]
        else:
            x, y = core.cells[src]
            p0, _ = to_world((x, y, H / 2), (0, 0, 0))
        if kind == RETRIEVE:
            pass
        elif kind == STAGE:
            x1, y1 = core.cells[dst]
            p1, _ = to_world((x1, y1, H / 2), (0, 0, 0))
            q1 = q0
            way = [p0, p0 + (0, 0, 0.12), p1 + (0, 0, 0.12), p1]
            qs = [q0, q0, q1, q1]
        elif o >= 2:                                # 위에서 내려놓기: 들어 올려 -> 자리 위로 -> 수직으로 내려놓기
            p1, _ = to_world(center, (0, 0, 0))
            q1 = body_quat(o)
            lift = p0 + (0, 0, 0.12)
            above = p1 + (0, 0, 0.12)
            way = [p0, lift, above, p1 + (0, 0, 0.02), p1]
            qs = [q0, q0, q1, q1, q1]
        else:
            p1, _ = to_world(center, (0, 0, 0))
            q1 = np.array([1.0, 0, 0, 0]) if o == 0 else _quat_x90()
            pre = p1 + (-0.12, 0, 0.002)
            lift = p0 + (0, 0, 0.12)
            way = [p0, lift, pre, p1 + (0, 0, 0.002), p1]
            qs = [q0, q0, q1, q1, q1]
        mid = env.box_mocap[jid]
        segs = len(way) - 1
        for s in range(segs):
            n = max(2, self.frames // segs)
            for t in np.linspace(0, 1, n, endpoint=False):
                env.d.mocap_pos[mid] = way[s] + t * (way[s + 1] - way[s])
                env.d.mocap_quat[mid] = _slerp(qs[s], qs[s + 1], t)
                mujoco.mj_forward(env.m, env.d)
                self.shot()

    def close(self):
        self.shot(self.fps)
        self.writer.close()
        self.r.close()


def run(job, policy: Policy, search=0, samples=4, physics=True, video=None, seed=0, cfg=None, log_every=0, robust=False,
        fps=20, frames=18, motion=False, skip_stuck=0, order=False, dig=False, rearrange=None, temperature=0.0):
    cfg = cfg or BufferConfig(physics=physics)
    env = BufferPackEnv(cfg, render_mode='rgb_array' if video else None)
    obs, info = env.reset(options={'job': job})
    if rearrange is not None:                       # 적재하면서 버퍼로 계속 재배치 (빔 탐색)
        planner = SearchPlanner(policy, seed=seed, **rearrange)
    else:
        planner = (DigPlanner if dig else OrderPlanner if order else Planner)(policy, search=search, samples=samples, seed=seed)
        planner.temperature = temperature           # > 0: 순위를 정책 확률로 뽑는다 (--plans)
    mc = None
    if motion:
        from rig.motion import MotionChecker
        mc = MotionChecker(n_boxes=cfg.max_boxes + 8, seed=seed)
    ik = {}                                         # 로봇 이동 번호 -> 자세별 레일·관절 (모션 검사 결과)
    twin = StackTwin(env.core.job, cfg.box_types, cfg.depth, cfg.width, cfg.height, seed=seed) if robust else None
    rec = Recorder(env, video, fps=fps, frames=frames) if video else None
    if rec:
        rec.shot(10)
    t0, done, n, stuck, skips, consec = time.time(), False, 0, False, 0, 0
    skip_log = []                                   # [그때까지 한 로봇 이동 수, 뺀 상자 job 번호]
    motion_closed = False                           # zone: 모션 검사를 통과하는 이동이 없어 마감했다

    def can_skip():                                 # 막혔을 때 투입 칸 상자를 빼고 이어 갈 수 있나 (연속 skip_stuck 개까지)
        c = env.core
        return skip_stuck > consec and c.slots[0] is not None and c.next < len(c.job)
    while not done:
        core = env.core
        k = planner.choose(core, env if physics else None, twin, motion=mc)
        if k < 0:                                   # 모션 검사를 통과하는 이동이 없다
            if can_skip():
                skip_log.append([n, env.core.slots[0]])
                done = env.skip_arrival()
                skips, consec = skips + 1, consec + 1
                continue
            if cfg.zone:                            # 더 받지 않고 마감한다 (칸 자리까지 쓴다). 그래도 없으면 칸 상자를 둔 채로 끝
                done, motion_closed = env.finish(), True
                continue
            stuck = True
            break
        if mc is not None and planner.last_motion is not None and planner.last_motion[0] == k:
            ik[n + 1] = planner.last_motion[1]['poses']
        cand = core.cand
        kind, src, dst = int(cand['kind'][k]), int(cand['src'][k]), int(cand['dst'][k])
        o, i, j = int(cand['o'][k]), int(cand['i'][k]), int(cand['j'][k])
        if cfg.pool and core.slots[0] is None and kind == PLACE:      # 목록 방식: 고른 상자를 투입 칸에 넣는다 (투입 순서)
            env.feed(int(cand['jid'][k]))
            a = env.encode(PLACE, 0, o, i, j)
        else:
            a = env.encode_cand(cand, k)
        assert env.action_masks()[a], 'policy chose an action outside the official mask'
        if rec:
            if kind == RETRIEVE:
                pb = core.placed[int(cand['pidx'][k])]
                rec.move(RETRIEVE, pb['jid'], None, dst, pb['o'], pb['center'])
            else:
                center = core.place_info(src, o, i, j)['center'] if kind == PLACE else None
                rec.move(kind, core.slots[src], src, dst, o, center)
        obs, r, done, _, info = env.step(a)
        sync_twin(env.core, twin)
        n += 1
        if kind == PLACE:
            consec = 0
        if done and env.core.reason == 'no_space' and can_skip():
            skip_log.append([n, env.core.slots[0]])
            done = env.skip_arrival()
            skips, consec = skips + 1, consec + 1
        if log_every and n % log_every == 0:
            print(f"  이동 {n}: 적재 {info['placed']}, 사용률 {info['util'] * 100:.1f}%", flush=True)
    if rec:
        rec.close()
    info = {**info, **env.core.metrics()}
    m = {k: info[k] for k in ('util', 'util_rig', 'placed', 'kept', 'total', 'left', 'violations', 'stage_moves', 'retrieves', 'moves',
                              'moves_per_box', 'reason')}
    if stuck:
        m['reason'] = 'motion_stuck'
    m.update(unload_check(env.core.placed))
    m.update(unstable_skipped=planner.stats['unstable_skipped'], twin_skipped=planner.stats['twin_skipped'],
             motion_skipped=planner.stats['motion_skipped'], boxes_skipped=skips, skip_log=skip_log, motion_closed=motion_closed, seconds=time.time() - t0,
             ms_per_move=(time.time() - t0) / max(1, n) * 1000)
    return env, m, ik


def read_manifest(path, box_types: dict) -> list:
    """적재 목록 -> [(상자 id, 지점, 종류)] (목록 순서). 지점 1 = 먼저 내리는 곳.
    JSON: [{"id": "A01", "stop": 3, "type": 5}, {"id": "A02", "stop": 1, "size": [160, 80, 100]}, ...] 또는 {"boxes": [...]}
    CSV : 머리줄 id,stop,type 또는 id,stop,L,W,H (mm)
    크기로 주면 이 모드의 상자 종류에서 바닥면 (L, W 순서 무관) 과 높이 H 가 같은 것을 찾는다."""
    p = Path(path)
    if p.suffix.lower() == '.csv':
        import csv
        rows = [{k.strip(): (v or '').strip() for k, v in r.items()} for r in csv.DictReader(p.read_text(encoding='utf-8-sig').splitlines())]
        rows = [r if r.get('type') else dict(r, size=[r['L'], r['W'], r['H']]) for r in rows]
    else:
        d = json.loads(p.read_text(encoding='utf-8'))
        rows = d['boxes'] if isinstance(d, dict) else d
    by_size = {(min(L, W), max(L, W), H): k for k, (L, W, H) in box_types.items()}
    boxes = []
    for n, r in enumerate(rows):
        bid = str(r.get('id') or f'box{n + 1:03d}')
        if r.get('type') not in (None, ''):
            k = int(r['type'])
        else:
            L, W, H = (int(round(float(v))) for v in r['size'])
            k = by_size.get((min(L, W), max(L, W), H))
        if k not in box_types:
            raise SystemExit(f"{bid}: 이 모드의 상자 종류에 없다 ({r}). 종류 = {box_types}")
        boxes.append((bid, int(r['stop']), k))
    return boxes


def write_csv(path, moves: list) -> None:
    """이동 목록 -> 표 (한 줄 = 로봇 이동 하나, 실행 순서). center_mm·dims_xyz_mm = 컨테이너 좌표 mm, *_m = 로봇 좌표 m."""
    import csv
    v3 = lambda v: '' if v is None else ' '.join(f'{x:.4f}' for x in v)  # noqa: E731
    with open(path, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['step', 'box_id', 'stop', 'type', 'kind', 'approach', 'from', 'to', 'center_mm', 'dims_xyz_mm', 'orientation',
                    'from_grasp_m', 'to_grasp_m', 'to_pre_insert_m'])
        for mv in moves:
            c = mv['to'].get('container_mm') or mv['from'].get('container_mm') or {}
            w.writerow([mv['step'], mv.get('box_id', mv['box']), mv['stop'], mv['type'], mv['kind'], mv.get('approach', ''),
                        mv['from'].get('slot', 'container'), mv['to'].get('slot', 'container'),
                        ' '.join(f'{x:g}' for x in c.get('center', [])), ' '.join(f'{x:g}' for x in c.get('dims_xyz', [])),
                        mv.get('orientation', ''), v3(mv['from']['grasp_m']), v3(mv['to']['grasp_m']), v3(mv['to'].get('pre_insert_m'))])


def list_report(core, ids: list) -> tuple:
    """목록 방식의 투입 순서 (정책이 고른 순서. 투입 칸에 넣는 순서 = 로봇 이동 순서) 와 싣지 못한 상자."""
    kept = {m['jid'] for m in core.moves if m['kind'] == 'keep'}
    feed = [dict(feed=n + 1, id=ids[j], stop=core.job[j][0], type=core.job[j][1], status='kept' if j in kept else 'loaded')
            for n, j in enumerate(core.fed)]
    fed = set(core.fed)
    missed = [dict(id=ids[j], stop=s, type=k, why='no_space') for j, (s, k) in enumerate(core.job) if j not in fed]
    return feed, missed


def manifest_report(core, ids: list, skip_log: list) -> tuple:
    """오프라인 계획의 투입 순서 (투입 칸에 넣을 상자 순서, 상태) 와 싣지 못한 상자."""
    loaded = {p['jid'] for p in core.placed} | {m['jid'] for m in core.moves if m['kind'] == 'keep'}
    removed = {j: n for n, j in skip_log}
    feed, missed = [], []
    for j in range(len(core.job)):
        stop, k = core.job[j]
        row = dict(feed=j + 1, id=ids[j], stop=stop, type=k)
        if j >= core.next:
            missed.append(dict(row, why='not_fed'))     # 실을 자리가 없어 넣지 않는다
            continue
        if j in loaded:
            row['status'] = 'loaded'
        elif j in removed:                              # 투입 칸에서 꺼내 다음 차로 (로봇 이동 n 번 뒤)
            row.update(status='skipped', remove_after_step=removed[j])
        else:                                           # 끝날 때 칸에 남음 (다음 차로)
            row.update(status='left_in_cell', slot=SLOT_NAMES[core.slots.index(j)])
        feed.append(row)
        if row['status'] != 'loaded':
            missed.append(dict(row, why=row['status']))
    return feed, missed


def _plan_try(args):
    """--plans 의 계획 하나 (다른 프로세스): 지표만 돌려준다."""
    job, policy_path, cfg, seed, temperature, kw = args
    env, m, _ = run(job, Policy.load(policy_path), seed=seed, cfg=cfg, temperature=temperature, **kw)
    return {k: m[k] for k in ('util_rig', 'placed', 'kept', 'violations', 'moves', 'unstable_skipped', 'motion_skipped', 'reason')}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--mode', choices=list(MODES), default='stable',
                    help='stable: 목록 방식 (목록 전체를 보고 다음 상자와 자리를 정함, 버퍼 = 가운데 투입 칸 하나), 받침 90%% (기본) / '
                         'big: 큰 상자 5종, 목록 방식 / stable_minbuf·big_minbuf: 이전 기본 (문 쪽 2칸, 들어오는 순서대로) / '
                         'stable830·big830: 적재 830 + 문 쪽 버퍼 1칸 / stable3·big3: 830 + 재배치 버퍼 3칸 / max: 받침 80%% / '
                         'zone: 버퍼 구역까지 채움 (재배치 3칸)')
    ap.add_argument('--policy', help='정책 파일 (주지 않으면 --mode 에 맞는 것)')
    ap.add_argument('--job', help='작업 목록 JSON: [[지점, 종류], ...] 도착 순서')
    ap.add_argument('--kind', default='full', choices=['team', 'team_random', 'full', 'full_near', 'full_random', 'mixed'])
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--search', type=int, default=0, help='0 = 정책만, m = 상위 m 개 이동을 굴려 보고 고름')
    ap.add_argument('--samples', type=int, default=4)
    ap.add_argument('--no-physics', action='store_true', help='MuJoCo 안정성 확인(pack_env._settle)을 끈다')
    ap.add_argument('--robust', action='store_true', help='동적 적재 검사(stack_twin)를 켠다 (실험용: 평가에서 도움이 되지 않았다)')
    ap.add_argument('--stage-order', help='버퍼 칸 고르는 순서, 예: 1,3,2 (기본 1,2,3)')
    ap.add_argument('--skip-stuck', type=int, default=0, help='막히면 투입 칸 상자를 빼고 (다음 트럭으로) 이어 간다. 연속 N 개까지')
    ap.add_argument('--order', action='store_true', help='버퍼로 순서 바로잡기 (나중 지점 상자가 보이면 투입 칸 상자를 버퍼에 들고 있는다)')
    ap.add_argument('--dig', action='store_true', help='파내고 넣기: 실은 상자를 다시 꺼내 순서를 바로잡는다 (DigPlanner)')
    ap.add_argument('--rearrange', type=int, default=0, metavar='DEPTH',
                    help='적재하면서 버퍼로 계속 재배치: 넣기·버퍼로 옮기기·다시 꺼내기를 DEPTH 수 앞까지 탐색 (SearchPlanner)')
    ap.add_argument('--rearrange-objective', choices=['pairs', 'blockers'], default='pairs',
                    help='탐색이 줄일 것: pairs = 막는 쌍 수 (재작업을 가장 많이 줄임), blockers = 막는 상자 수 (이동이 적음)')
    ap.add_argument('--move-cost', type=float, default=.25, help='탐색에서 추가 로봇 이동 1번의 비용 (위반 1건 = 0.5)')
    ap.add_argument('--motion', action='store_true', help='R17 + Piper 모델 (rig/) 로 모션 검사: 안 되는 이동은 다음 순위. 이동마다 ik 를 적는다')
    ap.add_argument('--manifest', help='적재 목록 JSON/CSV (상자 id, 지점, 종류 또는 크기). 투입 순서를 정하고 전체 이동을 미리 계획한다 '
                                       '(출력 feed_order = 투입 칸에 넣을 순서, 이동마다 box_id)')
    ap.add_argument('--feed-tries', type=int, default=128, help='--manifest: 같은 지점 안 투입 순서 후보 수 (정책으로 미리 굴려 가장 좋은 것)')
    ap.add_argument('--plans', type=int, default=16, help='목록 방식: N 번 계획해 가장 많이 실은 것 (첫 번째 = 정책 그대로, 나머지 = 확률로 뽑음)')
    ap.add_argument('--temperature', type=float, default=1.0, help='--plans: 순위를 뽑는 온도')
    ap.add_argument('--workers', type=int, default=16, help='--feed-tries·--plans 를 나눠 돌릴 프로세스 수')
    ap.add_argument('--csv', help='이동 목록 표 (CSV). --manifest 이면 주지 않아도 --out 이름으로 만든다')
    ap.add_argument('--out', default='moves.json')
    ap.add_argument('--video')
    ap.add_argument('--fps', type=int, default=20)
    ap.add_argument('--frames-per-move', type=int, default=18)
    a = ap.parse_args()
    policy_path, support, rules = MODES[a.mode]
    if a.dig or a.rearrange:
        rules = dict(rules, retrieve=True, max_retrieve_per_box=1 if a.rearrange else 3)
    if a.stage_order:
        rules = dict(rules, stage_order=tuple(int(x) for x in a.stage_order.split(',')))
    cfg = BufferConfig(physics=not a.no_physics, support_min=support, **rules)
    policy = Policy.load(a.policy or policy_path)
    ids = tries = None
    if a.manifest and cfg.pool:                     # 목록 방식: 목록 전체를 정책에 준다. 투입 순서도 정책이 정한다
        boxes = read_manifest(a.manifest, cfg.box_types)
        ids, job = [b[0] for b in boxes], [[s, k] for _, s, k in boxes]
    elif a.manifest:                                # 오프라인 (이전 모드): 나중 지점부터 + 같은 지점 안 순서를 미리 굴려 고른다
        boxes = read_manifest(a.manifest, cfg.box_types)
        listed = [[s, k] for _, s, k in boxes]
        if a.feed_tries > 1:
            from multiprocessing import Pool
            with Pool(min(a.workers, a.feed_tries)) as pool:
                order, tries = feed_search(listed, policy, cfg, a.feed_tries, a.seed, pool)
        else:
            order, tries = feed_search(listed, policy, cfg, 1)
        ids, job = [boxes[n][0] for n in order], [listed[n] for n in order]
        best = next(t for t in tries if t['chosen'])
        print(f"투입 순서: 나중 지점부터, 같은 지점 안 순서 {len(tries)}가지를 미리 굴려 봄 (물리·모션 검사 없이) -> "
              f"목록 순서 사용률 {tries[0]['util'] * 100:.1f}% / 상자 {tries[0]['placed']}, 고른 순서 {best['util'] * 100:.1f}% / {best['placed']}")
    else:
        job = json.loads(Path(a.job).read_text()) if a.job else make_job(np.random.default_rng(a.seed), cfg, a.kind)
    kw = dict(search=a.search, samples=a.samples, physics=not a.no_physics, robust=a.robust, motion=a.motion, skip_stuck=a.skip_stuck,
              order=a.order, dig=a.dig,
              rearrange=dict(depth=a.rearrange, objective=a.rearrange_objective, w_move=a.move_cost) if a.rearrange else None)
    best_t, plan_tries = 0, None
    if cfg.pool and a.plans > 1:                    # 여러 번 계획 (물리·모션 검사 포함) 해서 가장 많이 실은 것
        from multiprocessing import Pool
        args = [(job, str(a.policy or policy_path), cfg, a.seed + t, a.temperature if t else 0.0, kw) for t in range(a.plans)]
        with Pool(min(a.workers, a.plans)) as pool:
            plan_tries = pool.map(_plan_try, args, chunksize=1)
        best_t = max(range(a.plans), key=lambda t: (plan_tries[t]['violations'] == 0, plan_tries[t]['util_rig'], -plan_tries[t]['moves']))
        plan_tries = [dict(r, chosen=t == best_t) for t, r in enumerate(plan_tries)]
        print(f"계획 {a.plans}번: 정책 그대로 {plan_tries[0]['util_rig'] * 100:.1f}% / 상자 {plan_tries[0]['placed']}, "
              f"가장 좋은 것 (#{best_t}) {plan_tries[best_t]['util_rig'] * 100:.1f}% / {plan_tries[best_t]['placed']}")
    env, m, ik = run(job, policy, video=a.video, seed=a.seed + best_t, cfg=cfg, log_every=20, fps=a.fps, frames=a.frames_per_move,
                     temperature=a.temperature if best_t else 0.0, **kw)
    if plan_tries is not None and (m['placed'], round(m['util_rig'], 9)) != (plan_tries[best_t]['placed'], round(plan_tries[best_t]['util_rig'], 9)):
        print('주의: 고른 계획을 다시 돌린 결과가 다르다', m['placed'], plan_tries[best_t]['placed'])
    moves = env.robot_moves()
    for mv in moves:
        if mv['step'] in ik:
            mv['ik'] = ik[mv['step']]
        if ids is not None:
            mv['box_id'] = ids[int(mv['box'][1:]) - 1]
    out = dict(frame='robot (m) = world - robot_origin; axes = world', unit='m', mode=a.mode, support_min=support,
               zone=cfg.zone, robot_origin_world_m=ROBOT_ORIGIN.tolist(), container_origin_world_m=C_ORIGIN.tolist(),
               container_mm=dict(depth=env.core.D, load_depth=env.core.D if cfg.zone else cfg.depth, width=cfg.width, height=cfg.height,
                                 axes='x 출구 방향 = 월드 -x, y = 월드 -y, z 위'),
               load_note=('목록 방식: 버퍼는 문 쪽 줄 가운데 투입 칸 (arrival) 하나뿐이고 나머지 바닥 전체가 적재 공간. '
                          '상자마다 로봇 이동 하나 (투입 칸 -> 끝 자리). 이동 순서 = feed_order (투입 칸에 넣는 순서)'
                          if cfg.pool else
                          'zone: 쓰는 버퍼 칸 (buffer_cells_robot_m) 을 뺀 바닥 전체가 적재 공간. 마지막 상자들은 칸에 둔 채로 싣는다 (kept)')
               if cfg.zone else None,
               buffer_cells_robot_m={SLOT_NAMES[s]: (to_world((x, y, 0), (0, 0, 0))[0] - ROBOT_ORIGIN).round(4).tolist()
                                     for s, (x, y) in enumerate(buffer_cells(cfg)[:env.core.n_slots])
                                     if not cfg.stage_order or s == 0 or s in cfg.stage_order},       # 쓰는 칸만
               ik_note='ik = 이 저장소 R17 + Piper 모델 (rig/model.py, 레일 0 = 베이스 x 0.2 m, 툴 80 mm) 의 참고 해' if a.motion else None,
               box_types={str(k): list(v) for k, v in cfg.box_types.items()},
               job=[list(b) for b in env.core.job], metrics=m, placements=env.export(), moves=moves, kept=kept_boxes(env.core))
    if cfg.pool:
        out['feed_order'], out['not_loaded'] = list_report(env.core, ids or [f'b{n + 1:03d}' for n in range(len(env.core.job))])
        out['plan_tries'] = plan_tries
    elif ids is not None:
        out['feed_order'], out['not_loaded'] = manifest_report(env.core, ids, m['skip_log'])
        out['feed_search'] = tries
    if ids is not None:
        for p in out['placements']:
            p['box_id'] = ids[int(p['id'][1:]) - 1]
    Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1))
    csv_path = a.csv or (str(Path(a.out).with_suffix('.csv')) if ids is not None else None)
    if csv_path:
        write_csv(csv_path, moves)
    if ids is not None:
        print('투입 칸에 넣을 순서:', ' '.join(f"{r['id']}" for r in out['feed_order']))
        print(f"{'순서':>4} {'상자':>8} {'지점':>4} {'종류':>4} {'이동':6} {'시작':8} {'끝':9} {'컨테이너 자리 중심 (mm)':24} 방향")
        for mv in moves[:12]:
            c = mv['to'].get('container_mm', {}).get('center')
            print(f"{mv['step']:4d} {mv['box_id']:>8} {mv['stop']:4d} {mv['type']:4d} {mv['kind']:6} {mv['from']['slot']:8} {mv['to']['slot']:9} "
                  f"{str(c) if c else '-':24} {mv.get('orientation', '-')}")
        print(f"  ... 전체 {len(moves)}개 이동 -> {csv_path}. 싣지 못한 상자 {len(out['not_loaded'])}개"
              + (': ' + ' '.join(r['id'] for r in out['not_loaded'][:20]) if out['not_loaded'] else ''))
    print(f"사용률 {m['util'] * 100:.1f}% (1200 전체 기준 {m['util_rig'] * 100:.1f}%), 상자 {m['placed']}/{m['total']} 적재 "
          f"(칸에 둠 {m['kept']}, 뺀 상자 {m['boxes_skipped']}), 하차 순서 위반 {m['violations']}, 하차 재작업 상자 {m['rework_boxes']}, "
          f"버퍼 이동 {m['stage_moves']}, 로봇 이동 {m['moves']}회, 불안정해서 건너뜀 {m['unstable_skipped']}, "
          f"모션 검사로 건너뜀 {m['motion_skipped']}, {m['ms_per_move']:.0f} ms/이동, 종료 {m['reason']} -> {a.out}")


if __name__ == '__main__':
    main()
