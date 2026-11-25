"""실행용 배치 플래너: 학습한 정책 + (선택) 짧은 탐색 + MuJoCo 안정성 확인.

  search = 0   정책 점수가 가장 높은 이동 (상자 하나에 수 ms)
  search = m   정책 상위 m 개 이동마다, 보이지 않는 뒤쪽 상자를 samples 번 무작위로 채워 정책으로 끝까지
               굴려 보고 평균 보상이 가장 높은 이동을 고른다 (상자 하나에 수 초. 로봇 한 번 이동보다 짧다)
  env          BufferPackEnv(physics=True) 를 주면 고른 자리를 pack_env._settle 로 먼저 떨어뜨려 보고,
               불안정하면 다음 순위로 넘어간다

보이는 정보만 쓴다: 지금 상자들 (투입 칸·버퍼), 다음 lookahead 개, 전체 상자 수.
그 뒤 상자는 지금까지 본 종류 비율로 뽑고, 지점은 마지막으로 보인 지점을 이어 쓴다 (나중 지점부터 도착하므로).
"""
from __future__ import annotations

import numpy as np

from buffer_core import Core, PLACE, RETRIEVE, LOOK
from buffer_policy import Policy, run_episode


def finish(core: Core, policy: Policy, max_steps: int | None = None) -> float:
    tot, n = 0.0, 0
    while not core.done and (max_steps is None or n < max_steps):
        tot += core.step(int(np.argmax(policy.scores(core.state_vector(), core.cand['feat']))))
        n += 1
    return tot


def sample_future(core: Core, rng: np.random.Generator, lookahead: int = LOOK) -> list:
    """보이는 부분 (도착한 상자 + 다음 lookahead 개) 은 그대로 두고 나머지 종류·지점을 다시 뽑은 작업 목록."""
    seen = core.next + lookahead
    if seen >= len(core.job):
        return core.job
    known = core.job[:seen]
    types = sorted(core.cfg.box_types)
    counts = np.ones(len(types))
    for _, k in known:
        counts[types.index(k)] += 1
    p = counts / counts.sum()
    last_stop = known[-1][0]
    rest = [(last_stop, types[int(t)]) for t in rng.choice(len(types), size=len(core.job) - seen, p=p)]
    return known + rest


class Planner:
    def __init__(self, policy: Policy, search: int = 0, samples: int = 4, seed: int = 0, max_steps: int | None = None):
        self.policy, self.search, self.samples, self.max_steps = policy, search, samples, max_steps
        self.rng = np.random.default_rng(seed)
        self.stats = dict(decisions=0, unstable_skipped=0, twin_skipped=0, motion_skipped=0)
        self.last_motion = None

    def ranking(self, core: Core) -> np.ndarray:
        """후보 번호를 선호 순서대로."""
        s = self.policy.scores(core.state_vector(), core.cand['feat'])
        order = np.argsort(-s)
        if self.search <= 1 or len(order) == 1:
            return order
        top = order[: self.search]
        value = np.zeros(len(top))
        for _ in range(self.samples):
            fut = sample_future(core, self.rng)
            for n, k in enumerate(top):
                c = core.clone()
                c.job = fut
                value[n] += c.step(int(k)) + finish(c, self.policy, self.max_steps)
        return np.concatenate([top[np.argsort(-value, kind='stable')], order[self.search:]])

    def _motion_ok(self, core: Core, k: int, motion) -> bool:
        from buffer_env import check_motion
        res = check_motion(core, int(k), motion)
        if res['ok']:
            self.last_motion = (int(k), res)
        else:
            self.stats['motion_skipped'] += 1
        return res['ok']

    def choose(self, core: Core, env=None, twin=None, exclude=(), motion=None) -> int:
        """다음 이동의 후보 번호. 선호 순서대로 보면서
          env    (BufferPackEnv, physics=True)  pack_env._settle 로 떨어뜨려 보고 불안정하면 건너뛴다
          motion (rig.motion.MotionChecker)     집기·대기점·밀어 넣기 자세의 IK 와 팔·상자 충돌을 보고 안 되면 건너뛴다
          twin   (StackTwin)                    쌓인 상자 전체를 움직일 수 있게 두고 몇 mm 어긋나게 놓아 봐서
                                                무언가 밀리거나 기울면 건너뛴다 (실물 대비)
        모두 통과하는 자리가 없으면 버퍼로 옮기기, 그것도 없으면 env·motion 만 통과한 첫 자리."""
        self.stats['decisions'] += 1
        self.last_motion = None
        order = self.ranking(core)
        if exclude:
            order = np.array([k for k in order if int(k) not in exclude], int)
            if len(order) == 0:
                return -1
        if len(order) == 0:
            return -1
        if env is None and twin is None and motion is None:
            return int(order[0])
        cand = core.cand
        fallback, stage = None, None
        for k in order:
            if cand['kind'][k] != PLACE:
                if stage is None and (motion is None or self._motion_ok(core, k, motion)):
                    stage = int(k)
                continue
            src, o, i, j = int(cand['src'][k]), int(cand['o'][k]), int(cand['i'][k]), int(cand['j'][k])
            if env is not None:
                ok, _, _ = env.check_stable(src, o, i, j)
                if not ok:
                    self.stats['unstable_skipped'] += 1
                    continue
            if motion is not None and not self._motion_ok(core, k, motion):
                continue
            if fallback is None:
                fallback = int(k)
            if twin is not None:
                p = core.place_info(src, o, i, j)
                ok, _, _ = twin.test(core.slots[src], p['center'], p['dims'])
                if not ok:
                    self.stats['twin_skipped'] = self.stats.get('twin_skipped', 0) + 1
                    continue
            if stage is not None and self.ranking_is_stage_first(order, stage, k):
                return stage
            return int(k)
        if stage is not None:
            return stage
        if fallback is not None:
            return fallback
        return -1 if motion is not None else int(order[0])      # 모션 검사를 통과한 이동이 없다

    @staticmethod
    def ranking_is_stage_first(order, stage, k):
        """정책이 버퍼로 옮기기를 이 자리보다 앞에 두었으면 그대로 따른다."""
        pos = {int(x): n for n, x in enumerate(order)}
        return pos[stage] < pos[int(k)]


class OrderPlanner(Planner):
    """버퍼로 순서 바로잡기 (온라인). 버퍼 칸에 상자를 들고 있다가, 보이는 상자 (투입 칸 + 버퍼 칸) 중 가장 나중에 내릴
    지점 상자부터 넣는다. 보이는 정보만 쓴다 (칸의 상자 + 다음 LOOK 개).
      투입 칸 상자보다 나중 지점 상자가 칸이나 다음 LOOK 개에 보이고, 버퍼에 빈 칸이 있다 -> 투입 칸 상자를 버퍼로
      그 밖에는 보이는 상자 중 지점 번호가 가장 큰 상자를 넣는다 (지점순 도착이면 그냥 투입 칸 상자를 넣는다)
    자리는 학습 정책 순위로 고른다. 원하는 이동이 안 되면 (자리 없음·불안정·모션) 정책 순위대로 다음 이동."""

    def ranking(self, core: Core) -> np.ndarray:
        order = super().ranking(core)
        c = core.cand
        stops = {s: core.job[j][0] for s, j in enumerate(core.slots) if j is not None}
        if not stops:
            return order
        can_stage = 0 in stops and bool(core.staging_slots()) and core.more_coming()
        seen = list(stops.values()) + [st for st, _ in core.job[core.next:core.next + LOOK]]   # 칸 + 다음 LOOK 개
        if can_stage and stops[0] < max(seen):                                    # 더 나중 지점 상자가 보이면
            first = [k for k in order if c['kind'][k] != PLACE]                  # 투입 칸 상자를 버퍼로 (그 상자를 먼저 넣으려고)
        else:
            top = max(stops.values())
            src = [s for s, v in stops.items() if v == top]
            first = [k for k in order if c['kind'][k] == PLACE and int(c['src'][k]) in src]
        rest = [k for k in order if k not in set(first)]
        return np.array(first + rest, int)


class DigPlanner(Planner):
    """파내고 넣기 (실은 상자 다시 꺼내기, cfg.retrieve=True 와 함께). 투입 칸 상자를 어디 넣어도 먼저 내릴 상자를 막게 되면,
    앞·위가 빈 상자를 최대 max_dig 개 꺼내 버퍼 칸에 두고 그 상자를 넣은 뒤 꺼낸 상자를 다시 넣는다.
    Core 사본으로 미리 굴려 위반이 줄 때만 하고, 단계마다 안정성·모션 검사를 한다 (걸리면 그 순서를 버리고 보통 이동).
    지점순 도착이면 거의 하지 않는다."""

    def __init__(self, policy: Policy, max_dig: int = 3, **kw):
        super().__init__(policy, **kw)
        self.max_dig = max_dig
        self.pending = []
        self.n_in = policy.w['c0.w'].shape[0]               # 정책이 받는 후보 특징 수 (다시 꺼내기 특징은 쓰지 않는다)
        self.stats.update(digs=0, dig_aborted=0)

    def ranking(self, core: Core) -> np.ndarray:
        s = self.policy.scores(core.state_vector(), core.cand['feat'][:, :self.n_in])
        order = np.argsort(-s)
        return np.array([k for k in order if core.cand['kind'][k] != RETRIEVE], int)   # 보통 이동만

    def _best(self, core, mask):
        if core.cand['n'] == 0:
            return None
        idx = np.flatnonzero(mask)
        if len(idx) == 0:
            return None
        s = self.policy.scores(core.state_vector(), core.cand['feat'][:, :self.n_in])
        return int(idx[np.argmax(s[idx])])

    @staticmethod
    def _min_blocks(core, jid):
        c = core.cand
        if c['n'] == 0:
            return None
        m = (c['kind'] == PLACE) & (c['jid'] == jid)
        return int(c['blocks'][m].min()) if m.any() else None

    def dig(self, core: Core):
        """실행할 후보 번호 목록 (꺼내기들, 투입 칸 상자 넣기, 꺼낸 상자 다시 넣기). 이득이 없으면 None."""
        jid = core.slots[0]
        if jid is None or not core.cfg.retrieve:
            return None
        mb = self._min_blocks(core, jid)
        if mb is None or mb == 0:
            return None
        sim, seq = core.clone(), []
        for _ in range(self.max_dig):
            best_r, best_mb = None, mb
            if sim.cand['n'] == 0:
                return None
            for r in np.flatnonzero(sim.cand['kind'] == RETRIEVE):
                s2 = sim.clone()
                s2.step(int(r))
                m2 = self._min_blocks(s2, jid)
                if m2 is not None and m2 < best_mb:
                    best_r, best_mb = int(r), m2
            if best_r is None:
                return None
            sim.step(best_r)
            seq.append(best_r)
            mb = best_mb
            if mb == 0:
                break
        if mb != 0:
            return None
        if sim.cand['n'] == 0:
            return None
        k = self._best(sim, (sim.cand['kind'] == PLACE) & (sim.cand['jid'] == jid) & (sim.cand['blocks'] == 0))
        if k is None:
            return None
        sim.step(k)
        seq.append(k)
        for j in [j for j in sim.slots[1:] if j is not None and j in sim.retrieved]:
            c = sim.cand
            if c['n'] == 0:
                return None
            m = (c['kind'] == PLACE) & (c['jid'] == j)
            if not m.any():
                return None
            kk = self._best(sim, m & (c['blocks'] == c['blocks'][m].min()))
            sim.step(kk)
            seq.append(kk)
        base = core.clone()
        kb = self._best(base, base.cand['kind'] != RETRIEVE) if base.cand['n'] else None
        if kb is not None:
            base.step(kb)
            if sim.violations >= base.violations:
                return None
        return seq

    def _step_ok(self, core, k, env, motion):
        c = core.cand
        if k >= c['n']:
            return False
        if c['kind'][k] == PLACE and env is not None:
            if not env.check_stable(int(c['src'][k]), int(c['o'][k]), int(c['i'][k]), int(c['j'][k]))[0]:
                return False
        if motion is not None and not self._motion_ok(core, k, motion):
            return False
        return True

    def choose(self, core: Core, env=None, twin=None, exclude=(), motion=None) -> int:
        self.last_motion = None
        if self.pending:
            k = self.pending.pop(0)
            if self._step_ok(core, k, env, motion):
                return k
            self.pending = []
            self.stats['dig_aborted'] += 1
        seq = self.dig(core)
        if seq and self._step_ok(core, seq[0], env, motion):
            self.pending = seq[1:]
            self.stats['digs'] += 1
            return seq[0]
        return Planner.choose(self, core, env, twin, exclude, motion)


class SearchPlanner(Planner):
    """적재하면서 버퍼로 계속 재배치 (cfg.retrieve=True 와 함께). 매 이동마다 넣기 / 투입 칸 상자를 버퍼로 / 실은 상자를
    다시 꺼내 버퍼로 를 depth 수 앞까지 빔 탐색한다 (빔 width). 각 수순 끝에서 칸에 있는 상자와 다음 상자를 정책으로
    rollout 수만큼 마저 넣어 보고 평가한다:
        값 = w_util x 사용률 - w_block x 나쁨 - w_move x 추가 이동 - (버퍼에 남은 상자마다 w_hold + w_block x 그 상자를
             지금 넣을 때의 최소 위반, 넣을 곳이 없으면 w_stuck)
    나쁨: objective='blockers' 면 먼저 내릴 상자를 막는 상자 수 (하차 때 옮길 상자에 가깝다), 'pairs' 면 막는 쌍 수.
    추가 이동: 버퍼로 옮기기 1번, 다시 꺼내기 2번 (꺼내기 + 다시 넣기). 버퍼에 남은 상자의 위반을 미리 세어서
    "꺼내 들고만 있으면 위반이 사라져 보이는" 착시를 막는다.
    보이는 정보만 쓴다 (rollout 이 다음 LOOK 개를 넘지 않는다). 고른 첫 이동은 Planner.choose 가 안정성·모션 검사를 한다."""

    def __init__(self, policy: Policy, depth: int = 2, width: int = 4, branch_place: int = 3, branch_ret: int = 3,
                 rollout: int = 3, w_move: float = .25, w_hold: float = .3, w_stuck: float = 2.0, objective: str = 'blockers', **kw):
        super().__init__(policy, **kw)
        self.depth, self.width, self.bp, self.br, self.rollout = depth, width, branch_place, branch_ret, min(rollout, LOOK)
        self.w_move, self.w_hold, self.w_stuck, self.objective = w_move, w_hold, w_stuck, objective
        self.n_in = policy.w['c0.w'].shape[0]
        self.stats.update(searches=0)

    def _scores(self, core):
        return self.policy.scores(core.state_vector(), core.cand['feat'][:, :self.n_in])

    def _children(self, core):
        c = core.cand
        if c['n'] == 0:
            return []
        s = self._scores(core)
        place = np.flatnonzero(c['kind'] == PLACE)
        out = list(place[np.argsort(-s[place])][:self.bp])
        out += list(np.flatnonzero((c['kind'] != PLACE) & (c['kind'] != RETRIEVE)))           # 버퍼로 옮기기
        ret = np.flatnonzero(c['kind'] == RETRIEVE)
        if len(ret):
            out += list(ret[np.lexsort((-s[ret], -c['unblocks'][ret]))][:self.br])         # 위반을 많이 푸는 것부터
        return [int(k) for k in out]

    def _value(self, core):
        sim, n = core.clone(), 0
        while not sim.done and n < self.rollout and sim.cand['n']:
            c = sim.cand
            m = c['kind'] != RETRIEVE
            if not m.any():
                break
            s = self._scores(sim)
            idx = np.flatnonzero(m)
            sim.step(int(idx[np.argmax(s[idx])]))
            n += 1
        cfg = core.cfg
        pen = 0.0
        c = sim.cand
        for j in sim.slots[1:]:
            if j is None:
                continue
            m = (c['kind'] == PLACE) & (c['jid'] == j) if c['n'] else None
            pen += self.w_hold + (cfg.w_block * float(c['blocks'][m].min()) if m is not None and m.any() else self.w_stuck)
        if self.objective == 'blockers' and sim.placed:   # 먼저 내릴 상자를 막는 상자 수 (= 하차 때 옮길 상자에 가깝다)
            B = sim._block_matrix()
            bad = float((B & (sim.pstop[None, :] < sim.pstop[:, None])).any(1).sum())
        else:                                               # 막는 쌍 수 (pack_env 위반)
            bad = float(sim.violations)
        return cfg.w_util * sim.util - cfg.w_block * bad - pen

    def _cost(self, core, k):
        kind = core.cand['kind'][k]
        return 0.0 if kind == PLACE else self.w_move * (2 if kind == RETRIEVE else 1)

    def ranking(self, core: Core) -> np.ndarray:
        base = super().ranking(core) if self.n_in == core.cand_dim else np.argsort(-self._scores(core))
        base = [int(k) for k in base if core.cand['kind'][k] != RETRIEVE]
        roots = self._children(core)
        if len(roots) <= 1:
            return np.array(roots + [k for k in base if k not in roots], int)
        self.stats['searches'] += 1
        beam = []
        for k in roots:
            ch = core.clone()
            cost = self._cost(core, k)
            ch.step(k)
            beam.append((self._value(ch) - cost, cost, ch, k))
        beam.sort(key=lambda t: -t[0])
        beam = beam[:self.width]
        for _ in range(self.depth - 1):
            nxt = []
            for v, cost, node, first in beam:
                kids = self._children(node) if not node.done else []
                if not kids:
                    nxt.append((v, cost, node, first))
                    continue
                for k in kids:
                    ch = node.clone()
                    c2 = cost + self._cost(node, k)
                    ch.step(k)
                    nxt.append((self._value(ch) - c2, c2, ch, first))
            nxt.sort(key=lambda t: -t[0])
            beam = nxt[:self.width]
        best = beam[0][3]
        return np.array([best] + [k for k in roots if k != best] + [k for k in base if k != best and k not in roots], int)


def sync_twin(core: Core, twin) -> None:
    """실제로 놓인 상자를 사본에 반영한다 (step 뒤에 부른다)."""
    if twin is None:
        return
    for p in core.placed:
        if p['jid'] not in twin.nominal:
            twin.place(p['jid'], p['center'], p['dims_xyz'])


def _feed_try(args):
    policy, cfg, job = args
    m = run_episode(Core(cfg), job, policy if isinstance(policy, Policy) else Policy.load(policy))
    return {k: m[k] for k in ('return', 'util', 'placed', 'violations', 'stage_moves')}


def feed_search(job: list, policy, cfg, tries: int = 128, seed: int = 0, pool=None):
    """적재 목록을 미리 다 알 때 (오프라인) 투입 순서 정하기.
    나중 지점부터 넣고 (앞에서 밀어 넣으면 먼저 넣은 상자가 안쪽에 간다), 같은 지점 안 순서는 tries 가지를 정책으로 끝까지
    굴려 보고 (Core: 물리·모션 검사 없이) 정책 목표 (사용률 - 벌점) 가 가장 높은 것을 쓴다. 첫 후보 = 같은 지점 안 목록 순서.
    job = [[지점, 종류], ...] (목록 순서). 돌려주는 것: (투입 순서 = job 번호 목록, 후보마다 지표)."""
    rng = np.random.default_rng(seed)
    stops = sorted({s for s, _ in job}, reverse=True)
    groups = {s: [n for n, b in enumerate(job) if b[0] == s] for s in stops}
    orders = [[n for s in stops for n in (groups[s] if t == 0 else rng.permutation(groups[s]).tolist())]
              for t in range(max(1, tries))]
    args = [(policy, cfg, [job[n] for n in o]) for o in orders]
    res = pool.map(_feed_try, args) if pool is not None else [_feed_try(x) for x in args]
    best = max(range(len(orders)), key=lambda t: res[t]['return'])
    return orders[best], [dict(r, chosen=t == best) for t, r in enumerate(res)]


class GreedyPlanner:
    """기준선: pack_env.greedy 를 버퍼로 넓힌 것 (안쪽 -> 낮은 곳 -> y 작은 쪽, 막혔을 때만 버퍼). 같은 안정성 확인을 한다."""

    def __init__(self):
        self.stats = dict(decisions=0, unstable_skipped=0, twin_skipped=0, motion_skipped=0)
        self.last_motion = None

    def ranking(self, core: Core) -> np.ndarray:
        c = core.cand
        place = np.flatnonzero(c['kind'] == PLACE)
        order = place[np.lexsort((c['j'][place], c['base'][place], c['i'][place]))]
        return np.concatenate([order, np.flatnonzero(c['kind'] != PLACE)])

    choose = Planner.choose
    _motion_ok = Planner._motion_ok
    ranking_is_stage_first = Planner.ranking_is_stage_first


class OnlinePlanner:
    """실제 운용용. 상자가 들어오는 대로 알려 주면 다음 로봇 이동(시작점·끝점, 로봇 좌표)을 돌려준다.

        p = OnlinePlanner('policy_minbuf.npz', total_hint=150)
        p.add([(6, 5), (6, 3)])     # 투입 칸에 들어온 상자와 줄에 보이는 다음 상자들 (도착 순서, (지점, 종류))
        mv = p.next_move()          # dict (from / to ...) 또는 None (둘 수 있는 이동이 없음: 새 상자를 기다리거나 끝)
        ...모션 플래너가 경로를 못 찾으면  mv = p.reject()  (그 이동을 빼고 다음 순위)...
        ...로봇이 mv 를 실행...
        p.commit()                  # 실행이 끝났으면 상태를 넘긴다
        p.close()                   # (zone) 더 올 상자가 없다고 알린다. total_hint 를 주면 저절로 안다
        p.stuck 이면 (모션 검사를 통과하는 이동이 없음) p.skip() 으로 투입 칸 상자를 빼고 이어 가거나, (zone) p.close() 로 마감한다
    physics=True 면 고른 자리를 MuJoCo 로 떨어뜨려 보고 불안정하면 다음 순위를 쓴다.
    motion=True 면 R17 + Piper 모델 (rig/) 로 집기·대기점·밀어 넣기 자세의 IK 와 충돌을 보고 안 되면 다음 순위를 쓴다.
    그때 mv['ik'] 에 자세별 레일 위치와 관절 값이 들어 있다 (이 저장소 모델 기준, 참고용).
    order=True 면 버퍼로 순서를 바로잡는다 (OrderPlanner): 투입 칸 상자보다 나중 지점 상자가 보이면 투입 칸 상자를 버퍼에 둔다.
    dig=True 면 파내고 넣기 (DigPlanner): 실은 상자를 다시 꺼내 (mv['kind'] == 'retrieve') 순서를 바로잡는다.
    """

    def __init__(self, policy='policy_minbuf.npz', cfg=None, search: int = 0, samples: int = 4, total_hint: int | None = None,
                 physics: bool = True, motion: bool = False, seed: int = 0, order: bool = False, dig: bool = False):
        from pathlib import Path
        from buffer_core import BufferConfig
        from buffer_env import BufferPackEnv
        reach = str(Path(__file__).resolve().parent / 'rig' / 'reach_front.npz')
        arm = str(Path(__file__).resolve().parent / 'rig' / 'arm_front.npz')
        # 기본 = 버퍼 최소화 (policy_minbuf.npz): 버퍼는 문 쪽 줄 2칸 (투입 칸 + buffer1) 뿐이고 나머지 바닥은 적재 공간 (zone),
        # 받침 90%, 버퍼 칸 상자와 부딪히는 자리·팔이 닿지 않는 자리 빼기. 830 + 문 쪽 1칸 = policy_door1.npz 와
        # BufferConfig(slot_clear=True, reach=..., stage_order=(1,)). 버퍼 3칸 = policy.npz 와 stage_order=(). policy_s08.npz 는 support_min=0.8
        self.cfg = cfg or BufferConfig(physics=physics, slot_clear=True, reach=reach, arm=arm, zone=True, stage_order=(1,), keep_stage=True,
                                       retrieve=dig, max_retrieve_per_box=3)
        self.policy = policy if isinstance(policy, Policy) else Policy.load(policy)
        cls = DigPlanner if dig else (OrderPlanner if order else Planner)
        self.planner = cls(self.policy, search=search, samples=samples, seed=seed)
        self.env = BufferPackEnv(self.cfg) if physics else None
        self.core = self.env.core if self.env else Core(self.cfg)
        self.core.reset([], total_hint=total_hint)
        self.core.open_end = total_hint is None         # 전체 수를 모르면 close() 전까지 상자가 더 온다고 본다
        self.motion = None
        if motion:
            from rig.motion import MotionChecker
            self.motion = MotionChecker(n_boxes=self.cfg.max_boxes + 8, seed=seed)
        self.pending = None
        self.excluded = set()
        self._stuck = False

    @property
    def stuck(self) -> bool:
        """할 수 있는 이동은 있는데 모션 검사를 통과하는 것이 하나도 없다 (next_move() 가 None). skip() 으로 투입 칸 상자를 뺀다."""
        return self._stuck

    def skip(self):
        """투입 칸 상자를 싣지 않고 뺀다 (작업자가 꺼낸다). 그다음 이동을 돌려준다."""
        self.core.skip_arrival()
        self.pending, self.excluded, self._stuck = None, set(), False
        if self.env is not None:
            self.env._show_slots()
        return self.next_move()

    @property
    def full(self) -> bool:
        """적재를 마감했는지 (zone: 놓을 자리가 없어 더 받지 않는다)."""
        return self.core.closed

    def add(self, boxes):
        self.core.extend(boxes)
        self.pending, self.excluded = None, set()
        if self.env is not None:
            self.env.job, self.env.n_stop = self.core.job, self.core.n_stop
            self.env.hm = self.core.hm
            self.env._build_scene()
            for p in self.core.placed:
                if not p.get('kept'):
                    self.env._show(p['jid'], p)
            self.env._show_slots()

    def close(self):
        """더 올 상자가 없다 (zone 에서 투입 칸 자리까지 채울 수 있게 된다)."""
        c = self.core
        c.open_end, c.total_hint = False, len(c.job)
        c.done, c.reason = False, None
        c.cand = c.candidates()
        c._check_end()
        self.pending, self.excluded = None, set()

    def next_move(self):
        from buffer_env import candidate_record
        c = self.core
        if c.cand['n'] == 0:
            return None
        k = self.planner.choose(c, self.env, exclude=self.excluded, motion=self.motion)
        self._stuck = k < 0 and not self.excluded
        if k < 0:
            return None
        self.pending = k
        rec = candidate_record(c, k, sum(m['kind'] != 'keep' for m in c.moves) + 1)
        lm = self.planner.last_motion
        if lm is not None and lm[0] == k:
            rec['ik'] = lm[1]['poses']
        return rec

    def reject(self):
        """방금 받은 이동을 로봇이 할 수 없을 때 (경로 없음 등): 그 이동을 빼고 다음 순위를 돌려준다."""
        assert self.pending is not None, 'next_move() 먼저'
        self.excluded.add(self.pending)
        self.pending = None
        return self.next_move()

    def commit(self):
        assert self.pending is not None, 'next_move() 먼저'
        jid = self.core.slots[int(self.core.cand['src'][self.pending])]
        n_before = len(self.core.placed)
        self.core.step(self.pending)
        if self.env is not None:
            if len(self.core.placed) > n_before:
                p = next((p for p in self.core.placed if p['jid'] == jid and not p.get('kept')), None)
                if p is not None:
                    self.env._show(p['jid'], p)
            self.env._show_slots()
        self.pending, self.excluded = None, set()

    def kept(self) -> list:
        """zone: 끝날 때 칸에 둔 채로 실은 상자."""
        from buffer_env import kept_boxes
        return kept_boxes(self.core)
