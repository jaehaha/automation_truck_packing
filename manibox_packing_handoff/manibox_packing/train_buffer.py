"""버퍼 배치 정책 학습: greedy 모방(BC)으로 시작 -> GRPO.

GRPO: 같은 작업(같은 도착 순서)을 여러 번 샘플링으로 굴려, 그 그룹 안에서 보상을 표준화한 값을
그 에피소드의 모든 결정에 이점(advantage)으로 주고 PPO-clip 으로 갱신한다.
보상 = 10 x 부피 사용률 - w_block x 하차 순서 위반 - w_stage x 버퍼 이동 (buffer_core.Core.step).

torch 가 필요하다 (예: conda 환경). 롤아웃은 CPU 여러 프로세스에서 numpy 정책으로 돈다.
  python train_buffer.py --output runs/buf_v1
결과: runs/.../policy.npz (가장 좋은 dev 점수), last.npz, log.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import time
from multiprocessing import get_context
from pathlib import Path

os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from buffer_core import BufferConfig, Core, greedy_index, make_job  # noqa: E402
from buffer_policy import Policy  # noqa: E402


class Net(nn.Module):
    def __init__(self, state_dim: int, cand_dim: int, width: int = 256, cw: int = 128):
        super().__init__()
        self.s0, self.s1 = nn.Linear(state_dim, width), nn.Linear(width, cw)
        self.c0, self.c1 = nn.Linear(cand_dim, cw), nn.Linear(cw, cw)
        self.h0, self.h1 = nn.Linear(3 * cw, cw), nn.Linear(cw, 1)

    def forward(self, s, c):
        """s (B, S), c (B, K, F) -> logits (B, K)"""
        g = lambda x: F.gelu(x, approximate='tanh')  # noqa: E731
        hs = g(self.s1(g(self.s0(s))))[:, None, :]
        hc = g(self.c1(g(self.c0(c))))
        z = torch.cat([hc, hs.expand_as(hc), hc * hs], -1)
        return self.h1(g(self.h0(z)))[..., 0]

    def export(self) -> dict:
        out = {}
        for n in ('s0', 's1', 'c0', 'c1', 'h0', 'h1'):
            lin = getattr(self, n)
            out[f'{n}.w'] = lin.weight.detach().cpu().numpy().T.copy()
            out[f'{n}.b'] = lin.bias.detach().cpu().numpy().copy()
        return out


# ---------- 롤아웃 (작업자 프로세스) ----------
_CACHE = {}


def _policy(path, version):
    if path is None:
        return None
    if _CACHE.get('v') != version:
        _CACHE['p'], _CACHE['v'] = Policy.load(path), version
    return _CACHE['p']


def rollout(args):
    path, version, job, seed, temperature, top_k, record, cfg_kw = args
    core = Core(BufferConfig(**cfg_kw))
    core.reset(job)
    pol = _policy(path, version)
    rng = np.random.default_rng(seed)
    S, C, CH, LP = [], [], [], []
    total = 0.0
    while not core.done:
        st, feat = core.state_vector(), core.cand['feat']
        if pol is None:
            k, lp, keep = greedy_index(core.cand), 0.0, np.arange(len(feat))
        else:
            s = pol.scores(st, feat)
            keep = np.arange(len(s)) if len(s) <= top_k else np.argpartition(-s, top_k)[:top_k]
            if temperature > 0:
                z = s / temperature
                z = z - z.max()
                p = np.exp(z)
                p /= p.sum()
                k = int(rng.choice(len(p), p=p))
            else:
                k = int(np.argmax(s))
            if k not in keep:
                keep = np.append(keep, k)
            if temperature > 0:
                zk = s[keep] / temperature
                zk = zk - zk.max()
                lp = float(zk[np.flatnonzero(keep == k)[0]] - np.log(np.exp(zk).sum()))
            else:
                lp = 0.0
        if record:
            S.append(st)
            C.append(feat[keep])
            CH.append(int(np.flatnonzero(keep == k)[0]))
            LP.append(lp)
        total += core.step(k)
    m = core.metrics()
    m['return'] = total
    data = (np.stack(S), C, np.array(CH), np.array(LP, np.float32)) if record and S else None
    return m, data


# ---------- 학습 ----------
def pad(cands, device):
    K = max(len(c) for c in cands)
    out = np.zeros((len(cands), K, cands[0].shape[1]), np.float32)
    mask = np.zeros((len(cands), K), bool)
    for n, c in enumerate(cands):
        out[n, :len(c)] = c
        mask[n, :len(c)] = True
    return torch.from_numpy(out).to(device), torch.from_numpy(mask).to(device)


def flatten(datas):
    S = np.concatenate([d[0] for d in datas])
    C = [c for d in datas for c in d[1]]
    CH = np.concatenate([d[2] for d in datas])
    LP = np.concatenate([d[3] for d in datas])
    return S, C, CH, LP


def bc_train(net, opt, datas, epochs, mb, device, log):
    S, C, CH, _ = flatten(datas)
    N = len(S)
    for ep in range(epochs):
        perm = np.random.permutation(N)
        tot, acc = 0.0, 0.0
        for a in range(0, N, mb):
            idx = perm[a:a + mb]
            s = torch.from_numpy(S[idx]).to(device)
            c, mask = pad([C[i] for i in idx], device)
            y = torch.from_numpy(CH[idx]).to(device)
            logits = net(s, c).masked_fill(~mask, -1e9)
            loss = F.cross_entropy(logits, y)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            tot += float(loss) * len(idx)
            acc += float((logits.argmax(1) == y).float().sum())
        log(dict(phase='bc', epoch=ep, loss=tot / N, acc=acc / N))


def ppo_update(net, opt, datas, advs, a, device):
    S, C, CH, LP = flatten(datas)
    A = np.concatenate([np.full(len(d[2]), adv, np.float32) for d, adv in zip(datas, advs)])
    N = len(S)
    stats = dict(loss=0.0, clip=0.0, ent=0.0, kl=0.0, n=0)
    for _ in range(a.epochs):
        perm = np.random.permutation(N)
        for b in range(0, N, a.minibatch):
            idx = perm[b:b + a.minibatch]
            s = torch.from_numpy(S[idx]).to(device)
            c, mask = pad([C[i] for i in idx], device)
            y = torch.from_numpy(CH[idx]).to(device)
            old = torch.from_numpy(LP[idx]).to(device)
            adv = torch.from_numpy(A[idx]).to(device)
            logits = (net(s, c) / a.temperature).masked_fill(~mask, -1e9)
            logp_all = torch.log_softmax(logits, 1)
            logp = logp_all.gather(1, y[:, None])[:, 0]
            ratio = torch.exp(logp - old)
            surr = torch.min(ratio * adv, torch.clamp(ratio, 1 - a.clip, 1 + a.clip) * adv)
            p = logp_all.exp()
            ent = -(p * logp_all.masked_fill(~mask, 0)).sum(1)
            loss = -(surr.mean() + a.entropy * ent.mean())
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            stats['loss'] += float(loss)
            stats['clip'] += float(((ratio - 1).abs() > a.clip).float().mean())
            stats['ent'] += float(ent.mean())
            stats['kl'] += float((old - logp).mean())
            stats['n'] += 1
    n = max(1, stats.pop('n'))
    return {k: v / n for k, v in stats.items()}


def evaluate(pool, path, version, jobs, cfg_kw):
    res = pool.map(rollout, [(path, version, j, 0, 0.0, 64, False, cfg_kw) for j in jobs], chunksize=1)
    ms = [r[0] for r in res]
    return dict(util=float(np.mean([m['util'] for m in ms])), violations=float(np.mean([m['violations'] for m in ms])),
                stage_moves=float(np.mean([m['stage_moves'] for m in ms])), placed=float(np.mean([m['placed'] for m in ms])),
                ret=float(np.mean([m['return'] for m in ms])))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--init', type=Path, help='시작 가중치 (.npz); 없으면 BC 부터')
    ap.add_argument('--bc-episodes', type=int, default=512)
    ap.add_argument('--bc-epochs', type=int, default=6)
    ap.add_argument('--iters', type=int, default=150)
    ap.add_argument('--jobs-per-iter', type=int, default=24)
    ap.add_argument('--group', type=int, default=8)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--bc-lr', type=float, default=5e-4)
    ap.add_argument('--epochs', type=int, default=2)
    ap.add_argument('--minibatch', type=int, default=1024)
    ap.add_argument('--clip', type=float, default=0.2)
    ap.add_argument('--entropy', type=float, default=0.001)
    ap.add_argument('--temperature', type=float, default=1.0)
    ap.add_argument('--top-k', type=int, default=64)
    ap.add_argument('--workers', type=int, default=96)
    ap.add_argument('--eval-every', type=int, default=5)
    ap.add_argument('--dev', type=int, default=32, help='dev 작업 수 (묶음마다)')
    ap.add_argument('--train-kind', default='mixed', help='학습 작업 분포 (buffer_core.make_job kind)')
    ap.add_argument('--dev-kinds', nargs='+', default=['full', 'full_near'])
    ap.add_argument('--w-block', type=float, default=None, help='하차 순서 위반 1건 벌점 (기본 BufferConfig 값)')
    ap.add_argument('--zone', action='store_true', help='버퍼 구역(830~1200)도 적재 공간으로 (BufferConfig.zone)')
    ap.add_argument('--zone-late', action='store_true', help='830 안에 놓을 자리가 없을 때 버퍼 구역을 연다')
    ap.add_argument('--slot-clear', action='store_true', help='버퍼 칸 상자와 부딪히는 자리를 뺀다')
    ap.add_argument('--reach', help='앞면 밀어 넣기 도달 표 (.npz)')
    ap.add_argument('--arm', help='팔 공간 표 (.npz, rig/arm_table.py): 실린 상자가 팔꿈치 등에 닿는 자리를 뺀다')
    ap.add_argument('--n-stage', type=int, default=3, help='재배치 버퍼 칸 수')
    ap.add_argument('--stage-order', help='버퍼로 옮길 때 쓰는 칸과 순서, 예: 1 (문 쪽 칸만) 또는 1,3,2. 기본 1,2,3')
    ap.add_argument('--keep-stage', action='store_true', help='zone: 재배치 칸도 늘 비워 둔다 (버퍼 = 투입 칸 + 재배치 칸, 나머지는 적재 공간)')
    ap.add_argument('--retrieve', action='store_true', help='실은 상자를 다시 꺼내는 이동도 쓴다 (BufferConfig.retrieve)')
    ap.add_argument('--box-types', help='상자 종류 JSON, 예: \'{"3": [160, 80, 80], "6": [160, 160, 120]}\' (L, W, H mm). 기본 pack_env 5종')
    ap.add_argument('--shares', help='종류별 도착 비율 JSON, 예: \'{"1": 1, "5": 3}\'. 없으면 균등')
    ap.add_argument('--grid', type=int, default=5)
    ap.add_argument('--support', type=float, default=0.8, help='받침 비율 하한')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    (a.output / 'args.json').write_text(json.dumps({k: str(v) for k, v in vars(a).items()}, indent=1))
    logf = open(a.output / 'log.jsonl', 'a')

    def log(d):
        d['t'] = round(time.time() - t0, 1)
        logf.write(json.dumps(d) + '\n')
        logf.flush()
        print(json.dumps(d), flush=True)

    t0 = time.time()
    cfg_kw = dict(grid=a.grid, support_min=a.support)
    if a.w_block is not None:
        cfg_kw['w_block'] = a.w_block
    cfg_kw.update(zone=a.zone or a.zone_late, zone_late=a.zone_late, slot_clear=a.slot_clear or a.zone or a.zone_late, reach=a.reach,
                  n_stage=a.n_stage)
    if a.retrieve:
        cfg_kw['retrieve'] = True
    if a.stage_order:
        cfg_kw['stage_order'] = tuple(int(x) for x in a.stage_order.split(','))
    if a.keep_stage:
        cfg_kw['keep_stage'] = True
    if a.arm:
        cfg_kw['arm'] = a.arm
    if a.box_types:
        cfg_kw['box_types'] = {int(k): tuple(v) for k, v in json.loads(a.box_types).items()}
    if a.shares:
        cfg_kw['shares'] = {int(k): float(v) for k, v in json.loads(a.shares).items()}
    cfg = BufferConfig(**cfg_kw)
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    wpath = str(a.output / 'current.npz')
    dev_rng = np.random.default_rng(123456)
    dev = [make_job(dev_rng, cfg, kind) for kind in a.dev_kinds for _ in range(a.dev)]
    ctx = get_context('fork')
    with ctx.Pool(a.workers) as pool:          # 작업자를 먼저 만든다 (CUDA 초기화 전에 fork)
        core0 = Core(cfg)
        net = Net(core0.state_dim, core0.cand_dim).to(a.device)
        version = 0
        g = evaluate(pool, None, 0, dev, cfg_kw)
        log(dict(phase='dev_greedy_baseline', **g))
        if a.init:
            w = dict(np.load(a.init))
            sd = {}
            for n in ('s0', 's1', 'c0', 'c1', 'h0', 'h1'):
                sd[f'{n}.weight'] = torch.from_numpy(w[f'{n}.w'].T.copy())
                sd[f'{n}.bias'] = torch.from_numpy(w[f'{n}.b'].copy())
            for n in ('s0', 'c0'):                  # 입력 특징이 늘었으면 (zone·retrieve) 새 열은 0 으로 시작
                want = getattr(net, n).weight.shape
                have = sd[f'{n}.weight'].shape
                if have != want:
                    padw = torch.zeros(want)
                    padw[:, :have[1]] = sd[f'{n}.weight']
                    sd[f'{n}.weight'] = padw
            net.load_state_dict(sd)
        else:
            jobs = [make_job(rng, cfg, a.train_kind) for _ in range(a.bc_episodes)]
            res = pool.map(rollout, [(None, 0, j, 0, 0.0, 10 ** 6, True, cfg_kw) for j in jobs], chunksize=4)
            datas = [r[1] for r in res if r[1] is not None]
            log(dict(phase='bc_data', episodes=len(datas), steps=int(sum(len(d[2]) for d in datas))))
            opt = torch.optim.Adam(net.parameters(), lr=a.bc_lr)
            bc_train(net, opt, datas, a.bc_epochs, a.minibatch, a.device, log)
        np.savez(wpath, **net.export())
        version += 1
        best = evaluate(pool, wpath, version, dev, cfg_kw)
        log(dict(phase='dev', it=0, **best))
        np.savez(a.output / 'policy.npz', **net.export())
        opt = torch.optim.Adam(net.parameters(), lr=a.lr)
        for it in range(1, a.iters + 1):
            ti = time.time()
            jobs = [make_job(rng, cfg, a.train_kind) for _ in range(a.jobs_per_iter)]
            tasks = [(wpath, version, j, int(rng.integers(1 << 30)), a.temperature, a.top_k, True, cfg_kw)
                     for j in jobs for _ in range(a.group)]
            res = pool.map(rollout, tasks, chunksize=1)
            rets = np.array([r[0]['return'] for r in res]).reshape(a.jobs_per_iter, a.group)
            mu, sd = rets.mean(1, keepdims=True), rets.std(1, keepdims=True)
            advs = ((rets - mu) / (sd + 1e-3)).ravel()
            datas, keep_adv = [], []
            for r, adv in zip(res, advs):
                if r[1] is not None:
                    datas.append(r[1])
                    keep_adv.append(adv)
            tr = time.time() - ti
            st = ppo_update(net, opt, datas, keep_adv, a, a.device)
            np.savez(wpath, **net.export())
            version += 1
            ms = [r[0] for r in res]
            log(dict(phase='train', it=it, util=float(np.mean([m['util'] for m in ms])),
                     violations=float(np.mean([m['violations'] for m in ms])),
                     stage=float(np.mean([m['stage_moves'] for m in ms])), ret=float(rets.mean()),
                     steps=int(sum(len(d[2]) for d in datas)), rollout_s=round(tr, 1), update_s=round(time.time() - ti - tr, 1), **st))
            if it % a.eval_every == 0 or it == a.iters:
                d = evaluate(pool, wpath, version, dev, cfg_kw)
                log(dict(phase='dev', it=it, **d))
                if d['ret'] > best['ret']:
                    best = d
                    np.savez(a.output / 'policy.npz', **net.export())
                    log(dict(phase='best', it=it, **d))
        np.savez(a.output / 'last.npz', **net.export())


if __name__ == '__main__':
    main()
