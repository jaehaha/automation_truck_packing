"""학습한 배치 정책 (numpy 만으로 실행). 가중치는 policy.npz.

정책은 지금 가능한 이동 후보마다 점수를 매기고 가장 높은 것을 고른다.
  상태    높이 지도(풀링) + 투입 칸·버퍼 3칸의 상자 + 다음 5개 상자 + 진행률   (Core.state_vector)
  후보    위치·크기·종류·지점·어느 칸에서 오는지·받침·빈 공간·접촉·하차 순서 위반 수 등 (CAND_FEATURES)
  점수    s = head([h_c, h_s, h_c * h_s]),  h_s = MLP(상태), h_c = MLP(후보)
학습 코드(train_buffer.py)의 torch 모델과 같은 계산이다.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from buffer_core import Core, greedy_index

LAYERS = ('s0', 's1', 'c0', 'c1', 'h0', 'h1')


def gelu(x):
    return 0.5 * x * (1.0 + np.tanh(0.7978845608028654 * (x + 0.044715 * x ** 3)))


class Policy:
    def __init__(self, weights: dict):
        self.w = {k: np.asarray(v, np.float32) for k, v in weights.items()}

    @classmethod
    def load(cls, path):
        with np.load(Path(path)) as z:
            return cls(dict(z))

    def scores(self, state: np.ndarray, feat: np.ndarray) -> np.ndarray:
        w = self.w
        hs = gelu(gelu(state @ w['s0.w'] + w['s0.b']) @ w['s1.w'] + w['s1.b'])
        hc = gelu(gelu(feat @ w['c0.w'] + w['c0.b']) @ w['c1.w'] + w['c1.b'])
        z = np.concatenate([hc, np.broadcast_to(hs, hc.shape), hc * hs], 1)
        return (gelu(z @ w['h0.w'] + w['h0.b']) @ w['h1.w'] + w['h1.b'])[:, 0]

    def choose(self, core: Core, rng=None, temperature: float = 0.0):
        """후보 번호와 (점수, 로그확률). temperature 0 이면 최고 점수."""
        s = self.scores(core.state_vector(), core.cand['feat'])
        if temperature <= 0 or rng is None:
            return int(np.argmax(s)), s, None
        z = s / temperature
        z = z - z.max()
        p = np.exp(z)
        p /= p.sum()
        k = int(rng.choice(len(p), p=p))
        return k, s, float(np.log(p[k] + 1e-12))


def run_episode(core: Core, job, policy: Policy | None = None, rng=None, temperature: float = 0.0):
    """정책(없으면 greedy)으로 한 작업을 끝까지 진행하고 지표를 돌려준다."""
    core.reset(job)
    total = 0.0
    while not core.done:
        k = greedy_index(core.cand) if policy is None else policy.choose(core, rng, temperature)[0]
        total += core.step(k)
    m = core.metrics()
    m['return'] = total
    return m
