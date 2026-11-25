"""정해진 네 단계로 상자를 옮기는 경로 (../../pipeline/details_by_modules/mp.md).

  1. 집기        투입 위치로 가서 윗면을 흡착                      <- 상자 종류마다 저장해 둔 경로
  2. 이송 자세   들어올리기 -> 세우기 -> 돌리기 -> 팔을 놓는 자세로   <- (상자, 방향, y, z) 마다 저장해 둔 경로
  3. 레일 이동   팔은 고정, 레일만                                  <- 놓을 자리의 x 로 계산
  4. 놓기        내리기 -> 밀어 넣기 -> 놓기 -> 온 길로 복귀          <- 2 와 같은 표, 레일 값만 x 만큼 옮김

경로를 실행 중에 탐색하지 않는다. 빈 장면에서 미리 계산해 둔 경로(틀)를 꺼내 레일 값만 옮기고,
그 경로가 지금 쌓인 상자와 부딪히지 않는지만 검사한다.
역기구학과 충돌 검사는 같은 폴더의 stacking.py 의 Stacker 를 쓴다.
"""
from __future__ import annotations

import hashlib
import pickle
from pathlib import Path

import mujoco
import numpy as np

import stacking
import rules

HERE = Path(__file__).resolve().parent

for _k, (_dims, _color) in rules.EXTRA_BOXES.items():       # Stacker 를 만들기 전에 상자 종류를 더한다
    stacking.BOXES[_k], stacking.COLORS[_k] = _dims, _color

from stacking import BOXES, CLEAR, EX, EY, EZ, FLOOR, GAP, PICK_XY, Plan, Stacker  # noqa: E402

CACHE = HERE / "cache" / "templates.pkl"
STAGES = ["틀 없음 (빈 장면에서도 이 자리·방향으로는 경로를 못 만듦)", "레일 범위를 벗어남", "집기·세우기 구간이 쌓인 상자에 막힘",
          "레일 이동 구간이 막힘", "내리기·밀어 넣기 구간이 막힘", "놓은 뒤 빠져나오는 길이 막힘", "투입 위치로 가는 길이 막힘"]


def _fingerprint():
    h = hashlib.md5()
    for p in (HERE / "rules.py", Path(__file__), Path(stacking.__file__), stacking.ROOT / "model" / "manibox_modified.xml"):
        h.update(p.read_bytes())
    return h.hexdigest()


class Sequencer:
    def __init__(self, types):
        self.s = Stacker(types)            # 실제 장면 (쌓인 상자가 장애물)
        self.e = Stacker([])               # 빈 장면: 틀을 계산하는 데만 쓴다
        self.tpl = {}
        if CACHE.exists():
            saved = pickle.loads(CACHE.read_bytes())
            if saved.get("fingerprint") == _fingerprint():
                self.tpl = saved["tpl"]
        self.new_templates = 0

    def save(self):
        if self.new_templates:
            CACHE.parent.mkdir(exist_ok=True)
            CACHE.write_bytes(pickle.dumps(dict(fingerprint=_fingerprint(), tpl=self.tpl)))

    # ---------- 틀: 빈 장면에서 한 번 계산해 저장 ----------
    def _pick_tpl(self, k, g, usign):
        """1 + 2 앞부분: 집기 -> 띄우기 -> 들어올리기 -> 세우기. 상자 종류와 잡는 방향에만 달렸다."""
        key = ("pick", k, g, usign)
        if key not in self.tpl:
            e, held = self.e, self.e.held_ids[(k, g)]
            e.m.geom_pos[held, :2] = 0
            pos = np.array([*PICK_XY, FLOOR + BOXES[k][2] + GAP])
            self.tpl[key] = e.depart(pos, usign * (EY if g == 0 else EX), held, tilt_up=True)
            self.new_templates += 1
        return self.tpl[key]

    def _tpl(self, k, la, y, z, cand, x_ref):
        """집기부터 놓기까지의 틀. (상자 종류, 긴 변 방향, 놓을 y, z) 와 후보에 달렸고 x 와는 무관하다."""
        key = ("full", k, la, round(y, 4), round(z, 4), cand["camera"], x_ref)
        if key in self.tpl:
            return self.tpl[key]
        e, H = self.e, BOXES[k][2]
        v0 = -{"up": EZ, "down": -EZ, "+y": EY, "-y": -EY}[cand["camera"]]   # 놓는 순간의 공구 x축. 카메라 받침은 공구 -x 쪽에 있다
        g = 0 if abs(v0 @ (EY if la == "y" else EZ)) > 0.5 else 1   # 공구 x축이 상자 가로(L)와 나란하면 0, 세로(W)와 나란하면 1
        held = e.held_ids[(k, g)]
        e.m.geom_pos[held, :2] = 0
        out = None
        place = np.array([x_ref - H / 2 - GAP, y, z])
        dq = e.depart(place, v0, held, tilt_up=False)     # 4 를 거꾸로: 놓인 자리 -> 띄우기 -> 물러나기 -> 올리기
        if dq is not None:
            wb, pb, xb = dq
            for usign in (1, -1):                         # 집을 때 공구 x축의 부호. 세운 뒤 돌려서 맞춘다
                dp = self._pick_tpl(k, g, usign)
                if dp is None:
                    continue
                wa, pa, xa = dp
                wr = e._roll(wa[-1], pa, xa, xb, held)    # 2: 돌리기
                if wr is None:
                    continue
                qm = wb[-1].copy()                        # 2 끝: 팔은 놓는 쪽 자세, 레일은 아직 투입 구역
                qm[0] = np.clip(qm[0] - (pb[0] - pa[0]), e.lo[0], e.hi[0])
                if e.path(wr[-1], qm, held):
                    out = dict(wa=wa, wr=wr, qm=qm, wb=wb, g=g, quat=self._quat(la))
                    break
        self.tpl[key] = out
        self.new_templates += 1
        return out

    @staticmethod
    def _quat(la):
        la = EY if la == "y" else EZ
        q = np.zeros(4)
        mujoco.mju_mat2Quat(q, np.column_stack([la, np.cross(EX, la), EX]).flatten())
        return q

    # ---------- 실행 전 검사: 틀의 레일 값만 옮겨 지금 장면에서 확인 ----------
    def _check(self, i, k, c, t, dx):
        """반환: (통과한 단계 수, Plan 또는 None)."""
        s = self.s
        held = s.held_ids[(k, t["g"])]
        s.m.geom_pos[held, :2] = 0
        wb = [q + np.array([dx, 0, 0, 0, 0, 0, 0]) for q in t["wb"]]
        if min(q[0] for q in wb) < s.lo[0] - 1e-9 or max(q[0] for q in wb) > s.hi[0] + 1e-9:
            return 1, None
        wa, wr, qm = t["wa"], t["wr"], t["qm"]

        def run(wps, contact, h):
            """경유 자세를 차례로 검사. contact 에 든 번호의 구간만 상자와 맞닿음을 허용한다."""
            ok = True
            for n, (a, b) in enumerate(zip(wps[:-1], wps[1:])):
                s.clear = None if n in contact else CLEAR
                ok = ok and s.path(a, b, h)
            s.clear = CLEAR
            return ok

        s._hide(i)
        if not run(wa + wr[1:] + [qm], {0}, held):                  # 1, 2
            return 2, None
        if not run([qm, wb[-1]], set(), held):                      # 3
            return 3, None
        down = wb[::-1]
        if not run(down, {len(down) - 2}, held):                    # 4
            return 4, None
        s.set_box(i, c, t["quat"])                                  # 놓인 상자를 장애물로 두고 빈손 복귀
        back = wb + [qm, wr[-1]]
        ok = run(back, {0, 1}, -1)
        s._hide(i)
        if not ok:
            return 5, None
        appr = s._link(s.q, wa[-1])
        if appr is None:
            return 6, None
        carry = wa + wr[1:] + [qm] + down
        return 7, Plan(True, i, "", "side", np.array(c, float), t["quat"], held, appr + wa[::-1][1:], carry, back)

    def plan(self, i, center, size_xyz):
        """상자 i 를 center(월드), size_xyz 로 놓는다. 흡착면이 문 쪽(-x)을 보는 옆면 방식만 다룬다.
        반환: Plan. 성공하면 plan.candidate 에 쓰인 후보 번호가 있다. commit 은 부르는 쪽에서 한다."""
        k = self.s.types[i]
        L, W, H = BOXES[k]
        c, sz = np.asarray(center, float), np.asarray(size_xyz, float)
        if abs(sz[0] - H) > 1e-6 or not np.allclose(sorted(sz[1:]), sorted((L, W)), atol=1e-6):
            return Plan(False, i, "옆면 방식(상자 높이가 x 방향)으로 놓을 수 없는 방향")
        la = "y" if abs(sz[1] - L) < 1e-6 else "z"
        if rules.KEEPOUT_X is not None and c[0] - H / 2 < rules.KEEPOUT_X - 1e-6:
            return Plan(False, i, "투입 구역 (싣지 않는 자리)")
        best = 0
        box = dict(type=k, height=H, la=la, x=c[0], y=c[1], z=c[2])
        for n, cand in enumerate(rules.place_candidates(box)):
            x_ref = cand["x_ref"] if cand["x_ref"] is not None else round(c[0], 2)
            t = self._tpl(k, la, c[1], c[2], cand, x_ref)
            if t is None:
                continue
            stage, p = self._check(i, k, c, t, c[0] - x_ref)
            if p is not None:
                p.candidate = n
                return p
            best = max(best, stage)
        return Plan(False, i, STAGES[best])
