"""ManiBox 버퍼 적재 코어 (numpy 전용, MuJoCo 없음). 학습·실행·검증이 모두 이 코어를 쓴다.

공간 (컨테이너 좌표, mm. pack_env 와 같음: 원점 = 안쪽 벽 바닥 모서리, x = 출구 방향)
  컨테이너  x 0 ~ 830, y 0 ~ 480, z 0 ~ 568
  버퍼      x 830 ~ 1200 에 2 x 2 칸 (칸 안쪽 170 = 160 상자 + 사방 5 mm, 칸막이 10 mm)
            문 쪽 줄의 한 칸 = 투입 칸 (상자가 들어오는 곳), 나머지 3칸 = 재배치 버퍼

로봇 한 번의 이동 = 상자 하나를 시작 칸에서 끝 자리로 옮긴다.
  place  투입 칸이나 버퍼 칸의 상자를 컨테이너에 넣는다 (윗면 흡착 -> 흡착면이 출구를 보게 돌려 앞에서 밀어 넣음)
  stage  투입 칸의 상자를 빈 버퍼 칸에 내려놓는다 (윗면 흡착 그대로)
투입 칸이 비면 다음 상자가 들어온다.

자리 규칙은 pack_env._feasible 과 같다. fast_feasible 은 같은 결과를 분리 가능한 최대값과 적분 영상으로
빠르게 계산한다 (tests/test_buffer.py 가 무작위 상태에서 두 결과가 같은지 확인한다).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from pack_env import Config, TOL

SLOT_NAMES = ('arrival', 'buffer1', 'buffer2', 'buffer3')
PLACE, STAGE, RETRIEVE = 0, 1, 2          # RETRIEVE: 컨테이너의 상자를 다시 꺼내 버퍼 칸에 둔다 (cfg.retrieve)
PRE_INSERT_MM = 40          # 밀어 넣기 전 대기점: 끝 자리에서 출구 쪽으로 (mm)


@dataclass
class BufferConfig(Config):
    depth: int = 830                    # 적재 깊이 (전체 1200 - 버퍼 370)
    height: int = 568
    grid: int = 5
    support_min: float = 0.9            # pack_env 기본 0.8 보다 높게: 실물 오차 시험에서 넘어지는 상자가 0 이 됐다
    max_boxes: int = 400
    n_stage: int = 3                    # 재배치 버퍼 칸 수
    cell: int = 170                     # 버퍼 칸 안쪽 (mm)
    wall: int = 10                      # 버퍼 칸막이 (mm)
    w_block: float = 0.5                # 하차 순서 위반 1건 벌점
    w_stage: float = 0.02               # 버퍼로 옮기는 이동 1회 벌점 (로봇 시간)
    safe_order: bool = True             # 위반 없는 자리가 하나라도 있으면 위반 자리는 후보에서 뺀다
    max_cands: int = 256                # 한 결정에서 정책이 보는 후보 수 상한
    zone: bool = False                  # True: 버퍼 구역(830~1200)도 적재 공간. 상자가 있는 칸과 (상자가 더 올 동안) 투입 칸만 비워 둔다
    zone_late: bool = False             # zone 이고 True: 830 안에 놓을 자리가 없을 때 (버퍼도 다 찼을 때) 비로소 버퍼 구역을 연다
    slot_clear: bool = False            # True: 넣는 상자나 팔이 버퍼 칸의 상자와 부딪히는 자리는 뺀다
    arm_pass: int = 200                 # 팔(툴·손목)이 차지하는 길이: 대기점의 상자 앞면에서 출구 쪽으로 (mm)
    arm_radius: int = 45                # 팔 반경 (mm). 툴 축 = 상자 앞면 중심
    reach: str | None = None            # 앞면 밀어 넣기 도달 표 (.npz, rig/reach_table.py). 주면 팔이 닿지 않는 자리는 뺀다
    arm: str | None = None              # 팔 공간 표 (.npz, rig/arm_table.py). 주면 실린 상자가 팔 (베이스 ~ link4, 팔꿈치) 에 닿는 자리는 뺀다
    stage_order: tuple = ()             # 버퍼로 옮길 때 칸을 고르는 순서 (예: (1, 3, 2)). 비우면 1, 2, 3. (1,) = 문 쪽 buffer1 만
    keep_stage: bool = False            # zone: 재배치 칸(stage_order)도 상자가 더 오는 동안 늘 비워 둔다 (그 위에 싣지 않는다).
                                        #   버퍼 = 투입 칸 + 재배치 칸뿐이고 나머지 칸 자리는 처음부터 적재 공간
    retrieve: bool = False              # True: 앞·위가 빈 상자를 다시 꺼내 (넣기의 역순) 빈 버퍼 칸에 둘 수 있다
    max_retrieve_per_box: int = 1       # 상자 하나를 다시 꺼낼 수 있는 횟수
    w_retrieve: float = 0.2             # 다시 꺼내기 1회 벌점 (로봇 이동 2번: 꺼내기 + 다시 넣기). 위반 1건 (w_block 0.5) 보다 작게


def zone_depth(cfg) -> int:
    """버퍼 구역 깊이 (mm): 칸 2줄 + 칸막이 3개 = 370."""
    return 2 * cfg.cell + 3 * cfg.wall


def space_volume(cfg) -> float:
    """상자를 실을 수 있는 공간 부피 (mm^3). zone 이면 버퍼 구역까지 (1200)."""
    return (cfg.depth + (zone_depth(cfg) if cfg.zone else 0)) * cfg.width * cfg.height


_REACH = {}


def load_reach(path):
    """앞면 밀어 넣기 도달 표 (rig/reach_table.py 가 만든 .npz). ok[x, y, z]: 상자 앞면 중심이 (x, y, z) mm 일 때 닿는지."""
    if path not in _REACH:
        with np.load(path) as z:
            _REACH[path] = {k: z[k] for k in ('ok', 'x0', 'dx', 'y0', 'dy', 'z0', 'dz')}
    return _REACH[path]


_ARM = {}


def load_arm(path):
    """팔 공간 표 (rig/arm_table.py 가 만든 .npz). z[x, y, z, 해, 칸x, 칸y] = 그 칸에서 팔의 가장 낮은 높이 (unit mm 단위, 255 = 팔 없음)."""
    if path not in _ARM:
        with np.load(path) as z:
            _ARM[path] = {k: z[k] for k in z.files}
    return _ARM[path]


def reach_ok(tab, x, y, z):
    """도달 표 조회 (배열 브로드캐스트). x·y 는 둘레 격자점 4개가 모두 닿아야 참 (보수적), z 는 가까운 격자점."""
    T = tab['ok']
    fx = (np.asarray(x, float) - tab['x0']) / tab['dx']
    fy = (np.asarray(y, float) - tab['y0']) / tab['dy']
    iz = np.rint((np.asarray(z, float) - tab['z0']) / tab['dz']).astype(int)
    inside = (fx >= -1e-6) & (fx <= T.shape[0] - 1 + 1e-6)
    zin = (iz >= 0) & (iz < T.shape[2])
    fy = np.clip(fy, 0, T.shape[1] - 1)                 # y 는 표 끝 값으로 (벽 쪽 끝 자리)
    x0 = np.clip(np.floor(fx + 1e-6).astype(int), 0, T.shape[0] - 1)
    x1 = np.clip(np.ceil(fx - 1e-6).astype(int), 0, T.shape[0] - 1)
    y0 = np.clip(np.floor(fy + 1e-6).astype(int), 0, T.shape[1] - 1)
    y1 = np.clip(np.ceil(fy - 1e-6).astype(int), 0, T.shape[1] - 1)
    zc = np.clip(iz, 0, T.shape[2] - 1)
    ok = T[x0, y0, zc] & T[x1, y0, zc] & T[x0, y1, zc] & T[x1, y1, zc]
    return ok & inside & zin


def buffer_cells(cfg) -> np.ndarray:
    """버퍼 칸 중심 (x, y) mm, 순서 = SLOT_NAMES. 투입 칸과 buffer1 은 문 쪽 줄, buffer2/3 은 컨테이너 쪽 줄."""
    x_in = cfg.depth + cfg.wall + cfg.cell / 2
    x_door = cfg.depth + 2 * cfg.wall + 1.5 * cfg.cell
    y0 = (cfg.width - (2 * cfg.cell + 3 * cfg.wall)) / 2
    ya = y0 + cfg.wall + cfg.cell / 2
    yb = ya + cfg.cell + cfg.wall
    return np.array([(x_door, ya), (x_door, yb), (x_in, ya), (x_in, yb)], float)


def cells_of(dims, cfg):
    g, gap = cfg.grid, cfg.gap
    return math.ceil((dims[0] + gap) / g), math.ceil((dims[1] + gap) / g)


def _integral(a: np.ndarray) -> np.ndarray:
    s = np.zeros((a.shape[0] + 1, a.shape[1] + 1))
    s[1:, 1:] = a.cumsum(0).cumsum(1)
    return s


def _rect(S, i0, j0, i1, j1):
    """sum of a[i0:i1, j0:j1] from the integral image S (all arguments may be arrays)."""
    return S[i1, j1] - S[i0, j1] - S[i1, j0] + S[i0, j0]


def fast_feasible(hm: np.ndarray, dims, cfg):
    """pack_env._feasible 과 같은 (ok, base) 에 받침 비율 sup 과 칸 수 (nx, ny) 를 더해 돌려준다."""
    NX, NY = hm.shape
    nx, ny = cells_of(dims, cfg)
    ok = np.zeros((NX, NY), bool)
    base = np.zeros((NX, NY))
    sup = np.zeros((NX, NY))
    if nx > NX or ny > NY:
        return ok, base, sup, nx, ny
    colmax = sliding_window_view(hm, ny, axis=1).max(-1)                 # (NX, J): y 창 최대
    b = sliding_window_view(colmax, nx, axis=0).max(-1)                  # (I, J): 창 최대 = 바닥 높이
    I, J = b.shape
    s = np.empty((I, J))
    for v in np.unique(b):                                               # 높이는 80 의 배수라 몇 단계뿐
        S = _integral((hm >= v - TOL).astype(np.float64))
        cnt = S[nx:nx + I, ny:ny + J] - S[:I, ny:ny + J] - S[nx:nx + I, :J] + S[:I, :J]
        sel = b == v
        s[sel] = cnt[sel] / (nx * ny)
    suf = np.maximum.accumulate(colmax[::-1], axis=0)[::-1]
    front = np.full((I, J), -np.inf)
    idx = np.arange(I) + nx
    front[idx < NX] = suf[idx[idx < NX]]
    good = (b + dims[2] <= cfg.height + TOL) & (s >= cfg.support_min - 1e-9) & (front <= b + TOL)
    ok[:I, :J], base[:I, :J], sup[:I, :J] = good, b, s
    return ok, base, sup, nx, ny


def corner_positions(ok: np.ndarray, base: np.ndarray):
    """가능한 칸 중 같은 높이로 더 안쪽(-x)으로 갈 수 없고, 같은 높이로 왼쪽이나 오른쪽 끝인 칸."""
    same_up = np.zeros_like(ok)
    same_up[1:] = ok[:-1] & (base[:-1] == base[1:])
    same_l = np.zeros_like(ok)
    same_l[:, 1:] = ok[:, :-1] & (base[:, :-1] == base[:, 1:])
    same_r = np.zeros_like(ok)
    same_r[:, :-1] = ok[:, 1:] & (base[:, 1:] == base[:, :-1])
    return np.nonzero(ok & ~same_up & (~same_l | ~same_r))


# ---------- 후보 특징 ----------
CAND_FEATURES = (
    'x0', 'y0', 'z0', 'x1', 'y1', 'z1', 'dx', 'dy', 'dz',
    'type1', 'type2', 'type3', 'type4', 'type5', 'stop', 'stop_rel',
    'src_arrival', 'src_b1', 'src_b2', 'src_b3', 'stage',
    'support', 'waste_under', 'dead_behind', 'contact_back', 'contact_left', 'contact_right', 'flush_top',
    'blocks', 'front_free', 'empty_slots', 'volume')
CAND_DIM = len(CAND_FEATURES)
ZONE_FEATURES = ('in_zone', 'covers_cell')         # zone 일 때만 더한다 (버퍼 구역에 걸친 깊이 비율, 덮는 빈 칸 수)
RETRIEVE_FEATURES = ('retrieve', 'unblocks')       # retrieve 일 때만 더한다 (다시 꺼내기, 꺼내면 풀리는 위반 수)
POOL = 8          # 상태의 높이 지도 풀링 (칸)
LOOK = 5          # 상태에 넣는 다음 상자 수


class Core:
    """한 에피소드의 상태와 규칙. step() 은 candidates() 가 돌려준 후보 번호를 받는다."""

    def __init__(self, cfg: BufferConfig | None = None):
        self.cfg = c = cfg or BufferConfig()
        self.D = c.depth + (zone_depth(c) if c.zone else 0)            # 높이 지도 깊이 (zone 이면 1200)
        self.NX, self.NY = self.D // c.grid, c.width // c.grid
        self.NX_load = c.depth // c.grid
        self.vol = space_volume(c)
        self.vol_rig = (c.depth + zone_depth(c)) * c.width * c.height  # 1200 전체 (비교용 사용률)
        self.types = sorted(c.box_types)
        self.cells = buffer_cells(c)
        self.n_slots = 1 + c.n_stage
        self.cell_rect = [(x - c.cell / 2, x + c.cell / 2, y - c.cell / 2, y + c.cell / 2) for x, y in self.cells[:self.n_slots]]
        self.reach_tab = load_reach(c.reach) if c.reach else None
        self.arm_tab = load_arm(c.arm) if c.arm else None
        self._armh = (None, None)
        self.ret_off = CAND_DIM + (len(ZONE_FEATURES) if c.zone else 0)
        self.cand_dim = self.ret_off + (len(RETRIEVE_FEATURES) if c.retrieve else 0)
        px, py = -(-self.NX // POOL), -(-self.NY // POOL)
        self.state_dim = 2 * px * py + self.n_slots * 7 + LOOK * 7 + 8 + ((3 if c.zone_late else 2) if c.zone else 0)

    # ---------- 에피소드 ----------
    def dims(self, k, o):
        L, W, H = self.cfg.box_types[k]
        return (H, L, W) if o == 0 else (H, W, L)

    def orients(self, k):
        L, W, _ = self.cfg.box_types[k]
        return (0, 1) if L != W else (0,)

    def reset(self, job, total_hint: int | None = None):
        """job: 도착 순서의 (지점, 종류) 목록. 실제 운용에서는 아는 만큼만 주고 extend() 로 늘린다.
        total_hint: 전체 상자 수를 미리 알면 (진행률 특징에 쓴다). 없으면 len(job)."""
        self.job = [(int(s), int(k)) for s, k in job][: self.cfg.max_boxes]
        self.total_hint = total_hint
        self.n_stop = max([s for s, _ in self.job], default=1)
        self.hm = np.zeros((self.NX, self.NY))
        self.version = 0
        self.placed = []
        self.lo = np.zeros((0, 3))
        self.hi = np.zeros((0, 3))
        self.pstop = np.zeros(0, int)
        self.util = 0.0
        self.violations = 0
        self.n_stage_moves = 0
        self.moves = []
        self.next = 0
        self.slots = [None] * self.n_slots          # 0 = 투입 칸, 1.. = 재배치 버퍼. 값 = job 번호
        self.open_end = False                       # True: 상자가 더 올 수 있다 (실제 운용, OnlinePlanner 가 close() 전까지)
        self.closed = False                         # zone: 더 놓을 자리가 없어 적재를 마감했다 (더 받지 않는다)
        self.zone_open = not self.cfg.zone_late       # zone: 버퍼 구역에 실을 수 있다
        self.n_kept = 0                             # zone: 끝날 때 칸에 둔 채로 실은 상자 수
        self.skipped = []                           # skip_arrival() 로 뺀 상자 (job 번호)
        self.retrieved = {}                         # 다시 꺼낸 상자: job 번호 -> 횟수
        self.ret_bonus = {}                         # 다시 꺼낼 때 풀린 위반 수 (다시 넣지 않고 끝나면 보상을 되돌린다)
        self.n_retrieve = 0
        self._feas, self._aux = {}, {}
        self._arrive()
        self.done, self.reason = False, None
        self.cand = self.candidates()
        self._check_end()

    def clone(self) -> 'Core':
        """같은 설정의 독립된 사본 (탐색용). 캐시는 얕게 복사한다 (배열은 새로 만들기만 하고 고치지 않는다)."""
        c = Core.__new__(Core)
        c.__dict__.update(self.__dict__)
        c.hm = self.hm.copy()
        c.placed = list(self.placed)
        c.slots = list(self.slots)
        c.moves = list(self.moves)
        c.retrieved = dict(self.retrieved)
        c.ret_bonus = dict(self.ret_bonus)
        c._feas = dict(self._feas)
        c._aux = dict(self._aux)
        return c

    def extend(self, boxes):
        """도착 순서 뒤에 상자를 더한다 (실제 운용: 새로 들어왔거나 줄에 새로 보이는 상자)."""
        self.job += [(int(s), int(k)) for s, k in boxes]
        self.n_stop = max([self.n_stop] + [int(s) for s, _ in boxes])
        self._arrive()
        self.done, self.reason = False, None
        self.cand = self.candidates()
        self._check_end()

    def _arrive(self):
        if self.slots[0] is None and self.next < len(self.job) and not self.closed:
            self.slots[0] = self.next
            self.next += 1

    def more_coming(self) -> bool:
        """투입 칸에 상자가 더 들어오는지."""
        if self.closed:
            return False
        return self.next < len(self.job) or self.open_end or (self.total_hint is not None and self.next < self.total_hint)

    def cell_free(self, s) -> bool:
        """버퍼 칸 s 를 쓸 수 있는지 (zone 에서는 그 칸 위에 실은 상자가 없어야 한다)."""
        if not self.cfg.zone:
            return True
        g = self.cfg.grid
        x0, x1, y0, y1 = self.cell_rect[s]
        S = self._hm_integral()
        return _rect(S, int(x0 // g), int(y0 // g), int(math.ceil(x1 / g)), int(math.ceil(y1 / g))) <= 0

    def staging_slots(self) -> list:
        """투입 칸 상자를 옮길 수 있는 빈 버퍼 칸 (cfg.stage_order 순서)."""
        order = [s for s in self.cfg.stage_order if 0 < s < self.n_slots] or range(1, self.n_slots)
        return [s for s in order if self.slots[s] is None and self.cell_free(s)]

    def feasible(self, k, o):
        key = (k, o)
        hit = self._feas.get(key)
        if hit is None or hit[0] != self.version:
            dims = self.dims(k, o)
            ok, base, sup, nx, ny = fast_feasible(self.hm, dims, self.cfg)
            if self.reach_tab is not None and ok.any():
                g = self.cfg.grid
                face = np.arange(self.NX) * g + nx * g / 2 + dims[0] / 2
                yc = np.arange(self.NY) * g + ny * g / 2
                zc = base + dims[2] / 2
                ok = ok & reach_ok(self.reach_tab, face[:, None], yc[None, :], zc) \
                        & reach_ok(self.reach_tab, face[:, None] + PRE_INSERT_MM, yc[None, :], zc)
            hit = (self.version, (ok, base, sup, nx, ny))
            self._feas[key] = hit
        return hit[1]

    def _arm_map(self):
        """팔 규칙용 높이 지도: H[i, j] = 칸 i, j 에서 시작하는 팔 칸 (20 mm) 안의 가장 높은 상자 윗면. 창이 넘치지 않게 0 으로 둘러싼다."""
        v, H = self._armh
        if v == self.version:
            return H
        T, g = self.arm_tab, self.cfg.grid
        w = int(T['cell']) // g
        P = np.zeros((self.NX + w, self.NY + w))
        P[:self.NX, :self.NY] = self.hm
        H20 = np.max([P[a:a + self.NX, b:b + self.NY] for a in range(w) for b in range(w)], 0)
        px0, py0 = -int(T['wx0']) // g + 1, -int(T['wy0']) // g + 1
        nwx, nwy = T['z'].shape[4:]
        H = np.zeros((self.NX + px0 + nwx * w + w, self.NY + py0 + nwy * w + w))
        H[px0:px0 + self.NX, py0:py0 + self.NY] = H20
        self._armh = (self.version, H)
        return H

    def _arm_clear(self, xf, yc, zc):
        """상자 앞면 xf, 중심 yc·zc (mm, 배열) 에 넣을 때 팔 공간 표의 해 중 하나라도 실린 상자에 닿지 않는지."""
        T, g = self.arm_tab, self.cfg.grid
        nx_, ny_, nz_, K, nwx, nwy = T['z'].shape
        ix = np.clip(np.rint((xf - T['x0']) / T['dx']), 0, nx_ - 1).astype(int)
        iy = np.clip(np.rint((yc - T['y0']) / T['dy']), 0, ny_ - 1).astype(int)
        iz = np.clip(np.rint((zc - T['z0']) / T['dz']), 0, nz_ - 1).astype(int)
        w = int(T['cell']) // g
        H = self._arm_map()
        i0 = np.floor((xf + T['wx0']) / g).astype(int) + (-int(T['wx0']) // g + 1)
        j0 = np.floor((yc + T['wy0']) / g).astype(int) + (-int(T['wy0']) // g + 1)
        Hv = H[i0[:, None, None] + w * np.arange(nwx)[None, :, None], j0[:, None, None] + w * np.arange(nwy)[None, None, :]]
        Z = T['z'][ix, iy, iz]                                               # (n, K, nwx, nwy)
        dz = zc - (T['z0'] + iz * T['dz'])
        zmin = Z.astype(np.float32) * float(T['unit']) + dz[:, None, None, None].astype(np.float32)
        hit = ((Z != 255) & (Hv[:, None] > zmin)).any(axis=(2, 3))
        return ((~hit) & (np.arange(K)[None, :] < T['n'][ix, iy, iz][:, None])).any(1)

    def _arm_corners(self, k, o, ok, base, ii, jj, nx, ny):
        """팔 규칙: 실린 상자가 팔에 닿는 대표 자리를 빼고 대표 자리를 다시 고른다 (세 번까지, 그 뒤 남는 것만)."""
        g = self.cfg.grid
        dims = self.dims(k, o)
        for t in range(4):
            clear = self._arm_clear((ii + nx / 2) * g + dims[0] / 2, (jj + ny / 2) * g, base[ii, jj] + dims[2] / 2)
            if clear.all() or t == 3:
                return ii[clear], jj[clear]
            ok = ok.copy()
            ok[ii[~clear], jj[~clear]] = False
            ii, jj = corner_positions(ok, base)
            if len(ii) == 0:
                return ii, jj
        return ii, jj

    def _reserved(self, src) -> list:
        """zone: src 칸의 상자를 넣는 이 이동에서 비워 둬야 하는 칸 (mm 사각형). 투입 칸은 상자가 더 오는 동안 늘 비워 둔다."""
        out = []
        keep = set(self.cfg.stage_order) if self.cfg.keep_stage else set()
        for s in range(self.n_slots):
            busy = (self.slots[s] is not None and s != src) or ((s == 0 or s in keep) and self.more_coming())
            if busy:
                out.append(self.cell_rect[s])
        return out

    def _temps(self, src) -> list:
        """src 말고 칸에 있는 상자 (x0, x1, y0, y1, 높이) mm. 칸 안에서 흡착면이 위, L 이 x 방향."""
        out = []
        for s, jid in enumerate(self.slots):
            if jid is None or s == src:
                continue
            L, W, H = self.cfg.box_types[self.job[jid][1]]
            x, y = self.cells[s]
            out.append((x - L / 2, x + L / 2, y - W / 2, y + W / 2, H))
        return out

    def slot_feasible(self, src, k, o):
        """칸 src 의 상자 (종류 k) 를 방향 o 로 놓을 수 있는 칸 (feasible 위에 칸 규칙을 더한 것)."""
        ok, base, sup, nx, ny = self.feasible(k, o)
        c = self.cfg
        if not (c.zone or c.slot_clear) or not ok.any():
            return ok, base, sup, nx, ny
        g = c.grid
        dims = self.dims(k, o)
        xi = np.arange(self.NX) * g
        yj = np.arange(self.NY) * g
        x0, x1, y0, y1 = xi, xi + nx * g, yj, yj + ny * g
        bad = np.zeros_like(ok)
        if c.zone and not self.zone_open:               # 아직 830 안에만
            bad |= (x1 > c.depth + TOL)[:, None]
        if c.zone:
            for a0, a1, b0, b1 in self._reserved(src):
                bad |= ((x0 < a1) & (x1 > a0))[:, None] & ((y0 < b1) & (y1 > b0))[None, :]
        if c.slot_clear:
            face = xi + nx * g / 2 + dims[0] / 2
            yc = yj + ny * g / 2
            zc = base + dims[2] / 2
            r = c.arm_radius
            for a0, a1, b0, b1, h in self._temps(src):
                sweep = ((x0 < a1) & (x1 + PRE_INSERT_MM > a0))[:, None] & ((y0 < b1) & (y1 > b0))[None, :] & (h > base + TOL)
                arm = ((face < a1) & (face + PRE_INSERT_MM + c.arm_pass > a0))[:, None] & ((yc - r < b1) & (yc + r > b0))[None, :] \
                    & (h > zc - r)
                bad |= sweep | arm
        return ok & ~bad, base, sup, nx, ny

    # ---------- 높이 지도 보조 배열 (버전마다 다시 계산) ----------
    def _aux_get(self, key, fn):
        hit = self._aux.get(key)
        if hit is None or hit[0] != self.version:
            hit = (self.version, fn())
            self._aux[key] = hit
        return hit[1]

    def _front_max(self):
        def f():
            suf = np.maximum.accumulate(self.hm[::-1], axis=0)[::-1]
            fm = np.zeros_like(self.hm)
            fm[:-1] = suf[1:]
            return fm
        return self._aux_get('fm', f)

    def _dead_integral(self, z):
        def f():
            fm, hm = self._front_max(), self.hm
            now = np.maximum(0, fm - hm)
            return _integral(np.maximum(0, np.maximum(fm, z) - hm) - now)
        return self._aux_get(('dead', z), f)

    def _contact_integral(self, v, dz):
        return self._aux_get(('ct', v, dz), lambda: _integral(np.clip((self.hm - v) / dz, 0, 1)))

    def _flush_integral(self, z):
        return self._aux_get(('fl', z), lambda: _integral((np.abs(self.hm - z) < TOL).astype(np.float64)))

    def _hm_integral(self):
        return self._aux_get('hm', lambda: _integral(self.hm))

    # ---------- 다시 꺼내기 ----------
    def _block_matrix(self):
        """B[a, b]: 실린 상자 a 가 b 의 앞(출구 쪽)이나 위를 막는다 (pack_env._blocks 와 같은 정의)."""
        def f():
            lo, hi = self.lo, self.hi
            ov = [(np.minimum(hi[:, None, a], hi[None, :, a]) - np.maximum(lo[:, None, a], lo[None, :, a])) > TOL for a in range(3)]
            B = ((lo[:, None, 0] >= hi[None, :, 0] - TOL) & ov[1] & ov[2]) | ((lo[:, None, 2] >= hi[None, :, 2] - TOL) & ov[0] & ov[1])
            np.fill_diagonal(B, False)
            return B
        return self._aux_get('B', f)

    def _path_clear(self, i, j, nx, ny, base, dims) -> bool:
        """컨테이너 자리 (i, j) 의 상자를 앞으로 빼거나 넣는 길과 팔이 버퍼 칸 상자와 부딪히지 않는지 (slot_clear 와 같은 식)."""
        c, g = self.cfg, self.cfg.grid
        if not c.slot_clear:
            return True
        x0, x1, y0, y1 = i * g, (i + nx) * g, j * g, (j + ny) * g
        face, yc, zc, r = x0 + nx * g / 2 + dims[0] / 2, y0 + ny * g / 2, base + dims[2] / 2, c.arm_radius
        for a0, a1, b0, b1, h in self._temps(None):
            if x0 < a1 and x1 + PRE_INSERT_MM > a0 and y0 < b1 and y1 > b0 and h > base + TOL:
                return False
            if face < a1 and face + PRE_INSERT_MM + c.arm_pass > a0 and yc - r < b1 and yc + r > b0 and h > zc - r:
                return False
        return True

    def retrievable(self) -> list:
        """다시 꺼낼 수 있는 실린 상자 (placed 번호, 풀리는 위반 수, 놓일 버퍼 칸). 앞과 위가 비고, 꺼내는 길이 열려 있고,
        빈 버퍼 칸이 있고, 아직 max_retrieve_per_box 번 안 꺼낸 상자."""
        c = self.cfg
        if not c.retrieve or not self.placed:
            return []
        cells = self.staging_slots()
        if not cells:
            return []
        B = self._block_matrix()
        free = ~B.any(0)
        out = []
        for r in np.flatnonzero(free):
            p = self.placed[r]
            if p.get('kept') or p['cell'] is None or self.retrieved.get(p['jid'], 0) >= c.max_retrieve_per_box:
                continue
            nx, ny = cells_of(p['dims_xyz'], c)
            if not self._path_clear(p['cell'][0], p['cell'][1], nx, ny, p['lo'][2], p['dims_xyz']):
                continue
            unblocks = int((B[r] & (self.pstop < p['stop'])).sum())
            out.append((int(r), unblocks, cells[0]))
        return out

    def _recompute_hm(self, i0, i1, j0, j1):
        """높이 지도 [i0:i1, j0:j1] 를 남은 상자로 다시 계산한다 (각 칸 = 그 칸을 덮는 상자 윗면의 최대)."""
        sub = np.zeros((i1 - i0, j1 - j0))
        for p in self.placed:
            if p.get('kept') or p['cell'] is None:
                continue
            nx, ny = cells_of(p['dims_xyz'], self.cfg)
            a0, b0 = max(i0, p['cell'][0]), max(j0, p['cell'][1])
            a1, b1 = min(i1, p['cell'][0] + nx), min(j1, p['cell'][1] + ny)
            if a0 < a1 and b0 < b1:
                v = sub[a0 - i0:a1 - i0, b0 - j0:b1 - j0]
                np.maximum(v, p['hi'][2], out=v)
        self.hm[i0:i1, j0:j1] = sub

    # ---------- 후보 ----------
    def candidates(self) -> dict:
        """지금 가능한 모든 이동. 배열 dict (길이 C) 와 특징 행렬 'feat' (C, CAND_DIM)."""
        c, g = self.cfg, self.cfg.grid
        rows = []
        for s, jid in enumerate(self.slots):
            if jid is None:
                continue
            stop, k = self.job[jid]
            for o in self.orients(k):
                ok, base, sup, nx, ny = self.slot_feasible(s, k, o)
                ii, jj = corner_positions(ok, base)
                if self.arm_tab is not None and len(ii):
                    ii, jj = self._arm_corners(k, o, ok, base, ii, jj, nx, ny)
                if len(ii) == 0:
                    continue
                dims = self.dims(k, o)
                n = len(ii)
                rows.append(dict(kind=np.full(n, PLACE), src=np.full(n, s), jid=np.full(n, jid), stop=np.full(n, stop),
                                 type=np.full(n, k), o=np.full(n, o), i=ii, j=jj, nx=np.full(n, nx), ny=np.full(n, ny),
                                 base=base[ii, jj], sup=sup[ii, jj], dims=np.tile(np.array(dims, float), (n, 1)), dst=np.full(n, -1),
                                 pidx=np.full(n, -1), unblocks=np.zeros(n, int)))
        if rows:
            cand = {key: np.concatenate([r[key] for r in rows]) for key in rows[0]}
            cand['blocks'] = self._blocks(cand)
            if c.safe_order and (cand['blocks'] == 0).any():
                cand = {key: v[cand['blocks'] == 0] for key, v in cand.items()}
        else:
            cand = None
        empty = self.staging_slots()
        if not c.zone:
            empty = empty[:1]                       # 빈 칸 중 첫 칸 하나 (zone 이 아니면 칸마다 차이가 없다)
        if self.slots[0] is not None and empty:
            stop, k = self.job[self.slots[0]]
            n = len(empty)
            st = dict(kind=np.full(n, STAGE), src=np.zeros(n, int), jid=np.full(n, self.slots[0]), stop=np.full(n, stop),
                      type=np.full(n, k), o=np.zeros(n, int), i=np.zeros(n, int), j=np.zeros(n, int), nx=np.zeros(n, int),
                      ny=np.zeros(n, int), base=np.zeros(n), sup=np.zeros(n), dims=np.tile(np.array(self.dims(k, 0), float), (n, 1)),
                      dst=np.array(empty), blocks=np.zeros(n, int), pidx=np.full(n, -1), unblocks=np.zeros(n, int))
            cand = st if cand is None else {key: np.concatenate([cand[key], st[key]]) for key in st}
        ret = self.retrievable() if any(x is not None for x in self.slots) else []      # 넣을 상자가 있을 때만
        if ret:
            g = c.grid
            P = [self.placed[r] for r, _, _ in ret]
            n = len(ret)
            nxy = [cells_of(p['dims_xyz'], c) for p in P]
            rt = dict(kind=np.full(n, RETRIEVE), src=np.full(n, -1), jid=np.array([p['jid'] for p in P]),
                      stop=np.array([p['stop'] for p in P]), type=np.array([p['type'] for p in P]), o=np.array([p['o'] for p in P]),
                      i=np.array([p['cell'][0] for p in P]), j=np.array([p['cell'][1] for p in P]),
                      nx=np.array([a for a, _ in nxy]), ny=np.array([b for _, b in nxy]), base=np.array([p['lo'][2] for p in P]),
                      sup=np.zeros(n), dims=np.array([p['dims_xyz'] for p in P], float), dst=np.array([d for _, _, d in ret]),
                      blocks=np.zeros(n, int), pidx=np.array([r for r, _, _ in ret]), unblocks=np.array([u for _, u, _ in ret]))
            cand = rt if cand is None else {key: np.concatenate([cand[key], rt[key]]) for key in rt}
        if cand is None:
            return dict(n=0)
        cand['n'] = len(cand['kind'])
        cand['feat'] = self._features(cand)
        return cand

    def _centers(self, cand):
        g = self.cfg.grid
        d = cand['dims']
        return np.stack([(cand['i'] + cand['nx'] / 2) * g, (cand['j'] + cand['ny'] / 2) * g, cand['base'] + d[:, 2] / 2], 1)

    def _blocks(self, cand):
        """pack_env._blocks 와 같은 계산을 후보 전체에 한 번에: 먼저 내릴 상자의 앞이나 위를 막는 수."""
        if len(self.placed) == 0:
            return np.zeros(len(cand['kind']), int)
        ctr = self._centers(cand)
        lo, hi = ctr - cand['dims'] / 2, ctr + cand['dims'] / 2
        plo, phi = self.lo[None], self.hi[None]
        ov = [(np.minimum(hi[:, None, a], phi[..., a]) - np.maximum(lo[:, None, a], plo[..., a])) > TOL for a in range(3)]
        front = (lo[:, None, 0] >= phi[..., 0] - TOL) & ov[1] & ov[2]
        above = (lo[:, None, 2] >= phi[..., 2] - TOL) & ov[0] & ov[1]
        earlier = self.pstop[None, :] < cand['stop'][:, None]
        return ((front | above) & earlier).sum(1)

    def _features(self, cand):
        c, g = self.cfg, self.cfg.grid
        C = cand['n']
        F = np.zeros((C, self.cand_dim), np.float32)
        D, W, H = self.D, c.width, c.height
        d = cand['dims']
        place = cand['kind'] == PLACE
        i, j, nx, ny, b = cand['i'], cand['j'], cand['nx'], cand['ny'], cand['base']
        F[:, 6:9] = d / 170.0
        F[np.arange(C), 9 + np.searchsorted(self.types, cand['type'])] = 1
        F[:, 14] = cand['stop'] / 12.0
        known = [self.job[x][0] for x in self.slots if x is not None] + [s for s, _ in self.job[self.next:self.next + LOOK]]
        F[:, 15] = (cand['stop'] - min(known, default=1)) / max(1, self.n_stop - 1)
        has_src = cand['src'] >= 0
        F[np.flatnonzero(has_src), 16 + cand['src'][has_src]] = 1
        F[:, 20] = cand['kind'] == STAGE
        F[:, 28] = cand['blocks'] / 3.0
        F[:, 30] = (len(self.staging_slots()) if c.zone else sum(x is None for x in self.slots[1:])) / max(1, c.n_stage)
        F[:, 31] = d.prod(1) / 170.0 ** 3
        if c.zone:
            st = np.flatnonzero(~place)
            for q in st:                                  # 옮길 칸의 자리
                x0, x1, y0, y1 = self.cell_rect[int(cand['dst'][q])]
                F[q, 0], F[q, 1], F[q, 3], F[q, 4] = x0 / D, y0 / W, x1 / D, y1 / W
        if c.retrieve:
            rt = np.flatnonzero(cand['kind'] == RETRIEVE)
            if len(rt):
                lo = np.array([self.placed[int(r)]['lo'] for r in cand['pidx'][rt]])
                hi = np.array([self.placed[int(r)]['hi'] for r in cand['pidx'][rt]])
                F[rt, 0], F[rt, 1], F[rt, 2] = lo[:, 0] / D, lo[:, 1] / W, lo[:, 2] / H
                F[rt, 3], F[rt, 4], F[rt, 5] = hi[:, 0] / D, hi[:, 1] / W, hi[:, 2] / H
                F[rt, self.ret_off] = 1
                F[rt, self.ret_off + 1] = cand['unblocks'][rt] / 3.0
        if place.any():
            p = np.flatnonzero(place)
            ip, jp, nxp, nyp, bp = i[p], j[p], nx[p], ny[p], b[p]
            dz = d[p, 2]
            z1 = bp + dz
            F[p, 0], F[p, 1], F[p, 2] = ip * g / D, jp * g / W, bp / H
            F[p, 3], F[p, 4], F[p, 5] = (ip + nxp) * g / D, (jp + nyp) * g / W, z1 / H
            F[p, 21] = cand['sup'][p]
            Shm = self._hm_integral()
            under = bp * nxp * nyp - _rect(Shm, ip, jp, ip + nxp, jp + nyp)
            F[p, 22] = under * g * g / d[p].prod(1)
            F[p, 29] = (D - (ip + nxp) * g) / D
            for r, q in enumerate(p):
                ii, jj, nn, mm, bb, zz, ddz = ip[r], jp[r], nxp[r], nyp[r], bp[r], z1[r], dz[r]
                Sd = self._dead_integral(zz)
                F[q, 23] = _rect(Sd, 0, jj, ii, jj + mm) * g * g / self.vol * 50
                Sc = self._contact_integral(bb, ddz)
                Sf = self._flush_integral(zz)
                back = 1.0 if ii == 0 else _rect(Sc, ii - 1, jj, ii, jj + mm) / mm
                left = 1.0 if jj == 0 else _rect(Sc, ii, jj - 1, ii + nn, jj) / nn
                right = 1.0 if jj + mm >= self.NY else _rect(Sc, ii, jj + mm, ii + nn, jj + mm + 1) / nn
                fl = 0.0
                if ii > 0:
                    fl += _rect(Sf, ii - 1, jj, ii, jj + mm)
                if jj > 0:
                    fl += _rect(Sf, ii, jj - 1, ii + nn, jj)
                if jj + mm < self.NY:
                    fl += _rect(Sf, ii, jj + mm, ii + nn, jj + mm + 1)
                F[q, 24], F[q, 25], F[q, 26] = back, left, right
                F[q, 27] = fl / (mm + 2 * nn)
            if c.zone:
                x0, x1 = ip * g, (ip + nxp) * g
                F[p, CAND_DIM] = np.clip(x1 - c.depth, 0, None) / np.maximum(1, x1 - x0)
                for s in self.staging_slots():          # 덮으면 재배치 칸을 잃는다
                    a0, a1, b0, b1 = self.cell_rect[s]
                    y0, y1 = jp * g, (jp + nyp) * g
                    F[p, CAND_DIM + 1] += ((x0 < a1) & (x1 > a0) & (y0 < b1) & (y1 > b0)) / max(1, c.n_stage)
        return F

    def state_vector(self) -> np.ndarray:
        c = self.cfg
        hm = self.hm / c.height
        px, py = -(-self.NX // POOL), -(-self.NY // POOL)
        pad = np.zeros((px * POOL, py * POOL))
        pad[: self.NX, : self.NY] = hm
        blk = pad.reshape(px, POOL, py, POOL)
        v = [blk.max((1, 3)).ravel(), blk.mean((1, 3)).ravel()]

        def item(jid):
            f = np.zeros(7)
            if jid is not None and jid < len(self.job):
                stop, k = self.job[jid]
                f[0] = 1
                f[1 + self.types.index(k)] = 1
                f[6] = stop / 12.0
            return f
        v += [item(x) for x in self.slots]
        v += [item(self.next + t) for t in range(LOOK)]
        known = [self.job[x][0] for x in self.slots if x is not None]
        n = max(len(self.job), self.total_hint or 0, 1)
        v.append(np.array([self.util, self.next / n, (n - self.next) / 200.0, self.n_stop / 12.0, self.violations / 5.0,
                           sum(x is not None for x in self.slots[1:]) / max(1, c.n_stage),
                           (min(known) if known else 0) / 12.0, len(self.placed) / 200.0]))
        if c.zone:
            v.append(np.array([len(self.staging_slots()) / max(1, c.n_stage), float(self.more_coming())]
                              + ([float(self.zone_open)] if c.zone_late else [])))
        return np.concatenate(v).astype(np.float32)

    # ---------- 진행 ----------
    def step(self, idx: int) -> float:
        """후보 idx 를 실행하고 보상을 돌려준다. 끝나면 self.done."""
        cand = self.cand
        if int(cand['kind'][idx]) == STAGE:
            return self.do(STAGE, 0, dst=int(cand['dst'][idx]))
        if int(cand['kind'][idx]) == RETRIEVE:
            return self.do(RETRIEVE, int(cand['pidx'][idx]), dst=int(cand['dst'][idx]))
        return self.do(PLACE, int(cand['src'][idx]), int(cand['o'][idx]), int(cand['i'][idx]), int(cand['j'][idx]),
                       blocks=int(cand['blocks'][idx]))

    def place_info(self, src: int, o: int, i: int, j: int) -> dict:
        """칸 src 의 상자를 방향 o 로 격자 (i, j) 에 놓을 때의 바닥 높이·중심·치수 (규칙 검사는 하지 않는다)."""
        stop, k = self.job[self.slots[src]]
        dims = self.dims(k, o)
        nx, ny = cells_of(dims, self.cfg)
        base = float(self.hm[i:i + nx, j:j + ny].max())
        center = np.array([(i + nx / 2) * self.cfg.grid, (j + ny / 2) * self.cfg.grid, base + dims[2] / 2])
        return dict(stop=stop, type=k, dims=dims, nx=nx, ny=ny, base=base, center=center)

    def do(self, kind: int, src: int, o: int = 0, i: int = 0, j: int = 0, dst: int = -1, blocks: int | None = None) -> float:
        """이동 하나를 실행한다. kind=STAGE: 투입 칸 -> 버퍼 칸 dst. kind=PLACE: 칸 src -> 컨테이너 (o, i, j).
        kind=RETRIEVE: 실린 상자 placed[src] 를 다시 꺼내 버퍼 칸 dst 에 둔다."""
        c = self.cfg
        if kind == RETRIEVE:
            return self._retrieve(src, dst)
        jid = self.slots[src]
        stop, k = self.job[jid]
        before = dict(slots=list(self.slots), placed=len(self.placed))     # 이 이동 직전 장면 (모션 플래너용)
        if kind == STAGE:
            assert src == 0 and dst in self.staging_slots()
            self.slots[dst], self.slots[0] = jid, None
            self.n_stage_moves += 1
            self.moves.append(dict(kind='stage', jid=jid, type=k, stop=stop, src=0, dst=dst, before=before))
            self._arrive()
            reward = -c.w_stage
        else:
            info = self.place_info(src, o, i, j)
            dims, center, nx, ny, base = info['dims'], info['center'], info['nx'], info['ny'], info['base']
            lo, hi = center - np.array(dims) / 2, center + np.array(dims) / 2
            if blocks is None:
                one = dict(kind=np.array([PLACE]), stop=np.array([stop]), dims=np.array([dims], float), i=np.array([i]),
                           j=np.array([j]), nx=np.array([nx]), ny=np.array([ny]), base=np.array([base]))
                blocks = int(self._blocks(one)[0])
            self.hm[i:i + nx, j:j + ny] = base + dims[2]
            self.version += 1
            p = self._add_placed(jid, k, stop, o, center, dims, src, cell=(i, j))
            frac = float(np.prod(dims)) / self.vol
            self.util += frac
            self.violations += blocks
            self.slots[src] = None
            self.moves.append(dict(kind='place', jid=jid, type=k, stop=stop, src=src, o=o, center=p['center'],
                                   dims_xyz=list(dims), blocks=blocks, before=before))
            if src == 0:
                self._arrive()
            reward = c.w_util * frac - c.w_block * blocks
        self.cand = self.candidates()
        return reward + self._check_end()

    def _retrieve(self, r: int, dst: int) -> float:
        c = self.cfg
        assert dst in self.staging_slots()
        p = self.placed[r]
        before = dict(slots=list(self.slots), placed=len(self.placed))
        B = self._block_matrix()
        assert not B[:, r].any(), '앞이나 위가 막힌 상자는 꺼낼 수 없다'
        unblocks = int((B[r] & (self.pstop < p['stop'])).sum())
        self.placed.pop(r)
        self.lo = np.delete(self.lo, r, 0)
        self.hi = np.delete(self.hi, r, 0)
        self.pstop = np.delete(self.pstop, r)
        nx, ny = cells_of(p['dims_xyz'], c)
        i, j = p['cell']
        self._recompute_hm(i, i + nx, j, j + ny)
        self.version += 1
        frac = float(np.prod(p['dims_xyz'])) / self.vol
        self.util -= frac
        self.violations -= unblocks
        self.slots[dst] = p['jid']
        self.retrieved[p['jid']] = self.retrieved.get(p['jid'], 0) + 1
        self.ret_bonus[p['jid']] = unblocks
        self.n_retrieve += 1
        self.moves.append(dict(kind='retrieve', jid=p['jid'], type=p['type'], stop=p['stop'], o=p['o'], center=p['center'],
                               dims_xyz=list(p['dims_xyz']), dst=dst, unblocks=unblocks, before=before))
        self.cand = self.candidates()
        return -c.w_util * frac + c.w_block * unblocks - c.w_retrieve + self._check_end()

    def _add_placed(self, jid, k, stop, o, center, dims, src, cell=None, kept=False):
        lo, hi = center - np.array(dims) / 2, center + np.array(dims) / 2
        p = dict(id=f'b{jid + 1:03d}', jid=jid, type=k, stop=stop, o=o, center=np.round(center, 2).tolist(),
                 dims_xyz=list(dims), lo=lo, hi=hi, cell=cell, src=src, kept=kept)
        self.placed.append(p)
        self.lo = np.vstack([self.lo, lo])
        self.hi = np.vstack([self.hi, hi])
        self.pstop = np.append(self.pstop, stop)
        return p

    def _check_end(self) -> float:
        """할 이동이 없으면 끝낸다. zone 이면 칸에 남은 상자를 그 자리에 둔 채로 싣는다 (그 보상을 돌려준다)."""
        if self.cand['n'] != 0:
            return 0.0
        if self.cfg.zone and not self.zone_open and any(x is not None for x in self.slots):
            self.zone_open = True                   # 830 안에 놓을 자리가 없다: 버퍼 구역을 연다
            self.cand = self.candidates()
            if self.cand['n'] != 0:
                return 0.0
        if self.cfg.zone and not self.closed and self.more_coming():
            if self.slots[0] is None:               # 실제 운용: 다음 상자를 기다린다 (extend() 가 이어 간다)
                self.done, self.reason = True, 'waiting'
                return 0.0
            # 놓을 자리가 없는데 상자가 더 온다: 여기서 적재를 마감한다 (투입 칸을 비워 둘 필요가 없어진다)
            self.closed = True
            self.cand = self.candidates()
            if self.cand['n'] != 0:
                return 0.0
        self.done = True
        reward = 0.0
        if self.cfg.zone and not self.more_coming():
            c = self.cfg
            for s, jid in enumerate(self.slots):
                if jid is None:
                    continue
                stop, k = self.job[jid]
                L, W, H = c.box_types[k]
                x, y = self.cells[s]
                dims = (L, W, H)
                center = np.array([x, y, H / 2])
                one = dict(kind=np.array([PLACE]), stop=np.array([stop]), dims=np.array([dims], float),
                           i=np.array([(x - L / 2) / c.grid]), j=np.array([(y - W / 2) / c.grid]),
                           nx=np.array([L / c.grid]), ny=np.array([W / c.grid]), base=np.array([0.0]))
                blocks = int(self._blocks(one)[0])
                self._add_placed(jid, k, stop, -1, center, dims, s, kept=True)
                frac = float(np.prod(dims)) / self.vol
                self.util += frac
                self.violations += blocks
                self.n_kept += 1
                self.slots[s] = None
                self.moves.append(dict(kind='keep', jid=jid, type=k, stop=stop, src=s, center=np.round(center, 2).tolist(),
                                       dims_xyz=list(dims), blocks=blocks))
                reward += c.w_util * frac - c.w_block * blocks
        empty = all(x is None for x in self.slots) and self.next >= len(self.job)
        self.reason = 'done' if empty else ('closed' if self.closed else 'no_space')
        if self.cfg.retrieve:                       # 꺼내 놓고 다시 넣지 않은 상자: 꺼낼 때 받은 위반 보상을 되돌린다
            reward -= self.cfg.w_block * sum(self.ret_bonus.get(j, 0) for j in self.slots if j is not None and j in self.retrieved)
        return reward

    def finish(self) -> float:
        """zone: 더 받지 않고 적재를 마감한다. 놓을 자리가 없을 때와 같다 (예: 모션 검사를 통과하는 이동이 없을 때).
        마감하면 칸 자리까지 쓸 수 있다. 이미 마감했으면 남은 이동 없이 끝내고 칸에 남은 상자는 그 자리에 둔 채로 싣는다."""
        assert self.cfg.zone
        self.done, self.reason = False, None
        if self.closed:
            self.cand = dict(self.cand, n=0)
        else:
            self.closed = True
            self.cand = self.candidates()
        return self._check_end()

    def skip_arrival(self) -> None:
        """투입 칸 상자를 싣지 않고 뺀다 (실제 운용: 로봇이 넣을 수 없는 상자를 작업자가 꺼낸다. 다음 트럭으로)."""
        assert self.slots[0] is not None
        self.skipped.append(self.slots[0])
        self.slots[0] = None
        self._arrive()
        self.done, self.reason = False, None
        self.cand = self.candidates()
        self._check_end()

    def left(self) -> int:
        """싣지 못한 상자 수 (버퍼에 남은 것 + 아직 들어오지 않은 것 + 뺀 것)."""
        return sum(x is not None for x in self.slots) + len(self.job) - self.next + len(self.skipped)

    def metrics(self) -> dict:
        n_place = len(self.placed)
        vol = sum(float(np.prod(p['dims_xyz'])) for p in self.placed)
        n_moves = sum(m['kind'] != 'keep' for m in self.moves)
        return dict(util=self.util, util_rig=vol / self.vol_rig, placed=n_place, kept=self.n_kept, total=len(self.job),
                    left=self.left(), violations=self.violations, stage_moves=self.n_stage_moves, retrieves=self.n_retrieve,
                    abandoned=sum(j is not None and j in self.retrieved for j in self.slots), moves=n_moves,
                    moves_per_box=n_moves / max(1, n_place), reason=self.reason)


# ---------- 기준 정책 ----------
def greedy_index(cand: dict, stage_when_stuck: bool = True) -> int:
    """pack_env.greedy 를 버퍼로 넓힌 것: 놓을 수 있는 상자 중 안쪽 -> 낮은 곳 -> y 작은 쪽 순으로 첫 자리.
    투입 칸 상자를 놓을 수 없을 때만 버퍼로 옮긴다."""
    place = np.flatnonzero(cand['kind'] == PLACE)
    if len(place):
        key = np.lexsort((cand['j'][place], cand['base'][place], cand['i'][place]))
        return int(place[key[0]])
    stage = np.flatnonzero(cand['kind'] == STAGE)
    return int(stage[0]) if len(stage) else 0           # 남은 것이 다시 꺼내기뿐이면 첫 후보 (상자마다 횟수 제한이 있어 끝난다)


def unload_plan(placed: list) -> list:
    """하차 시뮬레이션의 단계 목록. 지점 1부터, 그 지점 상자 중 앞(출구 쪽)과 위가 빈 것부터 꺼낸다.
    그 지점 상자가 다른 지점 상자에 막혀 있으면 막는 상자를 잠시 꺼냈다가 그 지점이 끝나면 다시 넣는다 (재작업).
    막힘은 pack_env._blocks 와 같은 정의다 (앞: 출구 쪽에서 y·z 가 겹침, 위: x·y 가 겹침).
    돌려줌: [(지점, 'out' | 'aside' | 'back', [placed 번호, ...]), ...]  out = 하차, aside = 잠시 꺼냄, back = 다시 넣음."""
    n = len(placed)
    steps = []
    if n == 0:
        return steps
    lo = np.array([p['lo'] for p in placed], float)
    hi = np.array([p['hi'] for p in placed], float)
    stop = np.array([p['stop'] for p in placed])
    ov = [(np.minimum(hi[:, None, a], hi[None, :, a]) - np.maximum(lo[:, None, a], lo[None, :, a])) > TOL for a in range(3)]
    B = ((lo[:, None, 0] >= hi[None, :, 0] - TOL) & ov[1] & ov[2]) | ((lo[:, None, 2] >= hi[None, :, 2] - TOL) & ov[0] & ov[1])
    np.fill_diagonal(B, False)                          # B[a, b]: a 가 b 의 앞이나 위를 막는다
    present = np.ones(n, bool)
    for s in sorted(set(stop.tolist())):
        moved = np.zeros(n, bool)
        while True:
            tgt = present & (stop == s)
            if not tgt.any():
                break
            free = present & ~B[present].any(0)
            if (tgt & free).any():
                present[tgt & free] = False
                steps.append((s, 'out', np.flatnonzero(tgt & free).tolist()))
                continue
            direct = free & ~tgt & B[:, tgt].any(1)
            if not direct.any():                        # 막는 상자도 막혀 있으면 막는 사슬 전체에서 빈 것부터
                need = tgt.copy()
                while True:
                    add = present & ~need & B[:, need].any(1)
                    if not add.any():
                        break
                    need |= add
                direct = free & need & ~tgt
                if not direct.any():
                    break
            present[direct] = False
            moved |= direct
            steps.append((s, 'aside', np.flatnonzero(direct).tolist()))
        if moved.any():
            present |= moved
            steps.append((s, 'back', np.flatnonzero(moved).tolist()))
    return steps


def unload_check(placed: list) -> dict:
    """unload_plan 요약. rework (재작업으로 꺼낸 횟수 합, 같은 상자를 여러 지점에서 다시 꺼내면 여러 번),
    rework_boxes (한 번이라도 꺼내야 했던 상자 수), rework_stops (재작업이 있었던 지점 수)."""
    steps = unload_plan(placed)
    aside = [(s, ids) for s, k, ids in steps if k == 'aside']
    return dict(rework=sum(len(ids) for _, ids in aside), rework_boxes=len({i for _, ids in aside for i in ids}),
                rework_stops=len({s for s, _ in aside}))


def make_job(rng: np.random.Generator, cfg: BufferConfig, kind: str = 'mixed') -> list:
    """학습·평가용 작업 목록 [(지점, 종류)], 도착 순서대로.

    team    pack_env.make_job 과 같은 분포 (지점 6~12, 지점당 3~10개, 나중 지점부터)
    team_random  team 과 같고 도착 순서만 무작위 (pack_env order="random")
    full    지점 6곳, 공간 부피의 1.2~1.4배 (다 못 싣는다), 종류 균등, 나중 지점부터
    full_random  full 과 같고 도착 순서만 무작위
    full_near    full 과 같고 도착 순서를 조금 섞음 (이웃한 상자끼리 자리 바꿈, 상자 수의 20%)
    mixed   학습용: 지점 3~12, 부피 0.6~1.5배, 종류 비율 무작위, 순서 = 정렬 60% / 거의 정렬 30% / 무작위 10%
    mixed_random  mixed 와 같고 순서 = 정렬 10% / 거의 정렬 10% / 무작위 80%
    """
    types = sorted(cfg.box_types)
    vols = np.array([np.prod(cfg.box_types[k]) for k in types], float)
    space = space_volume(cfg)
    base = np.array([(cfg.shares or {}).get(k, 1.0) for k in types], float)     # cfg.shares (종류별 비율), 없으면 균등
    if kind in ('team', 'team_random'):
        n_stop = int(rng.integers(6, 13))
        job = []
        for stop in range(n_stop, 0, -1):
            job += [(stop, int(rng.choice(types))) for _ in range(int(rng.integers(3, 11)))]
        if kind == 'team_random':
            rng.shuffle(job)
        return [(int(s), int(k)) for s, k in job]
    if kind in ('full', 'full_random', 'full_near'):
        n_stop, frac, shares = 6, rng.uniform(1.2, 1.4), base.copy()
        order = {'full': 'sorted', 'full_random': 'random', 'full_near': 'near'}[kind]
    else:
        n_stop = int(rng.integers(3, 13))
        frac = rng.uniform(0.6, 1.5)
        v = rng.random()
        if cfg.shares:                                      # 비율을 정했으면 그 비율 60% / 무작위 20% / 균등 20%
            shares = base.copy() if v < 0.6 else (rng.dirichlet(np.full(len(types), 2.0)) if v < 0.8 else np.ones(len(types)))
        else:
            shares = rng.dirichlet(np.full(len(types), 2.0)) if v < 0.6 else np.ones(len(types))
        u = rng.random()
        if kind == 'mixed_random':                          # 무작위 도착 위주 (정렬 10% / 거의 정렬 10% / 무작위 80%)
            order = 'sorted' if u < 0.1 else ('near' if u < 0.2 else 'random')
        else:
            order = 'sorted' if u < 0.6 else ('near' if u < 0.9 else 'random')
    shares = shares / shares.sum()
    target = frac * space
    ks, total = [], 0.0
    while total < target:
        t = int(rng.choice(len(types), p=shares))
        ks.append(types[t])
        total += vols[t]
    weights = rng.dirichlet(np.full(n_stop, 3.0))
    stops = rng.choice(np.arange(1, n_stop + 1), size=len(ks), p=weights)
    job = list(zip(stops.tolist(), ks))
    if order == 'random':
        rng.shuffle(job)
    else:
        job.sort(key=lambda b: -b[0])                       # 나중에 내릴 지점부터 도착
        if order == 'near':
            for _ in range(len(job) // 5):
                a = int(rng.integers(0, len(job) - 3))
                b = a + int(rng.integers(1, 4))
                job[a], job[b] = job[b], job[a]
    return [(int(s), int(k)) for s, k in job]
