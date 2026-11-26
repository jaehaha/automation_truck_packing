"""BoxTable: 프레임워크가 가진 유일한 장면 정보.
  id -> {size, current, goal, state}

state 는 {'pending', 'placed', 'failed'}. placed 된 상자들의 goal 이 다음 상자의 장애물이 된다.

../pipeline/details_by_modules/unified.md §2 참고.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np


@dataclass
class BoxEntry:
    size: tuple[float, float, float]            # L, W, H (m)
    current: np.ndarray | None = None           # 월드 좌표
    goal: np.ndarray | None = None              # 월드 좌표
    size_xyz: tuple[float, float, float] | None = None   # 놓인 방향 (x, y, z 길이)
    state: str = "pending"                      # pending / placed / failed


@dataclass
class BoxTable:
    boxes: dict[str, BoxEntry] = field(default_factory=dict)

    def add(self, box_id: str, size: tuple[float, float, float]):
        self.boxes[box_id] = BoxEntry(size=size)

    def ids(self) -> list[str]:
        return list(self.boxes.keys())

    def remaining(self) -> list[str]:
        return [i for i, b in self.boxes.items() if b.state == "pending"]

    def placed(self) -> list[str]:
        return [i for i, b in self.boxes.items() if b.state == "placed"]

    def set_current(self, box_id: str, current):
        self.boxes[box_id].current = np.asarray(current, float)

    def set_goal(self, box_id: str, goal, size_xyz=None):
        self.boxes[box_id].goal = np.asarray(goal, float)
        if size_xyz is not None:
            self.boxes[box_id].size_xyz = tuple(size_xyz)

    def done(self, box_id: str):
        """실행 완료 신호: current <- goal, state=placed."""
        b = self.boxes[box_id]
        b.current = b.goal
        b.state = "placed"

    def fail(self, box_id: str):
        self.boxes[box_id].state = "failed"

    def remove(self, box_id: str):
        """UnPacking 완료: 버퍼로 꺼낸 상자를 표에서 제거."""
        self.boxes.pop(box_id, None)

    def __getitem__(self, box_id: str) -> BoxEntry:
        return self.boxes[box_id]
