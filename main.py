from __future__ import annotations
from typing import Callable, Iterator
from time import time
from dataclasses import dataclass


@dataclass(frozen=True, eq=True)
class PointScore:
    point: Point
    win: int
    lose: int


class PointException(Exception):
    def __init__(self, point: Point):
        # Call the base class constructor with the parameters it needs
        super().__init__(f"Point {point.run_fn.__name__}")
        self.point: Point = point


def to_instruction_tree_iterator(instruction_tree: InstructionTree) -> InstructionTreeIterator:
    return {k: iter(v) for k, v in instruction_tree.items()}


def to_instruction_tree_tuple(instruction_tree: InstructionTree) -> InstructionTreeTuple:
    return tuple(
        PointScore(point=p, win=sum(v), lose=len(v) - sum(v)) for p, v in instruction_tree.items() if v
    )


class Point():
    def __init__(self, run_fn: Callable[[InstructionTreeIterator], bool]) -> None:
        self.run_fn: Callable[[InstructionTreeIterator], bool] = run_fn
        self._tree: dict[InstructionTreeTuple, int] | None = None

    def __hash__(self) -> int:
        return hash(self.run_fn)

    def __repr__(self) -> str:
        return self.run_fn.__name__

    def __call__(self, x: InstructionTreeIterator) -> bool:
        try:
            iterator = x.get(self, iter([]))
            return next(iterator)
        except StopIteration:
            raise PointException(point=self)

    def _dfs_binary_tree(self, instruction_tree: InstructionTree):
        if self._tree is None:
            raise ValueError("set self._tree to {} before. Call this function by calling compute_tree")
        try:
            if self.run_fn(to_instruction_tree_iterator(instruction_tree)):
                instruction_tuple = to_instruction_tree_tuple(instruction_tree)
                self._tree.setdefault(instruction_tuple, 0)
                self._tree[instruction_tuple] += 1

        except PointException as e:
            instruction_tree.setdefault(e.point, [])
            instruction_tree[e.point].append(True)
            self._dfs_binary_tree(instruction_tree)
            instruction_tree[e.point].pop()

            instruction_tree[e.point].append(False)
            self._dfs_binary_tree(instruction_tree)
            instruction_tree[e.point].pop()

    def compute_tree(self, sub_points: bool = True):
        if self._tree is not None:
            return
        self._tree = {}
        self._dfs_binary_tree(instruction_tree={})

        if sub_points:
            for key in self._tree:
                for point_score in key:
                    point_score.point.compute_tree()

    def probability(self, base_point: dict[Point, float]) -> float:
        return self._probability(tuple(
            (p, prob) for p, prob in base_point.items()
        ))

    def _probability(self, base_point: tuple[tuple[Point, float], ...]) -> float:
        for point, prob in base_point:
            if point is self:
                return prob
        p: float = 0.0
        if self._tree is None:
            raise ValueError("Call compute_tree first")
        for wc, count in self._tree.items():
            pw: float = 1.0
            for point_score in wc:
                pp = point_score.point._probability(base_point=base_point)
                pw *= (pp ** point_score.win) * ((1 - pp) ** point_score.lose)
            p += pw * count
        return p


InstructionTreeIterator = dict[Point, Iterator[bool]]
InstructionTree = dict[Point, list[bool]]
InstructionTreeTuple = tuple[PointScore, ...]


def point(fn: Callable[[InstructionTreeIterator], bool]) -> Point:
    return Point(run_fn=fn)


@point
def core_point(x):
    return True


@point
def game(x):
    p1 = 0
    p2 = 0
    while max(p1, p2) < 4 or abs(p1 - p2) < 2:
        if core_point(x):
            p1 += 1
        else:
            p2 += 1
        if p1 != p2 and max(p1, p2) > 12:
            break
    return p1 > p2


@point
def tie_break(x):
    p1 = 0
    p2 = 0
    while max(p1, p2) < 7 or abs(p1 - p2) < 2:
        if core_point(x):
            p1 += 1
        else:
            p2 += 1
        if p1 != p2 and max(p1, p2) > 8:
            break
    return p1 > p2


@point
def set(x):
    p1 = 0
    p2 = 0
    while max(p1, p2) < 6 or (abs(p1 - p2) == 1 and max(p1, p2) == 6):
        if game(x):
            p1 += 1
        else:
            p2 += 1
    if p1 == p2:
        return tie_break(x)
    else:
        return p1 > p2


@point
def match(x):
    p1 = 0
    p2 = 0
    while max(p1, p2) < 3:
        if set(x):
            p1 += 1
        else:
            p2 += 1
    return p1 > p2


@point
def game_dry(x):
    p1 = 0
    p2 = 0
    while max(p1, p2) < 5:
        if core_point(x):
            p1 += 1
        else:
            p2 += 1
    return p1 > p2


s = time()
from rich import print
match.compute_tree()
game_dry.compute_tree()
print(f"computed in {time() - s}s")
import numpy as np
import matplotlib.pyplot as plt
ps = np.arange(.5, .9, 0.001)
probs = {}
matches = {}
for pp in [match, set, game, tie_break, game_dry]:
    for ppp in ps:
        matches.setdefault(pp, []).append(pp.probability({core_point: ppp}))

for k in matches:
    plt.plot(ps, matches[k], label=str(k))
plt.legend()
plt.show()
