from __future__ import annotations
from typing import Callable, Iterator
from dataclasses import dataclass
from enum import Enum


class GameEnd(Enum):
    WIN = 1
    LOSE = 2


@dataclass(frozen=True, eq=True)
class PointScore:
    point: Point
    win: int
    lose: int


class PointException(Exception):
    def __init__(self, point: Point, state: tuple | None):
        # Call the base class constructor with the parameters it needs
        super().__init__(f"Point {point.run_fn.__name__}")
        self.point: Point = point
        self.state: tuple | None = state

class FutureEndedException(Exception):
    pass


def to_instruction_tree_iterator(instruction_tree: InstructionTree) -> InstructionTreeIterator:
    return {k: iter(v) for k, v in instruction_tree.items()}


def to_instruction_tree_tuple(instruction_tree: InstructionTree) -> InstructionTreeTuple:
    return tuple(
        PointScore(point=p, win=sum(v), lose=len(v) - sum(v)) for p, v in instruction_tree.items() if v
    )

FrozenProbeSet = tuple[tuple['Point', tuple[bool, ...]], ...]
GameDirectedGraph = dict[tuple | None, dict[FrozenProbeSet, tuple | GameEnd]]


class Probe:
    def __init__(self, future: list[bool]) -> None:
        self.future = future
        self.index = -1
    
    def reset(self):
        self.index = -1

    def run(self):
        if self.index == len(self.future) - 1:
            raise StopIteration()
        self.index += 1
        return self.future[self.index]


class ProbeSet:
    def __init__(self, probes: list[Probe]) -> None:
        if len(set(probe.sym_var.id for probe in probes)) < len(probes):
            raise ValueError("There can be only one probe per sym_var. Multiple found")
        self.probes: dict[SymbolicVariable, Probe] = {probe.sym_var: probe for probe in probes}
    def to_params(self) -> dict[str, Probe]:
        return {k.id: v for k, v in self.probes.items()}
    def freeze(self) -> FrozenProbeSet:
        return tuple(tuple([k, tuple(v.future)]) for k, v in self.probes.items())
    def reset(self):
        for pk in self.probes:
            self.probes[pk].reset()
    def __str__(self) -> str:
        parts = []
        for p in self.probes:
            parts.append(self.probes[p].sym_var.id + ": " + "".join(['T' if f_ else 'F' for f_ in self.probes[p].future]))
        return " - ".join(parts)


class Point():
    def __init__(self, run_fn: Callable[[Probe], bool]) -> None:
        self.run_fn: Callable[[Probe], bool] = run_fn
        # self._tree: dict[tuple|None, dict[InstructionTreeTuple, int]] | None = {None: {}}
        self._directed_graph: GameDirectedGraph = {}

    def __hash__(self) -> int:
        return hash(self.run_fn)

    def __repr__(self) -> str:
        return self.run_fn.__name__

    def __call__(self, x: dict[Point, Probe], *, state: tuple | None = None) -> bool:
        x.setdefault(self, Probe())
        try:
            x[self].run()
        except StopIteration:
            raise PointException(point=self, state=state)


    def create_dg(self) -> None:
        self._directed_graph = {}
        to_compute: set[tuple | None] = {None}
        def _dfs(score: tuple | None, probes: dict[Point, Probe]):
            try:
                for probe in probes.values():
                    probe.reset()

                if self.run_fn(probe):
                    pass
                else:
                    pass
            except PointException as e:
                if e.state is None:
                    probes[e.point].future.append(True)
                    _dfs(score=score, probes=probes)
                    probes[e.point].future[-1] = False
                    _dfs(score=score, probes=probes)
                    probes[e.point].future.pop()
                else:
                    

        while to_compute:
            _dfs(score=to_compute.pop(), probes={})
    def _dfs_binary_tree(self, instruction_tree: InstructionTree) -> None:


        if self._tree is None:
            raise ValueError("set self._tree to {} before. Call this function by calling compute_tree")
        try:
            if self.run_fn(to_instruction_tree_iterator(instruction_tree)):
                instruction_tuple = to_instruction_tree_tuple(instruction_tree)
                self._tree[None].setdefault(instruction_tuple, 0)
                self._tree[None][instruction_tuple] += 1

        except PointException as e:
            if e.state is None:
                instruction_tree.setdefault(e.point, [])
                instruction_tree[e.point].append(True)
                self._dfs_binary_tree(instruction_tree)
                instruction_tree[e.point][-1] = False
                self._dfs_binary_tree(instruction_tree)
                instruction_tree[e.point].pop()
            else:
                pass


    def compute_tree(self, sub_points: bool = True) -> None:
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
def tp(x):
    return True


@point
def without_advantages(x):
    pp = [0, 0]
    while max(pp) < 8:
        pp[tp(x)] += 1
    return pp[1] > pp[0]


without_advantages.compute_tree()
import random
from random import choice
