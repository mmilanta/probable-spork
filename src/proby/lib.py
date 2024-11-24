from __future__ import annotations
from typing import Callable, Iterator
from dataclasses import dataclass
from enum import Enum
from polynome import Polynome, Monome, Monad


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


class Probe:
    def __init__(self) -> None:
        self.future = []
        self.index = -1
    
    def reset(self):
        self.index = -1

    def run(self):
        if self.index == len(self.future) - 1:
            raise StopIteration()
        self.index += 1
        return self.future[self.index]


class ProbeSet:
    def __init__(self) -> None:
        self.probes: dict[Point, Probe] = {}

    def reset(self) -> None:
        for probe in self.probes.values():
            probe.reset()

    def get_probe(self, point: Point) -> Probe:
        self.probes.setdefault(point, Probe())
        return self.probes[point]

    def freeze(self) -> FrozenProbeSet:
        return FrozenProbeSet(probes=tuple(tuple([k, tuple(v.future)]) for k, v in self.probes.items()))
    
    def futures_len(self):
        return sum(len(probe.future) for probe in self.probes.values())


@dataclass(frozen=True, eq=True)
class FrozenProbeSet:
    probes: tuple[tuple['Point', tuple[bool, ...]], ...]

    def unfreeze(self) -> ProbeSet:
        probeset = ProbeSet()
        for point, future in self.probes:
            probeset.get_probe(point).future = list(future)
        return probeset

    def to_polynome(self) -> Polynome:
        polynome = Polynome(monomes=(Monome(coefficient=1, monads=tuple()), ))
        one_monome = Monome(coefficient=1, monads=tuple())
        for point, futures in self.probes:
            for future in futures:
                if future:
                    polynome = polynome * Polynome(monomes=(Monome(coefficient=1, monads=(Monad(sym=point, power=1), )), ))
                else:
                    polynome = polynome * Polynome(monomes=(one_monome, Monome(coefficient=-1, monads=(Monad(sym=point, power=1), )), ))
        return polynome

GameDirectedGraph = dict[tuple | None, dict[FrozenProbeSet, tuple | GameEnd]]


class Point():
    def __init__(self, run_fn: Callable[[Probe], bool]) -> None:
        self.run_fn: Callable[[Probe], bool] = run_fn
        # self._tree: dict[tuple|None, dict[InstructionTreeTuple, int]] | None = {None: {}}
        self._directed_graph: GameDirectedGraph = {}
        self._path_to_state: dict[tuple, FrozenProbeSet] = {}

    def __hash__(self) -> int:
        return hash(self.run_fn)

    def __repr__(self) -> str:
        return self.run_fn.__name__

    def __call__(self, x: ProbeSet, *, state: tuple | None = None) -> bool:
        try:
            return x.get_probe(self).run()
        except StopIteration:
            raise PointException(point=self, state=state)


    def create_dg(self) -> None:
        self._directed_graph = {}
        to_compute: set[tuple | None] = {None}
        def _dfs(score: tuple | None, probe_set: ProbeSet):
            try:
                probe_set.reset()

                if self.run_fn(probe_set):
                    self._directed_graph[score][probe_set.freeze()] = GameEnd.WIN
                else:
                    self._directed_graph[score][probe_set.freeze()] = GameEnd.LOSE
            except PointException as e:
                if e.state is None or e.state == score:
                    probe_set.probes[e.point].future.append(True)
                    _dfs(score=score, probe_set=probe_set)
                    probe_set.probes[e.point].future[-1] = False
                    _dfs(score=score, probe_set=probe_set)
                    probe_set.probes[e.point].future.pop()
                else:
                    self._directed_graph[score][probe_set.freeze()] = e.state
                    if e.state not in self._directed_graph:
                        to_compute.add(e.state)
                    if e.state not in self._path_to_state or sum(len(probe[1]) for probe in self._path_to_state[e.state]) < probe_set.futures_len():
                        self._path_to_state[e.state] = probe_set.freeze()

        while to_compute:
            sc = to_compute.pop()
            start_ps = self._path_to_state[sc].unfreeze() if sc is not None else ProbeSet()
            self._directed_graph.setdefault(sc, {})
            _dfs(score=sc, probe_set=start_ps)

    def compute_polynomial(self) -> None:
        polys: dict[tuple | None, Polynome] = {}
        def _get_polynomial(calling_stack: list[tuple], root: tuple) -> Polynome:
            if root in calling_stack:
                raise NotImplementedError("This has not been solved yet")
            if root not in self._directed_graph:
                raise ValueError("root must be in graph")
            if root in polys:
                return polys[root]
            calling_stack.append(root)
            partian_polynoms: list[Polynome] = []
            for edge in self._directed_graph[root]:
                edge_poly = edge.to_polynome()
                if self._directed_graph[root][edge] == GameEnd.WIN:
                    partian_polynoms.append(edge_poly)
                elif self._directed_graph[root][edge] == GameEnd.LOSE:
                    continue
                else:
                    next_poly = _get_polynomial(calling_stack=calling_stack, root=self._directed_graph[root][edge])
                    partian_polynoms.append(next_poly * edge_poly)
            calling_stack.pop()
            polys[root] = sum(partian_polynoms)
            return polys[root]
        return _get_polynomial(calling_stack=[], root=None)


def point(fn: Callable[[FrozenProbeSet], bool]) -> Point:
    return Point(run_fn=fn)


@point
def tp(x):
    return True

N = 7
@point
def without_advantages(x):
    pp = [0, 0]
    while max(pp) < N or abs(pp[0] - pp[1]) < 2:
        pp[tp(x, state=tuple(pp))] += 1
        if pp == [N, N]:
            pp = [N - 1, N - 1]
    return pp[1] > pp[0]


without_advantages.create_dg()
print("Done")