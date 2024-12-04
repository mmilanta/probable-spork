from __future__ import annotations
from typing import Callable, Iterator
from dataclasses import dataclass
from enum import Enum
from polynome import Polynome, Monome, Monad, SymbolicVariable


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
                    polynome = polynome * Polynome(monomes=(Monome(coefficient=1, monads=(Monad(sym=str(point), power=1), )), ))
                else:
                    polynome = polynome * Polynome(monomes=(one_monome, Monome(coefficient=-1, monads=(Monad(sym=str(point), power=1), )), ))
        return polynome

    def futures_len(self) -> int:
        return sum(len(future) for _, future in self.probes)

GameDirectedGraph = dict[tuple | None, dict[FrozenProbeSet, tuple | GameEnd]]


class Point():
    def __init__(self, run_fn: Callable[[Probe], bool]) -> None:
        self.run_fn: Callable[[Probe], bool] = run_fn
        # self._tree: dict[tuple|None, dict[InstructionTreeTuple, int]] | None = {None: {}}
        self._directed_graph: GameDirectedGraph = None
        self._path_to_state: dict[tuple, FrozenProbeSet] = {}
        self._poly: Polynome | None = None
        self._sym_to_point: dict[SymbolicVariable, Point] = {}

    def __hash__(self) -> int:
        return hash(self.run_fn)

    def __repr__(self) -> str:
        return self.run_fn.__name__

    def __call__(self, x: ProbeSet, *, state: tuple | None = None) -> bool:
        try:
            return x.get_probe(self).run()
        except StopIteration:
            raise PointException(point=self, state=state)


    def create_dg(self, compute_for_subpoints: bool = True) -> None:
        if self._directed_graph is not None:
            return
        sub_points: set[Point] = set()
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
                sub_points.add(e.point)
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
                    if e.state not in self._path_to_state or self._path_to_state[e.state].futures_len() < probe_set.futures_len():
                        self._path_to_state[e.state] = probe_set.freeze()
            if compute_for_subpoints:
                for point in sub_points:
                    point.create_dg()

        while to_compute:
            sc = to_compute.pop()
            start_ps = self._path_to_state[sc].unfreeze() if sc is not None else ProbeSet()
            self._directed_graph.setdefault(sc, {})
            _dfs(score=sc, probe_set=start_ps)

    def _compute_polynomial(self) -> None:
        if self._poly is not None:
            return self._poly
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
                for point, _ in edge.probes:
                    self._sym_to_point.setdefault(str(point), point)
                if self._directed_graph[root][edge] == GameEnd.WIN:
                    partian_polynoms.append(edge_poly)
                elif self._directed_graph[root][edge] == GameEnd.LOSE:
                    continue
                else:
                    next_poly = _get_polynomial(
                        calling_stack=calling_stack,
                        root=self._directed_graph[root][edge]
                    )                       
                    partian_polynoms.append(next_poly * edge_poly)
            calling_stack.pop()
            polys[root] = sum(partian_polynoms)
            return polys[root]
        self._poly = _get_polynomial(calling_stack=[], root=None)
        for point in self._sym_to_point.values():
            point._compute_polynomial()

    def get_poly(self, variable_points: set[Point]) -> Polynome:
        variable_syms = set(str(point) for point in variable_points)
        self.create_dg(compute_for_subpoints=True)
        out_poly = self._poly
        for sym_var in self._poly.get_sym_vars():
            if sym_var in variable_syms:
                continue
            out_poly = out_poly.substitute(sym_var, self._sym_to_point[sym_var].get_poly(variable_points=variable_points))
        return out_poly

            






def point(fn: Callable[[FrozenProbeSet], bool]) -> Point:
    return Point(run_fn=fn)


@point
def tp(x):
    return True

N = 7
@point
def without_advantages(x):
    pp = [0, 0]
    while max(pp) < N:
        pp[tp(x, state=tuple(pp))] += 1
    return pp[1] > pp[0]



@point
def core_point(x):
    return True


@point
def game(x):
    p1 = 0
    p2 = 0
    while max(p1, p2) < 4 or abs(p1 - p2) < 2:
        if core_point(x, state=(p1, p2)):
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
        if core_point(x, state=(p1, p2)):
            p1 += 1
        else:
            p2 += 1
        if p1 != p2 and max(p1, p2) > 8:
            break
    return p1 > p2


@point
def set_tennis(x):
    p1 = 0
    p2 = 0
    while max(p1, p2) < 6 or (abs(p1 - p2) == 1 and max(p1, p2) == 6):
        if game(x, state=(p1, p2)):
            p1 += 1
        else:
            p2 += 1
    if p1 == p2:
        return tie_break(x, state=(p1, p2))
    else:
        return p1 > p2



set_tennis.create_dg()
set_tennis.compute_polynomial()
tp.create_dg()
tp.compute_polynomial()
print("Done")