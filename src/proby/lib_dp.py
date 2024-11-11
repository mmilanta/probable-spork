from typing import Optional
from typing import Callable
from time import time
from inspect import signature
from typing import Iterable
from polynome import SymbolicVariable, Monad, Monome, Polynome
from collections import namedtuple
from time import time
from enum import Enum


class GameEnd(Enum):
    WIN = 1
    LOSE = 2


class FutureEndedException(Exception):
    def __init__(self, sym_var: SymbolicVariable):
        super().__init__(f"SymbolicVariable {sym_var.id}")
        self.sym_var: SymbolicVariable = sym_var


class GameOver(Exception):
    def __init__(self, p1_wins: bool):
        super().__init__(f"Player {'1' if p1_wins else '2'} won")
        self.p1_wins: bool = p1_wins


class Probe:
    def __init__(self, sym_var: SymbolicVariable, future: list[bool]) -> None:
        self.sym_var = sym_var
        self.future = future
        self.index = -1
    
    def reset(self):
        self.index = -1

    def run(self):
        if self.index == len(self.future) - 1:
            raise FutureEndedException(sym_var=self.sym_var)
        self.index += 1
        return self.future[self.index]


FrozenProbeSet = tuple[tuple[SymbolicVariable, tuple[bool, ...]], ...]
GameDirectedGraph = dict[tuple, dict[FrozenProbeSet, tuple | GameEnd]]


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


Score = namedtuple('Score', ['total', 'p1', 'p2'])
def play_match(score: Score, p: Probe, q: Probe) -> tuple | GameEnd:
    sp = p if ((score.p1 + score.p2 + 1) // 2) % 2 else q
    if sp.run():
        score = Score(total=score.total, p1=score.p1 + 1, p2=score.p2)
    else:
        score = Score(total=score.total, p1=score.p1, p2=score.p2 + 1)
    
    # advantages
    # if score.p1 == score.p2 == score.total - 1:
    #     score = Score(total=score.total, p1=score.p1 - 1, p2=score.p2 - 1)
    
    if score.p1 == score.total:
        return GameEnd.WIN
    elif score.p2 == score.total:
        return GameEnd.LOSE
    return score


def create_dg(score: tuple, play_fn: Callable) -> GameDirectedGraph:
    sym_vars = [SymbolicVariable(id=s) for s in list(signature(play_fn).parameters)[1:]]

    dg: GameDirectedGraph = {}
    to_compute: set[tuple] = {score}
    def _dfs(score: tuple, probes: ProbeSet) -> None:
        try:
            probes.reset()
            n_score = play_fn(score, **probes.to_params())
            dg.setdefault(score, {})
            dg[score][probes.freeze()] = n_score
            if isinstance(n_score, tuple) and n_score not in dg:
                to_compute.add(n_score)
        except FutureEndedException as e:
            probes.probes[e.sym_var].future.append(True)
            _dfs(score=score, probes=probes)
            probes.probes[e.sym_var].future[-1] = False
            _dfs(score=score, probes=probes)
            probes.probes[e.sym_var].future.pop()
    while to_compute:
        probes=ProbeSet(probes=[Probe(sym_var=sym_var, future=[]) for sym_var in sym_vars])
        _dfs(score=to_compute.pop(), probes=probes)
    return dg


def get_poly_from_frozen_probe_set(frozen_probe_set: FrozenProbeSet) -> Polynome:
    polynome = Polynome(monomes=(Monome(coefficient=1, monads=tuple()), ))
    one_monome = Monome(coefficient=1, monads=tuple())
    for sym_var, futures in frozen_probe_set:
        for future in futures:
            if future:
                polynome = polynome * Polynome(monomes=(Monome(coefficient=1, monads=(Monad(sym=sym_var, power=1), )), ))
            else:
                polynome = polynome * Polynome(monomes=(one_monome, Monome(coefficient=-1, monads=(Monad(sym=sym_var, power=1), )), ))
    return polynome


def get_polynomial(graph: GameDirectedGraph, root: tuple) -> Polynome:
    polys: dict[tuple, Polynome] = {}
    def get_polynomial(calling_stack: list[tuple], root: tuple) -> Polynome:
        if root in calling_stack:
            raise NotImplementedError("This has not been solved yet")
        if root not in graph:
            raise ValueError("root must be in graph")
        if root in polys:
            return polys[root]
        calling_stack.append(root)
        partian_polynoms: list[Polynome] = []
        for edge in graph[root]:
            edge_poly = get_poly_from_frozen_probe_set(edge)
            if graph[root][edge] == GameEnd.WIN:
                partian_polynoms.append(edge_poly)
            elif graph[root][edge] == GameEnd.LOSE:
                continue
            else:
                next_poly = get_polynomial(calling_stack=calling_stack, root=graph[root][edge])
                partian_polynoms.append(next_poly * edge_poly)
        calling_stack.pop()
        polys[root] = sum(partian_polynoms)
        return polys[root]
    return get_polynomial(calling_stack=[], root=root)

s_game = Score(total=13, p1=0, p2=0)
s_time = time()
dg = create_dg(s_game, play_fn=play_match)
print(f"TOOK: {time() - s_time}")
s_time = time()
p = get_polynomial(graph=dg, root=s_game)
print(f"TOOK: {time() - s_time}")

print("DONE")