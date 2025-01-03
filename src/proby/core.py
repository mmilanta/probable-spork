from typing import Callable
from time import time
from inspect import signature
from enum import Enum
import numpy as np
from typing import NamedTuple
from dataclasses import dataclass
from itertools import chain


@dataclass(frozen=True, eq=True)
class SymbolicVariable:
    id: str
    def __le__(self, other: "SymbolicVariable") -> bool:
        return self.id <= other.id

    def __str__(self) -> str:
        return self.id


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


@dataclass(frozen=True, eq=True)
class FrozenProbe():
    sym_var: SymbolicVariable
    future: tuple[bool, ...]


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

    def freeze(self) -> FrozenProbe:
        return FrozenProbe(sym_var=self.sym_var, future=tuple(self.future))


@dataclass(frozen=True, eq=True)
class FrozenProbeSet():
    probes: tuple[FrozenProbe, ...]

    def compute_probability(self, values: dict[SymbolicVariable, np.ndarray]) -> np.ndarray:
        shape = next(iter(values.values())).shape
        out = np.ones(shape)

        for frozen_probe in self.probes:
            for future_step in frozen_probe.future:
                if future_step:
                    out = out * values[frozen_probe.sym_var]
                else:
                    out = out * (1 - values[frozen_probe.sym_var])
        return out


class ProbeSet:
    def __init__(self, probes: list[Probe]) -> None:
        if len(set(probe.sym_var.id for probe in probes)) < len(probes):
            raise ValueError("There can be only one probe per sym_var. Multiple found")
        self.probes: dict[SymbolicVariable, Probe] = {probe.sym_var: probe for probe in probes}
    def to_params(self) -> dict[str, Probe]:
        return {k.id: v for k, v in self.probes.items()}
    def freeze(self) -> FrozenProbeSet:
        return FrozenProbeSet(
            probes=tuple(probe.freeze() for probe in self.probes.values())
        )
    def reset(self):
        for pk in self.probes:
            self.probes[pk].reset()
    def __str__(self) -> str:
        parts = []
        for p in self.probes:
            parts.append(self.probes[p].sym_var.id + ": " + "".join(['T' if f_ else 'F' for f_ in self.probes[p].future]))
        return " - ".join(parts)


class GameDirectedGraph:
    def __init__(self, dg: dict[tuple, dict[FrozenProbeSet, tuple | GameEnd]]) -> None:
        self.dg = dg

    @classmethod
    def from_play_func(self, play_func: Callable, starting_score: tuple) -> "GameDirectedGraph":
        sym_vars = [SymbolicVariable(id=s) for s in list(signature(play_func).parameters)[1:]]
        dg: dict[tuple, dict[FrozenProbeSet, tuple | GameEnd]] = {}
        to_compute: set[tuple] = {starting_score}
        def _dfs(score: tuple, probes: ProbeSet) -> None:
            try:
                probes.reset()
                n_score = play_func(score, **probes.to_params())
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
        return self(dg=dg)

    def compute_probability(self, root: tuple, **kwargs) -> np.ndarray:
        values: dict[SymbolicVariable, np.ndarray] = {}
        shape = None
        for k, val in kwargs.items():
            assert isinstance(k, str)
            assert isinstance(val, np.ndarray)
            values[SymbolicVariable(id=k)] = val
            if shape is None:
                shape = val.shape
            else:
                assert shape == val.shape
        probs: dict[tuple, Pol] = {}
        def get_probs(calling_stack: list[tuple], root: tuple) -> Pol:
            if root in calling_stack:
                return Pol(mons=(Mon(score=root, coeff=1.0),))
            if root not in self.dg:
                raise ValueError(f"state {root} not in graph must be in graph")
            if root in probs:
                return probs[root]
            calling_stack.append(root)
            out_probs: Pol = Pol(mons=tuple([]))
            for edge in self.dg[root]:
                edge_probs = edge.compute_probability(values)
                target = self.dg[root][edge]
                if target == GameEnd.WIN:
                    out_probs += Pol(mons=(Mon(score=None, coeff=edge_probs),))
                elif target == GameEnd.LOSE:
                    pass
                elif isinstance(target, tuple):
                    next_probs = get_probs(calling_stack=calling_stack, root=target)
                    out_probs += (next_probs * edge_probs)
            calling_stack.pop()

            for mon in out_probs.mons:
                if mon.score == root:
                    out_probs = Pol(mons=tuple(Mon(score=imon.score, coeff=imon.coeff / (1 - mon.coeff)) for imon in out_probs.mons if imon != mon))

            if len(out_probs.mons) == 1 and out_probs.mons[0].score is None:
                probs[root] = out_probs
            return out_probs

        prob_pol = get_probs(calling_stack=[], root=root)
        assert len(prob_pol.mons) == 1 and prob_pol.mons[0].score is None

        return prob_pol.mons[0].coeff



@dataclass(frozen=True, eq=True)
class Mon:
    score: tuple | None
    coeff: np.ndarray


@dataclass(frozen=True, eq=True)
class Pol:
    mons: tuple[Mon, ...]

    def __add__(self, other: "Pol") -> "Pol":
        monome_kdx: dict[tuple|None, np.ndarray] = {}
        for monome in chain(self.mons, other.mons):
            monome_kdx.setdefault(monome.score, 0)
            monome_kdx[monome.score] += monome.coeff
        return Pol(mons=tuple(Mon(score=k, coeff=v) for k, v in monome_kdx.items()))

    def __radd__(self, other: "Pol") -> "Pol": 
        if other == 0:
            return self
        else:
            return other + self
    def __mul__(self, other: float) -> "Pol":
        return Pol(mons=tuple(Mon(score=mon.score, coeff=mon.coeff * other) for mon in self.mons))




if __name__ == "__main__":
    class TieBreakScore(NamedTuple):
        p1: int
        p2: int
        p1_serving: bool
    N = 100
    def play_tie_break(score: TieBreakScore, p: Probe, q: Probe) -> TieBreakScore | GameEnd:
        sp = p if score.p1_serving else q
        if sp.run():
            score = TieBreakScore(p1=score.p1 + 1, p2=score.p2, p1_serving=score.p1_serving)
        else:
            score = TieBreakScore(p1=score.p1, p2=score.p2 + 1, p1_serving=score.p1_serving)
        if (score.p1 + score.p2) % 2 == 1:
            score = TieBreakScore(p1=score.p1, p2=score.p2, p1_serving=not score.p1_serving)
        if score.p1 == score.p2 == N - 1:
            score = TieBreakScore(p1=N - 2, p2=N - 2, p1_serving=score.p1_serving)
        if max(score.p1, score.p2) == N:
            return GameEnd.WIN if score.p1 > score.p2 else GameEnd.LOSE
        return score

    graph = GameDirectedGraph.from_play_func(play_tie_break, starting_score=TieBreakScore(p1=0, p2=0, p1_serving=True))

    def print_dg(graph: GameDirectedGraph):
        out = ""
        for state in graph.dg:
            out += str(state) + "\n"
            for k in graph.dg[state]:
                out += "    * " + str(graph.dg[state][k]) + "\n"
        return out
    # print(print_dg(dg))
    s = time()
    out = graph.compute_probability(p=np.arange(0,1, .01), q=np.arange(0,1, .01), root=TieBreakScore(p1=0, p2=0, p1_serving=True))      
    print(time() - s)   
    print(out)  
