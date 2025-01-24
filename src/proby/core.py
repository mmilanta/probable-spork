from time import time
from inspect import signature
from enum import Enum
from typing import NamedTuple, Literal, Any, Callable
from dataclasses import dataclass
from itertools import chain
import random

AnyTuple = tuple[Any, ...]


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


@dataclass(frozen=True, eq=True)
class FrozenProbe:
    sym_var: SymbolicVariable
    future: tuple[bool, ...]


class Probe:
    def __init__(self, sym_var: SymbolicVariable, future: list[bool]) -> None:
        self.sym_var = sym_var
        self.future = future
        self.index = -1

    def reset(self) -> None:
        self.index = -1

    def run(self) -> bool:
        if self.index == len(self.future) - 1:
            raise FutureEndedException(sym_var=self.sym_var)
        self.index += 1
        return self.future[self.index]

    def freeze(self) -> FrozenProbe:
        return FrozenProbe(sym_var=self.sym_var, future=tuple(self.future))


@dataclass(frozen=True, eq=True)
class FrozenProbeSet:
    probes: tuple[FrozenProbe, ...]

    def compute_probability(
        self, values: dict[SymbolicVariable, float]
    ) -> float:
        out = 1.0

        for frozen_probe in self.probes:
            for future_step in frozen_probe.future:
                if future_step:
                    out = out * values[frozen_probe.sym_var]
                else:
                    out = out * (1 - values[frozen_probe.sym_var])
        return out

    def size(self) -> int:
        return sum(len(probe.future) for probe in self.probes)

    def dump(self) -> list[dict[str, Any]]:
        return [
            {"id": probe.sym_var.id, "future": probe.future}
            for probe in self.probes
        ]


class ProbeSet:
    def __init__(self, probes: list[Probe]) -> None:
        if len(set(probe.sym_var.id for probe in probes)) < len(probes):
            raise ValueError(
                "There can be only one probe per sym_var. Multiple found"
            )
        self.probes: dict[SymbolicVariable, Probe] = {
            probe.sym_var: probe for probe in probes
        }

    def to_params(self) -> dict[str, Probe]:
        return {k.id: v for k, v in self.probes.items()}

    def freeze(self) -> FrozenProbeSet:
        return FrozenProbeSet(
            probes=tuple(probe.freeze() for probe in self.probes.values())
        )

    def reset(self) -> None:
        for pk in self.probes:
            self.probes[pk].reset()

    def __str__(self) -> str:
        parts = []
        for p in self.probes:
            parts.append(
                self.probes[p].sym_var.id
                + ": "
                + "".join(["T" if f_ else "F" for f_ in self.probes[p].future])
            )
        return " - ".join(parts)


@dataclass(frozen=True, eq=True)
class Mon:
    score: AnyTuple | None
    coeff: float


@dataclass(frozen=True, eq=True)
class Pol:
    mons: tuple[Mon, ...]

    def __add__(self, other: "Pol") -> "Pol":
        monome_kdx: dict[AnyTuple | None, float] = {}
        for monome in chain(self.mons, other.mons):
            if monome.score not in monome_kdx:
                monome_kdx[monome.score] = 0
            monome_kdx[monome.score] += monome.coeff
        return Pol(
            mons=tuple(Mon(score=k, coeff=v) for k, v in monome_kdx.items())
        )

    def __radd__(self, other: "Pol" | Literal[0]) -> "Pol":
        if other == 0:
            return self
        else:
            return other + self

    def __mul__(self, other: float) -> "Pol":
        return Pol(
            mons=tuple(
                Mon(score=mon.score, coeff=mon.coeff * other)
                for mon in self.mons
            )
        )


class GameDirectedGraph:
    def __init__(
        self,
        dg: dict[AnyTuple, dict[FrozenProbeSet, AnyTuple | GameEnd]],
        root: AnyTuple,
    ) -> None:
        self.dg = dg
        self.root = root

    @classmethod
    def from_play_func(
        self,
        play_func: Callable[..., AnyTuple | GameEnd],
        root: AnyTuple,
    ) -> "GameDirectedGraph":
        sym_vars = [
            SymbolicVariable(id=s)
            for s in list(signature(play_func).parameters)[1:]
        ]
        dg: dict[AnyTuple, dict[FrozenProbeSet, AnyTuple | GameEnd]] = {}
        to_compute: set[AnyTuple] = {root}

        def _dfs(score: AnyTuple, probes: ProbeSet) -> None:
            try:
                probes.reset()
                n_score = play_func(score, **probes.to_params())
                if score not in dg:
                    dg[score] = {}
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
            probes = ProbeSet(
                probes=[
                    Probe(sym_var=sym_var, future=[]) for sym_var in sym_vars
                ]
            )
            _dfs(score=to_compute.pop(), probes=probes)
        return self(dg=dg, root=root)

    @classmethod
    def _parse_kwargs_probes(
        cls, kwargs: dict[str, float]
    ) -> dict[SymbolicVariable, float]:
        values: dict[SymbolicVariable, float] = {}
        for k, val in kwargs.items():
            assert isinstance(k, str)
            try:
                val = float(val)
            except (TypeError, ValueError):
                raise ValueError(
                    f"Probe {k} value must be a float, or something that can be casted to float. Found {type(val)}"
                )
            if val > 1 or val < 0:
                raise ValueError(
                    f"Probe {k} value must be between 0 and 1, found {val}."
                )
            values[SymbolicVariable(id=k)] = val
        return values

    def compute_expected_value(self, **kwargs: float) -> float:
        values = GameDirectedGraph._parse_kwargs_probes(kwargs)
        means: dict[AnyTuple, Pol] = {}

        def get_means(calling_stack: list[AnyTuple], root: AnyTuple) -> Pol:
            if root in calling_stack:
                return Pol(mons=(Mon(score=root, coeff=1.0),))
            if root not in self.dg:
                raise ValueError(f"state {root} not in graph must be in graph")
            if root in means:
                return means[root]
            calling_stack.append(root)
            out_means: Pol = Pol(mons=tuple([]))
            for edge in self.dg[root]:
                edge_probs = edge.compute_probability(values)
                target = self.dg[root][edge]
                if isinstance(target, tuple):
                    next_means = get_means(
                        calling_stack=calling_stack, root=target
                    )
                else:
                    next_means = Pol(mons=tuple([]))
                out_means += (
                    next_means
                    + Pol(mons=(Mon(score=None, coeff=edge.size()),))
                ) * edge_probs
            calling_stack.pop()

            for mon in out_means.mons:
                if mon.score == root:
                    out_means = Pol(
                        mons=tuple(
                            Mon(
                                score=imon.score,
                                coeff=imon.coeff / (1 - mon.coeff),
                            )
                            for imon in out_means.mons
                            if imon != mon
                        )
                    )

            if len(out_means.mons) == 1 and out_means.mons[0].score is None:
                means[root] = out_means
            return out_means

        mean_pol = get_means(calling_stack=[], root=self.root)
        assert len(mean_pol.mons) == 1 and mean_pol.mons[0].score is None

        return mean_pol.mons[0].coeff

    def compute_probability(self, **kwargs: float) -> float:
        return (
            self._compute_probability(self.root, **kwargs)[self.root]
            .mons[0]
            .coeff
        )

    def compute_importance(self, **kwargs: float) -> dict[AnyTuple, float]:
        probs = self._compute_probability(self.root, **kwargs)
        while set(probs.keys()) != set(self.dg.keys()):
            for state in self.dg:
                if state not in probs:
                    new_probs = self._compute_probability(state, **kwargs)
                    break
            for new_state in new_probs:
                probs[new_state] = new_probs[new_state]

        importance: dict[AnyTuple, float] = {}
        for state in self.dg:
            if state not in probs:
                raise ValueError(f"state {state} not in probs")
            max_prob = 0.0
            min_prob = 1.0
            for edge in self.dg[state]:
                next_state = self.dg[state][edge]
                if next_state == GameEnd.LOSE:
                    prob = 0.0
                elif next_state == GameEnd.WIN:
                    prob = 1.0
                elif isinstance(next_state, tuple):
                    prob = probs[next_state].mons[0].coeff
                max_prob = max(max_prob, prob)
                min_prob = min(min_prob, prob)
            importance[state] = max_prob - min_prob
        return importance

    def _compute_probability(
        self, root: AnyTuple, **kwargs: float
    ) -> dict[AnyTuple, Pol]:
        values = GameDirectedGraph._parse_kwargs_probes(kwargs)
        probs: dict[AnyTuple, Pol] = {}

        def get_probs(calling_stack: list[AnyTuple], root: AnyTuple) -> Pol:
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
                    next_probs = get_probs(
                        calling_stack=calling_stack, root=target
                    )
                    out_probs += next_probs * edge_probs
            calling_stack.pop()

            for mon in out_probs.mons:
                if mon.score == root:
                    out_probs = Pol(
                        mons=tuple(
                            Mon(
                                score=imon.score,
                                coeff=imon.coeff / (1 - mon.coeff),
                            )
                            for imon in out_probs.mons
                            if imon != mon
                        )
                    )

            if len(out_probs.mons) == 1 and out_probs.mons[0].score is None:
                probs[root] = out_probs
            return out_probs

        _ = get_probs(calling_stack=[], root=root)
        for state in probs:
            # Assert that all probs are resolved (float)
            assert (
                len(probs[state].mons) == 1
                and probs[state].mons[0].score is None
            )

        return probs

    def simulate(self, **kwargs: float) -> list[AnyTuple | GameEnd]:
        values = GameDirectedGraph._parse_kwargs_probes(kwargs)
        history: list[AnyTuple | GameEnd] = [self.root]
        state: AnyTuple | GameEnd = self.root
        while not isinstance(state, GameEnd):
            rand = random.random()
            for edge in self.dg[state]:
                rand -= edge.compute_probability(values)
                if rand <= 0:
                    state = self.dg[state][edge]
                    break
            else:
                raise ValueError(
                    "No edge found. The sum of the probabilities should be 1."
                )
            history.append(state)
        return history

    def dump(self) -> dict[str, Any]:
        dg: dict[int, list[tuple[int, str | int]]] = {}
        edges: list[FrozenProbeSet] = []
        states: list[AnyTuple] = list(self.dg.keys())
        for state in self.dg:
            state_index = states.index(state)
            dg[state_index] = []
            for edge in self.dg[state]:
                if edge not in edges:
                    edges.append(edge)
                edge_index: int = edges.index(edge)
                new_state = self.dg[state][edge]
                if isinstance(new_state, tuple):
                    new_state_index = states.index(new_state)
                    dg[state_index].append((edge_index, new_state_index))
                elif new_state == GameEnd.WIN:
                    dg[state_index].append((edge_index, "WIN"))
                elif new_state == GameEnd.LOSE:
                    dg[state_index].append((edge_index, "LOSE"))
        return {
            "states": states,
            "graph": dg,
            "root": str(hash(self.root)),
            "edges": [edge.dump() for edge in edges],
        }


if __name__ == "__main__":

    class DeltaScore(NamedTuple):
        delta: int
        n_points: int

    def play_delta_match(score: DeltaScore, p: Probe) -> DeltaScore | GameEnd:
        if p.run():
            score = DeltaScore(score.delta + 1, score.n_points)
        else:
            score = DeltaScore(score.delta - 1, score.n_points)
        if score.delta == score.n_points:
            return GameEnd.WIN
        elif score.delta == -score.n_points:
            return GameEnd.LOSE
        return score

    delta_match_init = DeltaScore(delta=0, n_points=11)
    delta_match_graph = GameDirectedGraph.from_play_func(
        root=delta_match_init, play_func=play_delta_match
    )
    dmi = delta_match_graph.compute_importance(p=0.501)

    class TieBreakScore(NamedTuple):
        p1: int
        p2: int
        p1_serving: bool

    N = 10

    def play_tie_break(
        score: TieBreakScore, p: Probe, q: Probe
    ) -> TieBreakScore | GameEnd:
        sp = p if score.p1_serving else q
        if sp.run():
            score = TieBreakScore(
                p1=score.p1 + 1, p2=score.p2, p1_serving=score.p1_serving
            )
        else:
            score = TieBreakScore(
                p1=score.p1, p2=score.p2 + 1, p1_serving=score.p1_serving
            )
        if (score.p1 + score.p2) % 2 == 1:
            score = TieBreakScore(
                p1=score.p1, p2=score.p2, p1_serving=not score.p1_serving
            )
        if score.p1 == score.p2 == N - 1:
            score = TieBreakScore(
                p1=N - 2, p2=N - 2, p1_serving=score.p1_serving
            )
        if max(score.p1, score.p2) == N:
            return GameEnd.WIN if score.p1 > score.p2 else GameEnd.LOSE
        return score

    t = time()
    graph = GameDirectedGraph.from_play_func(
        play_tie_break,
        root=TieBreakScore(p1=0, p2=0, p1_serving=True),
    )
    print(time() - t)
    print("XXX")

    def print_dg(graph: GameDirectedGraph) -> str:
        out = ""
        for state in graph.dg:
            out += str(state) + "\n"
            for k in graph.dg[state]:
                out += "    * " + str(graph.dg[state][k]) + "\n"
        return out

    # print(print_dg(dg))
    s = time()
    out = []
    simulation = graph.simulate(p=0.5, q=0.5)

    for p in range(0, 100):
        t = time()
        out.append(
            graph.compute_expected_value(
                p=p / 100,
                q=p / 100,
            )
        )
        print(time() - t)
        print("-----")
    print(out)
