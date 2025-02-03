import probycapi
from enum import Enum
from typing import Any, Iterator, Callable, Literal, NamedTuple
from dataclasses import dataclass
from inspect import signature
import itertools
from concurrent.futures import ThreadPoolExecutor
import random


@dataclass(frozen=True, eq=True)
class Edge:
    future: tuple[bool, ...]
    target_state: int


@dataclass(frozen=True, eq=True)
class Node:
    state: int
    edges: tuple[Edge, ...]


class GameEnd(Enum):
    WIN = 1
    LOSE = 2


def generate_future(n_params: int) -> Iterator[tuple[bool, ...]]:
    if n_params == 1:
        yield (False,)
        yield (True,)
    else:
        for future in generate_future(n_params - 1):
            yield (False,) + future
            yield (True,) + future


@dataclass(frozen=True, eq=True)
class GameGraph:
    nodes: list[Node]
    states: list[Any]
    playing_function: Callable[..., Any | GameEnd]
    root: Any

    def _serialize_int(self) -> list[int]:
        output: list[int] = []
        n_edges: int = -1
        for node in self.nodes:
            if n_edges < 1:
                n_edges = len(node.edges)
            else:
                assert n_edges == len(node.edges)

        for node in self.nodes:
            for edge in node.edges:
                assert 2 ** len(edge.future) == n_edges
                k: int = len(edge.future)
        output = []
        output.append(k)
        output.append(len(self.nodes))
        for node in self.nodes:
            for edge in node.edges:
                output.append(edge.target_state)
        return output

    def serialize(
        self,
        byte_len: int = 4,
        byteorder: Literal["little", "big"] = "little",
    ) -> bytes:
        serial_ints = self._serialize_int()
        return b"".join(
            v.to_bytes(length=byte_len, byteorder=byteorder)
            for v in serial_ints
        )

    def _parse_kwargs_probes(
        self,
        kwargs: dict[str, float] | dict[str, list[float]],
        flip: bool = False,
    ) -> list[list[float]]:
        parameters: list[str] = list(
            signature(self.playing_function).parameters
        )[1:]
        vals: list[list[float] | None] = [None] * len(
            parameters
        )  # -1 because the first parameter is the score
        for k, val in kwargs.items():
            assert isinstance(k, str)
            try:
                param_index = parameters.index(k)
            except ValueError:
                raise ValueError(
                    f"Probe {k} is not a valid parameter for the playing function. Found {parameters}"
                )
            try:
                if isinstance(val, list):
                    val = [float(v) for v in val]
                else:
                    val = [float(val)]
            except (TypeError, ValueError):
                raise ValueError(
                    f"Probe {k} value must be a float, or a list of float. Found {val}"
                )
            for v in val:
                if v > 1 or v < 0:
                    raise ValueError(
                        f"Probe {k} value must be between 0 and 1, found {v}."
                    )
            vals[param_index] = val
        assert all(
            val is not None for val in vals
        ), "All probes must be defined."

        # Tell mypy that vals now only contains lists of floats
        vals_filtered = [v for v in vals if v is not None]
        # mypy now understands that `val` is `list[float]` in this block
        size = len(vals_filtered[0])
        for val in vals_filtered:
            if len(val) != size:
                raise ValueError("All probes must have the same length.")
        if flip:
            vals_filtered = vals_filtered[::-1]
        return vals_filtered

    def probability(self, index: int = 2, **kwargs: float) -> float:
        values = self._parse_kwargs_probes(kwargs, flip=True)
        for value in values:
            assert len(value) == 1
        out = probycapi.probability(
            self.serialize(), [v[0] for v in values], index
        )
        assert isinstance(out, float)
        return out

    def batch_probability(self, **kwargs: list[float]) -> list[float]:
        values = self._parse_kwargs_probes(kwargs, flip=True)
        parameter_genarator = map(list, zip(*values))
        self_serialized = itertools.repeat(self.serialize())
        index_iter = itertools.repeat(2)
        with ThreadPoolExecutor(max_workers=10) as executor:
            response = executor.map(
                probycapi.probability,
                self_serialized,
                parameter_genarator,
                index_iter,
            )
        return list(response)

    def edge_probabilities(self, ps: list[float]) -> list[float]:
        edge_ps: list[float] = []
        for future in generate_future(len(ps)):
            edge_p: float = 1.0
            for p, future_p in zip(ps, future):
                edge_p *= p if future_p else 1.0 - p
            edge_ps.append(edge_p)
        return edge_ps

    def montecarlo(self, **kwargs: float) -> bool:
        values = self._parse_kwargs_probes(kwargs)
        for value in values:
            assert len(value) == 1
        edge_probs = self.edge_probabilities([v[0] for v in values])
        current = 2
        while current > 1:
            next_edge = random.choices(
                self.nodes[current].edges, edge_probs, k=1
            )[0]
            current = next_edge.target_state
        return current == 0

    def expected_length(self, index: int = 2, **kwargs: float) -> float:
        values = self._parse_kwargs_probes(kwargs)
        for value in values:
            assert len(value) == 1

        out = probycapi.expected_length(
            self.serialize(), [v[0] for v in values], index
        )
        assert isinstance(out, float)
        return out

    def batch_expected_length(self, **kwargs: list[float]) -> list[float]:
        values = self._parse_kwargs_probes(kwargs)
        parameter_genarator = map(list, zip(*values))
        self_serialized = itertools.repeat(self.serialize())
        index_iter = itertools.repeat(2)
        with ThreadPoolExecutor(max_workers=10) as executor:
            response = executor.map(
                probycapi.expected_length,
                self_serialized,
                parameter_genarator,
                index_iter,
            )
        return list(response)

    def importance(self, **kwargs: float) -> dict[Any, float]:
        values = self._parse_kwargs_probes(kwargs)
        for value in values:
            assert len(value) == 1
        self_serialized = itertools.repeat(self.serialize())
        parsed_values = list(itertools.repeat([v[0] for v in values]))

        with ThreadPoolExecutor(max_workers=10) as executor:
            response = executor.map(
                probycapi.probability,
                self_serialized,
                parsed_values,
                range(2, len(self.states)),
            )
        probs = [1.0, 0.0] + list(response)
        importance: dict[Any, float] = {}
        for i, state in enumerate(self.states):
            if isinstance(state, GameEnd):
                continue
            max_prob = 0.0
            min_prob = 1.0
            for edge in self.nodes[i].edges:
                max_prob = max(max_prob, probs[edge.target_state])
                min_prob = min(min_prob, probs[edge.target_state])
            importance[state] = max_prob - min_prob
        return importance

    @classmethod
    def compute_graph(
        cls,
        playing_function: Callable[..., Any | GameEnd],
        root: Any,
    ) -> "GameGraph":
        n_params: int = len(signature(playing_function).parameters) - 1
        states: dict[Any, int] = {GameEnd.WIN: 0, GameEnd.LOSE: 1}
        nodes: list[Node] = [Node(state=0, edges=()), Node(state=1, edges=())]

        def _dfs(score: Any) -> None:
            states[score] = len(states)
            score_idx = len(states) - 1
            nodes.append(Node(state=score_idx, edges=()))
            edges: list[Edge] = []
            for probe in generate_future(n_params):
                n_score = playing_function(score, *probe)
                if n_score not in states and not isinstance(n_score, GameEnd):
                    _dfs(n_score)
                edges.append(Edge(future=probe, target_state=states[n_score]))
            nodes[score_idx] = Node(state=score_idx, edges=tuple(edges))

        _dfs(root)

        states_list = list(states.keys())
        for k in range(len(states_list)):
            states[states_list[k]] = k

        return GameGraph(
            nodes=nodes,
            states=states_list,
            playing_function=playing_function,
            root=root,
        )


if __name__ == "__main__":

    class Score(NamedTuple):
        p1: int
        p2: int
        serving: bool = True
        tot: int = 7

    def play_match(score: Score, p: bool) -> Score | GameEnd:
        if p:
            score = Score(p1=score.p1 + 1, p2=score.p2, tot=score.tot)
        else:
            score = Score(p1=score.p1, p2=score.p2 + 1, tot=score.tot)
        if score.p1 == score.tot:
            return GameEnd.WIN
        elif score.p2 == score.tot:
            return GameEnd.LOSE
        return score

    graph = GameGraph.compute_graph(
        play_match,
        Score(p1=0, p2=0, tot=11),
    )
    graph.probability(p=0.5)
    serialized_graph = graph.serialize(
        byte_len=4,
    )
    with open("serialized_graph.bin", "wb") as f:
        f.write(serialized_graph)
    print(serialized_graph)
    for i, state in enumerate(graph.states):
        print(f"{i}: {state}")
    int.from_bytes(serialized_graph[:4], byteorder="big", signed=False)
