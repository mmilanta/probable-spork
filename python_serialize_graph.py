from enum import Enum
from typing import NamedTuple, Any, Iterator, Callable, Literal
from copy import deepcopy
from dataclasses import dataclass
from inspect import signature


@dataclass(frozen=True, eq=True)
class Edge:
    future: tuple[bool, ...]
    target_state: int


@dataclass(frozen=True, eq=True)
class Node:
    state: int
    edges: tuple[Edge, ...]


@dataclass(frozen=True, eq=True)
class GameGraph:
    nodes: list[Node]

    def serialize(
        self,
        byte_len: int = 32,
        byteorder: Literal["little", "big"] = "little",
    ) -> bytes:
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
        output = b""
        output += k.to_bytes(byte_len, byteorder=byteorder, signed=False)
        output += len(self.nodes).to_bytes(
            byte_len, byteorder=byteorder, signed=False
        )
        for node in self.nodes:
            for edge in node.edges:
                output += edge.target_state.to_bytes(
                    byte_len, byteorder=byteorder, signed=False
                )
        return output


class GameEnd(Enum):
    WIN = 1
    LOSE = 2


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


class Delta(NamedTuple):
    delta: int
    tot: int = 7


def play_match_delta(s: Delta, p: bool) -> Delta | GameEnd:
    s = (
        Delta(delta=s.delta + 1, tot=s.tot)
        if p
        else Delta(delta=s.delta - 1, tot=s.tot)
    )
    if s.delta == s.tot:
        return GameEnd.WIN
    elif s.delta == -s.tot:
        return GameEnd.LOSE
    return s


def generate_future(n_params: int) -> Iterator[tuple[bool, ...]]:
    for i in range(2**n_params):
        yield tuple(bool(int(x)) for x in bin(i)[2:].zfill(n_params))


def compute_graph(
    playing_function: Callable[..., Any | GameEnd],
    root: Any,
) -> tuple[list[Any], GameGraph]:
    n_params: int = len(signature(playing_function).parameters) - 1
    states: list[Any] = [GameEnd.WIN, GameEnd.LOSE]
    nodes: list[Node] = [Node(state=0, edges=()), Node(state=1, edges=())]

    def _dfs(score: Any) -> None:
        states.append(score)
        score_idx = len(states) - 1
        nodes.append(Node(state=score_idx, edges=()))
        edges: list[Edge] = []
        for probe in generate_future(n_params):
            n_score = playing_function(deepcopy(score), *probe)
            if n_score not in states and not isinstance(n_score, GameEnd):
                _dfs(n_score)
            n_score_idx = states.index(n_score)
            edges.append(Edge(future=probe, target_state=n_score_idx))
        nodes[score_idx] = Node(state=score_idx, edges=tuple(edges))

    _dfs(root)
    return states, GameGraph(nodes=nodes)


match_init = Score(p1=0, p2=0, tot=50)
states, graph = compute_graph(
    play_match_delta,
    Delta(delta=0, tot=11),
)
serialized_graph = graph.serialize(
    byte_len=4,
)
with open("serialized_graph.bin", "wb") as f:
    f.write(serialized_graph)
print(serialized_graph)
for i, state in enumerate(states):
    print(f"{i}: {state}")
int.from_bytes(serialized_graph[:4], byteorder="big", signed=False)
