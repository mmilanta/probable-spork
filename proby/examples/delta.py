from proby.core import GameEnd
from typing import NamedTuple


class DeltaScore(NamedTuple):
    s: int
    delta: int


def play_delta(score: DeltaScore, p: bool) -> GameEnd | DeltaScore:
    if p:
        score = DeltaScore(s=score.s + 1, delta=score.delta)
    else:
        score = DeltaScore(s=score.s - 1, delta=score.delta)
    if score.s == score.delta:
        return GameEnd.WIN
    if score.s == -score.delta:
        return GameEnd.LOSE
    return score
