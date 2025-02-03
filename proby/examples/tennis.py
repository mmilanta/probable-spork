from proby.core import GameEnd
from typing import NamedTuple


class GameScore(NamedTuple):
    p1: int
    p2: int
    p1_serving: bool


init_game = GameScore(p1=0, p2=0, p1_serving=True)


def play_game(score: GameScore, p: bool, q: bool) -> GameScore | GameEnd:
    sp = p if score.p1_serving else q
    if sp:
        score = GameScore(
            p1=score.p1 + 1, p2=score.p2, p1_serving=score.p1_serving
        )
    else:
        score = GameScore(
            p1=score.p1, p2=score.p2 + 1, p1_serving=score.p1_serving
        )
    if score.p1 == score.p2 == 3:
        score = GameScore(p1=2, p2=2, p1_serving=score.p1_serving)
    if max(score.p1, score.p2) == 4:
        return GameEnd.WIN if score.p1 > score.p2 else GameEnd.LOSE
    return score


class TieBreakScore(NamedTuple):
    p1: int
    p2: int
    p1_serving: bool
    n_points: int = 7


init_tie_break = TieBreakScore(p1=0, p2=0, p1_serving=True)


def play_tie_break(
    score: TieBreakScore, p: bool, q: bool
) -> TieBreakScore | GameEnd:
    sp = p if score.p1_serving else q
    if sp:
        score = TieBreakScore(
            p1=score.p1 + 1,
            p2=score.p2,
            p1_serving=score.p1_serving,
            n_points=score.n_points,
        )
    else:
        score = TieBreakScore(
            p1=score.p1,
            p2=score.p2 + 1,
            p1_serving=score.p1_serving,
            n_points=score.n_points,
        )
    if (score.p1 + score.p2) % 2 == 1:
        score = TieBreakScore(
            p1=score.p1,
            p2=score.p2,
            p1_serving=not score.p1_serving,
            n_points=score.n_points,
        )
    if score.p1 == score.p2 == score.n_points - 1:
        score = TieBreakScore(
            p1=score.n_points - 2,
            p2=score.n_points - 2,
            p1_serving=score.p1_serving,
            n_points=score.n_points,
        )
    if max(score.p1, score.p2) == score.n_points:
        return GameEnd.WIN if score.p1 > score.p2 else GameEnd.LOSE
    return score


class SetScore(NamedTuple):
    p1: int
    p2: int
    current_game: GameScore
    tie_break: TieBreakScore


def play_set(score: SetScore, p: bool, q: bool) -> SetScore | GameEnd:
    if score.p1 == score.p2 == 6:
        new_set_score = play_tie_break(score=score.tie_break, p=p, q=q)
        if isinstance(new_set_score, GameEnd):
            return new_set_score
        else:
            return SetScore(
                p1=score.p1,
                p2=score.p2,
                current_game=score.current_game,
                tie_break=new_set_score,
            )
    else:
        current_game = play_game(score=score.current_game, p=p, q=q)
        if current_game == GameEnd.WIN:
            score = SetScore(
                p1=score.p1 + 1,
                p2=score.p2,
                current_game=GameScore(
                    p1=0, p2=0, p1_serving=not score.current_game.p1_serving
                ),
                tie_break=score.tie_break,
            )
        elif current_game == GameEnd.LOSE:
            score = SetScore(
                p1=score.p1,
                p2=score.p2 + 1,
                current_game=GameScore(
                    p1=0, p2=0, p1_serving=not score.current_game.p1_serving
                ),
                tie_break=score.tie_break,
            )
        elif isinstance(current_game, GameScore):
            score = SetScore(
                p1=score.p1,
                p2=score.p2,
                current_game=current_game,
                tie_break=score.tie_break,
            )
        if max(score.p1, score.p2) >= 6 and abs(score.p1 - score.p2) > 1:
            return GameEnd.WIN if score.p1 > score.p2 else GameEnd.LOSE
        return score


init_set = SetScore(
    p1=0,
    p2=0,
    current_game=GameScore(p1=0, p2=0, p1_serving=True),
    tie_break=TieBreakScore(p1=0, p2=0, p1_serving=True),
)


class MatchScore(NamedTuple):
    p1: int
    p2: int
    current_set: SetScore


def play_match(score: MatchScore, p: bool, q: bool) -> MatchScore | GameEnd:
    new_set_score = play_set(score=score.current_set, p=p, q=q)
    if new_set_score == GameEnd.WIN:
        score = MatchScore(p1=score.p1 + 1, p2=score.p2, current_set=init_set)
    elif new_set_score == GameEnd.LOSE:
        score = MatchScore(p1=score.p1, p2=score.p2 + 1, current_set=init_set)
    elif isinstance(new_set_score, SetScore):
        score = MatchScore(p1=score.p1, p2=score.p2, current_set=new_set_score)
    if score.p1 == 2:
        return GameEnd.WIN
    if score.p2 == 2:
        return GameEnd.LOSE
    return score


init_match = MatchScore(p1=0, p2=0, current_set=init_set)
