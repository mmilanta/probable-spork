from proby.tennis import set, game, match, core_point
from proby import Point
import pytest


@pytest.mark.parametrize(
    "p", [set, game, match, core_point,],
)
def test_tennis_point(p: Point):
    p.compute_tree()
    p.probability(base_point={core_point: .6})
