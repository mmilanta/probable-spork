from src.probablespork.tennis import match, game, set, tie_break
from time import time
import math

N = 4
ts = []


def mean(x):
    return sum(x) / len(x)


def std(x):
    return math.sqrt(sum([(t - mean(x)) ** 2 for t in x]) / (len(x) - 1))


for _ in range(N):
    match._tree = None
    game._tree = None
    set._tree = None
    tie_break._tree = None
    start_time = time()
    match.compute_tree()
    ts.append(time() - start_time)

print(match._tree)
print(f"took {mean(ts):.4f}s ({std(ts):.4f})")
