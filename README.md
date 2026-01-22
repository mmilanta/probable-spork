# proby
A python library to compute enhancing winning probability of matches composed of many smaller points. I.e. Tennis, Ping Pong, Volleybal, ...

To be rigorous, we solve the following task:
> Assume `Player1` has a probability `p` of winning a point against `Player2`. What is the probability of `Player1` winning the full match?
## Using the library
### Defining a match
The first challenge is to write down in `python` a description of the match structure. To describe a match, we need:
1. A variable `s0`, which indicates the initial state of the match when.
1. A function `play_fn`, which maps a state and a boolean, to another state or `GameEnd.WIN` or `GameEnd.LOSE`.
Let's make an example with a simple best of 7 tiebreaker, without advantages.
```python
from proby import GameEnd
s0 = (0, 0)
def play_fn(state, next_point: bool):
    if next_point:
        state = (state[0] + 1, state[1])
    else:
        state = (state[0], state[1] + 1)
    if state[0] == 7:
        return GameEnd.WIN
    elif state[0] == 7:
        return GameEnd.WIN
    else:
        return state
```
So, the `play_fn` describes how the match state changes, depending on the output of the `next_point`. By convention `next_point = True` means that `Player1` won the point.

We could use `play_fn` and `s0` to simulate a match as follows:
```python
import random
s = s0

while not isinstance(s, GameEnd):
    next_point = random.rand() < p: # where p is the probabiliy of `Player1` winning a point
    s = play_fn(s, next_point)
print(s)
```
### Compute probability
We could simply approximate the probability that `Player1` wins the match by running the script above many times, and counting the ratio of times `Player1` wins. However, this is only an approximation. `proby` offers an exact computation, and it is very fast, as the algorithm is written in `c`. To use it, simply:
```python
from proby import GameGraph
graph = GameGraph.compute_graph(playing_function=play_fn, root=s0)
print(graph.probability(p=0.3))
```
You can parallelize it for multiple values of `p` by using `batch_probability` instead `probability`.

### Compute expected length
This library supports also the computation of the expected number of points of a match, given `p`. Use `expected_length` and `batch_expected_length`.

## Compile to wasm
```
emcc proby/probycapi/algo.c \
    -o algo.js \
    -O3 \
    -s WASM=1 \
    -s EXPORTED_FUNCTIONS='["_prob", "_explen", "_malloc", "_free"]' \
    -s EXPORTED_RUNTIME_METHODS='["ccall", "cwrap"]' \
    -s ALLOW_MEMORY_GROWTH=1 
```
