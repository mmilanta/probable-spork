# Probability of wining in tennis

## Mathematical tools

### Tennis match definition

To study different sorts of tennis match structures, we need a way to rigoursly describe a match. To do so we require two things:

* An object describing the score. In particular we need $s_0$, the starting score.
* An function that maps a score, and the output of the current point, to the next score.

To make an example. We describe now a best-of-7 tiebreak, without adavatages. In this case we have $s_0 = [0, 0]$, and
$$
f_{\text{Tiebreak}(7)}(s, p) =
\begin{cases} 
 \text{Win } & \text{if } p \text{ and } s[0] = 6, \\
 \text{Lose } & \text{if not } p \text{ and } s[1] = 6, \\
[s[0] + 1, s[1]] & \text{if } p \text{ and } s[0] < 6, \\
[s[0], s[1] + 1] & \text{if not } p \text{ and } s[1] < 6, \\
\end{cases}
$$
This is all rather trivial. if $p$ is $\text{True}$, we increase the first entry of the vector, otherwise we increase the second. Furhtermore, if one of the two player reaches $7$, we output $\text{Win}$ or $\text{Lose}$.

### Game graph computation

### Compute the probability of winning

### Compute the expected length of the match

## Discoveries using the tools



