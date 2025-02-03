# Probability of wining in tennis

$$
\newcommand{\w}{\text{Win}}
\newcommand{\l}{\text{Lose}}
\newcommand{\tiebreak}{\text{Tiebreak}}
\newcommand{\M}{\mathcal{M}}
\newcommand{\p}{\mathcal{P}}
\newcommand{\pl}{\text{Player}}
\newcommand{\eff}{\mathcal{E}}
\newcommand{\L}{\mathcal{L}}
\w\l
$$



## Mathematical tools

### Tennis match definition

To study different sorts of tennis match structures, we need a way to rigoursly describe a match. To do so we require two things:

* The starting score.
* A function that takes as input a score, and the output of the current point (boolean); and outputs the updated score. In particular it outputs $\w$ or $\l$, if the played point yielded the end of the match

To make an example. We describe now a best-of-7 tiebreak, without adavatages.
$$
\begin{align}
s_0 &= [0, 0]\\
f_{\tiebreak_7}(s, p) &=
\begin{cases} 
 \w & \text{if } p \text{ and } s[0] = 6, \\
 \l & \text{if not } p \text{ and } s[1] = 6, \\
[s[0] + 1, s[1]] & \text{if } p \text{ and } s[0] < 6, \\
[s[0], s[1] + 1] & \text{if not } p \text{ and } s[1] < 6, \\
\end{cases}
\\
\M_{\tiebreak_7} &= [f_{\tiebreak_7}, s_0]
\end{align}
$$
This is all rather trivial. if $p$ is $\text{True}$, we increase by $1$ the first entry of the vector, otherwise the second. Furhtermore, if the first player reaches $7$, we output $\w$, if the  or $\l$. The python implementation of this would look like:

```python
s0 = (0, 0) # we use tuples cause state needs to be hashable
def f_tiebreak_7(s: tuple, p: bool) -> tuple:
  # updating the state
  if p:
    s = [s[0] + 1, s[1]]
  else:
    s = [s[0], s[1] + 1]
  # returning True or False if match is over, and the next state otherwise
  if s[0] == 6:
    return True
  elif  s[1] == 6:
    return False
  else:
    return s
```

### Game graph computation



### Compute the probability of winning

### Compute the expected length of the match

## Discoveries using the tools

### Match Probability function

Given a match $\M = [f, s_0]$, and $p \in [0, 1]$, we define the Match Probability function
$$
\p(\M, p) := \Pr\left(\pl_1\text{ wins the match} \vert \Pr\left(\pl_1\text{ wins a point}\right) = p\right)
$$
For instance, to come back to our simple example of a 2 point tiebreak, we have:
$$
\p(\M_{\tiebreak_2}, p) = p^2(3-2p)
$$
For a match $\M$ to be sensible, we require two properties:

* $\p(\M, p)$ to be monotone, non decreasing in $p$. This means that if $\pl_1$ is never penalized for playing better tennis
* $\p(\M, p) + \p(\M, 1-p) =1$. This guarantes that $\pl_1$ has no advantage compared to $\pl_2$. 

### Match Efficiency

We define the efficiency of a match $\M$ as
$$
\eff(\M):=\frac{d\p(\M, p)}{dp}\bigg|_{p=0.5}.
$$
Explanation. 

### Match Length

We define $\L(\M, p)$ to be the expected value of the number of points played in a match $\M$, where $\pl_1$ has probability $p$ to win each point.

Many matches can be very short if $p\approx 1$ or $p\approx 0$. But what we care about the most is when $p \approx 0.5$. For this reason, we define $\L(\M):=\L(\M, 0.5)$. 

### Efficiency

I have a conjecture

#### Conjecture

For any match structure $\M$, we have $\eff(\M)^2\leq \L(\M)$.
