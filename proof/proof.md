# Optimal Game Structure Inequality

**Marco Milanta**  
February 15, 2026

## Abstract

We prove the fairness–length inequality for general game graphs.

## Problem Formalization

To state the result rigorously, we need to formally define all the objects. We begin by the game graph.

**Definition (Game graph):** Let $G = (V, E)$ be a directed graph with two absorbing vertices $W$ and $L$. Every other vertex $v$ has two outgoing edges $v\to v_+$ and $v\to v_-$.

Furthermore, we denote a single vertex $s\in V\setminus \{W,L\}$ as starting node.

The game graph is the mathematical structure we will use to describe a tennis match. The vertices are all the possible states of the match: each will correspond to a specific score. From a state $v$, if the first player wins, the game will transition to $v_+$, else to $v_-$. The game being won by the first player corresponds to ending in $W$, loss in $L$.

Now we define the match itself as a random walk on the graph.

**Definition (Graph walk):** Given a game graph $G = (V, E)$, $s \in V$ and $p \in [0,1]$. We define $\mathbf{X}^{(G, s, p)}$ to be a stochastic process on $G$ starting from the starting node $\mathbf{X}^{(G, s, p)}_0 = s$, and with

$$
\mathbf{X}^{(G, s, p)}_{i+1}= \begin{cases}
\left(\mathbf{X}^{(G, s, p)}_i\right)_+ & \text{with prob. } p, \\
\left(\mathbf{X}^{(G, s, p)}_i\right)_- & \text{with prob. } 1 - p,
\end{cases}
$$

while $\mathbf{X}^{(G, s, p)}_i\notin \{W, L\}$, after that the random process ends. We call $\tau^{(G, s, p)} \in \mathbb N \cup\{\infty\}$ the number of steps needed to reach $\{W, L\}$. Finally, if $\tau^{(G, s, p)}<\infty$ we call the final state $O^{(G, s, p)} = \mathbf{X}^{(G, s, p)}_{\tau^{(G, s, p)}} \in \{W, L\}$.

The graph walk is the stochastic process corresponding to playing a match in the structure defined by $G$, and starting from score 0–0, which corresponds to the node $s$. To continue with the parallels, $\tau^{(G, s, p)}$ is the number of points played during the match, and $\mathbf{X}^{(G, s, p)}_{\tau^{(G, s, p)}}$ is the outcome of the match.

Of course, in our analysis we are interested in those games which are balanced, meaning that assuming the first player is just as strong as the second one $(p=1/2)$, then the probability of the first player to win the match is also $1/2$, formally this means that:

$$
\mathbb{P} \left( O_{(G, s, 1/2)} = W\right)= \frac{1}{2}.
$$

The last thing remained to defined is the fairness of the match. This is a quantity that indicates how much the stronger player actually benefits from the game structure.

**Definition (Game Fairness):** Given a game graph $G = (V, E)$, and a starting point $s$, we call the fairness of the match

$$
\mathcal{F}(G, s) := \frac{d}{dp}\mathbb{P} \left( O^{(G, s, p)} = W\right)\Big\vert_{p=\frac{1}{2}}.
$$

While this definition could look esoteric there is a nice way to think about it. Assume the first player has an edge of $\varepsilon$, meaning that $p = 1/2 + \varepsilon$, then we will have that

$$
\mathbb{P}\left(O_{(G, s, 1/2 + \varepsilon)} = W\right) = \frac{1}{2} + \varepsilon\mathcal{F}(G, s) + o(\varepsilon).
$$

Thus, $\mathcal{F}$ can be seen as the advantage multiplicator factor for small $\varepsilon$.

## Main Result

**Theorem:** Given a game graph $G$ and a starting position $s$, such that the game is balanced

$$
\mathbb{P} \left( O^{(G, s, 1/2)} = W\right)= \frac{1}{2},
$$

and the match has no hidden status

$$
\mathbf{X}^{(G, s, p)}_i \vert \mathbf{X}^{(G, s, p)}_{0}, \dots, \mathbf{X}^{(G, s, p)}_{i-1} = \mathbf{X}^{(G, s, p)}_i \vert \mathbf{X}^{(G, s, p)}_{i-1},
$$

then

$$
\mathcal{F}(G, s)\leq \left(\mathbb{E}[\tau_{(G, s, 1/2)}]\right)^2.
$$

This result is tight, as we can see by the optimal game structure defined in the blog post.
The two assumptions we do are also quite natural. The first one is that we are talking about a balanced match structure, where the first player has no inherent advantage over the second one. The second one is more subtle, and likely it does not hold in practice; basically we are assuming that the players will play each point independently of how the previous points in the match went. This assumption is however necessary; without it the result does not hold.

## Proof

The proof is divided into three parts. First we do a perturbation argument to derive a powerful formula to express the fairness $\mathcal{F}(G, s)$, then we do a variance identity, and finally we use the Cauchy–Schwarz inequality to derive the final bound. We first start by defining the key quantities we will use.

### Definitions

**Definition (Value function):**

$$
h_{(G)}: V \to [0,1], \quad\quad h_{(G)}(v) = \mathbb{P}\left( O^{(G, v, 1/2)} = W\right).
$$

Thus, $h_{(G)}(v)$ is the probability for the first player to win, given that the first player is just as strong as the second one $(p= 1/2)$, and given that the current score is $v$.
Note that $h_{(G)}(v)$ is not a random variable, and it always assumes that $p = 1/2$.

This quantity is a good proxy for the state of the match. Instead of a messy score, which could be misleading, this indicates well how far (probabilistically) is a player from winning the match.

We now show that $h_{(G)}(v)$ has a key harmonic property, that later on will be crucial.

**Lemma ($h_{(G)}(v)$ is harmonic):** Given a vertex $v\in G\setminus\{W, L\}$, let $v_+$ and $v_-$ be the vertices reached by the outward edges from $v$, then

$$
h_{(G)}(v) = \frac{1}{2}\left(h_{(G)}(v_+) + h_{(G)}(v_-) \right).
$$

**Proof:** The proof comes from rolling out one step of the random walk.

$$
\begin{align*}
    h_{(G)}(v) &= \mathbb{P}\left( O^{(G, v, 1/2)} = W\right) \\
    &\overset{(1)}{=} \mathbb{P}\left( O^{(G, v, 1/2)} = W\vert \mathbf{X}^{(G, v, 1/2)}_1 = v_+\right)\mathbb{P}\left(  \mathbf{X}^{(G, v, 1/2)}_1 = v_+\right) + \\ &\quad\quad\mathbb{P}\left( O^{(G, v, 1/2)} = W\vert \mathbf{X}^{(G, v, 1/2)}_1 = v_-\right)\mathbb{P}\left(  \mathbf{X}^{(G, v, 1/2)}_1 = v_-\right) \\
    &\overset{(2)}{=}h_{(G)}(v_+)\mathbb{P}\left(  \mathbf{X}^{(G, v, 1/2)}_1 = v_+\right) +  h_{(G)}(v_-)\mathbb{P}\left(\mathbf{X}^{(G, v, 1/2)}_1 = v_-\right) \\
    & \overset{(3)}{=} \frac{1}{2} h_{(G)}(v_+) + \frac{1}{2}h_{(G)}(v_-),
\end{align*}
$$

where $(1)$ is due to the law of total probability, dividing the probability space between the two possible outcomes of the first step. $(2)$ follows from the definition of $h_{(G)}(v)$. $(3)$ follows from the fact that we are looking at $\mathbf{X}^{(G, v, 1/2)}$, thus there is a $1/2$ probability of winning the first step. $\square$

Now that we defined the value function $h_{(G)}(v)$, we introduce one more quantity

**Definition (Importance function):**

$$
\Delta_G(v) := h_{(G)}(v_+) - h_{(G)}(v_-).
$$

This function indicates how much probability is at stake in a given situation $v$. If the game is balanced, if the first player wins the next point, then his probability of winning will go up by $\Delta_G(v)/2$. This quantity is a good proxy to identify pressure points as match points in a game.

Now that we defined those non-aleatory quantities, we can consider the problem with a probability distribution. Thus we define the following random process.

**Definition (Stochastic Value Process):** We define $M^{(G, s, p)}$ as follows:

$$
M^{(G, s, p)} := \left(h\left(\mathbf{X}^{(G, s, p)}_0\right), h\left(\mathbf{X}^{(G, s, p)}_1\right), \dots, h\left(\mathbf{X}^{(G, s, p)}_{\tau^{(G, s, p)}}\right)\right).
$$

The name $M^{(G, s, p)}$ is motivated by the fact that, in the specific case of $p=1/2$, $M^{(G, s, 1/2)}$ is a martingale.

To motivate these constructions, we notice that

$$
\mathbb{P}\left( O^{(G, s, p)} = W\right) = \mathbb{E}\left[M^{(G, s, p)}_{\tau^{(G, s, p)}}\right].
$$

The reason is that $\mathbf{X}^{(G, s, p)}_{\tau^{(G, s, p)}}$ is either $W$ or $L$, thus $M^{(G, s, p)}_{\tau^{(G, s, p)}}$ is the either $1$ or $0$, thus $ \mathbb{E}\left[M^{(G, s, p)}_{\tau^{(G, s, p)}}\right]$ is the probability that $M^{(G, s, p)}_{\tau^{(G, s, p)}}$ is $1$.

Furthermore, because the match is balanced (one of the conditions we assumed),

$$
M^{(G, s, p)}_{0} = h\left(\mathbf{X}^{(G, s, p)}_0\right) = 1/2.
$$

Thus, $M^{(G, s, p)}$ is a stocastic process which starts at $1/2$ and ends at $1$ or $0$, but even more interestingly, the expected value of $M^{(G, s, p)}$ starts from $1/2$ and ends at the probability of winning the match.

### Perturbation argument

As we discussed before, $M^{(G, s, 1/2)}$ is a martingale, thus

$$
\mathbb{E}\left[M^{(G, s, p)}_i\right] = 1/2 \quad \forall i.
$$

We now examin what happens if we perturb the $p$ slightly from $1/2$. So we compute the probability of winning with $p = 1/2 + \varepsilon$.

$$
\begin{align*}
    \mathbb{P}\left(O^{(G, s, 1/2 + \varepsilon)} = W\right) &= \mathbb{E}\left[M^{(G, s, 1/2 + \varepsilon)}_{\tau^{(G, s, 1/2 + \varepsilon)}}\right]\\
    &= \mathbb{E}\left[\underbrace{M^{(G, s, 1/2 + \varepsilon)}_0}_{=1/2} + \sum_{i=0}^{\tau^{(G, s, 1/2 + \varepsilon)}} \left(M^{(G, s, 1/2 + \varepsilon)}_i - M^{(G, s, 1/2 + \varepsilon)}_{i-1}\right)\right] \\
    &= \frac{1}{2} + \mathbb{E}\left[\sum_{i=0}^{\infty} 1_{i \leq \tau^{(G, s, 1/2 + \varepsilon)}}\left(M^{(G, s, 1/2 + \varepsilon)}_i - M^{(G, s, 1/2 + \varepsilon)}_{i-1}\right)\right]\\
    &\overset{(1)}{=} \frac{1}{2} +\sum_{i=0}^{\infty} \mathbb{E}\left[1_{i \leq \tau^{(G, s, 1/2 + \varepsilon)}}\left(M^{(G, s, 1/2 + \varepsilon)}_i - M^{(G, s, 1/2 + \varepsilon)}_{i-1}\right)\right]\\
    &\overset{(2)}{=} \frac{1}{2} +\sum_{i=0}^{\infty} \mathbb{E}\left[M^{(G, s, 1/2 + \varepsilon)}_i - M^{(G, s, 1/2 + \varepsilon)}_{i-1}\right]\\
    &\overset{(3)}{=} \frac{1}{2} +\sum_{i=0}^{\infty} \mathbb{E}\left[\mathbb{E}\left[M^{(G, s, 1/2 + \varepsilon)}_i - M^{(G, s, 1/2 + \varepsilon)}_{i-1}\vert \mathbf{X}^{(G, s, 1/2 + \varepsilon)}_{i-1}\right]\right]\\
    &\overset{(4)}{=} \frac{1}{2} +\varepsilon \sum_{i=0}^{\infty} \mathbb{E}\left[\Delta_G\left(\mathbf{X}^{(G, s, 1/2 + \varepsilon)}_{i-1}\right)\right]\\
    &\overset{(5)}{=} \frac{1}{2} +\varepsilon \; \mathbb{E}\left[\sum_{i=0}^{\tau^{(G, s, 1/2 + \varepsilon)}} \Delta_G\left(\mathbf{X}^{(G, s, 1/2 + \varepsilon)}_{i-1}\right)\right].
\end{align*}
$$

In $(1)$ we use the fact that the quantity we are trying to bound is a probability, thus is between $0$ and $1$, and thus we can swap the sum and the expectation.
In $(2)$ we use a trick: we momentarily define $M^{(G, s, 1/2 + \varepsilon)}_i = M^{(G, s, 1/2 + \varepsilon)}_{\tau^{(G, s, 1/2 + \varepsilon)}}$ for $i > \tau^{(G, s, 1/2 + \varepsilon)}$, notice that this does not alter the expected value because for $i > \tau^{(G, s, 1/2 + \varepsilon)}$ the quantity $M^{(G, s, 1/2 + \varepsilon)}_i - M^{(G, s, 1/2 + \varepsilon)}_{i-1}=0$. In $(3)$ we use the law of total expectation. Finally, $(4)$ is justified right after.

$$
\begin{align*}
& \mathbb{E}\left[\left(h\left(\mathbf{X}^{(G, s, 1/2 + \varepsilon)}_i\right) - h\left(\mathbf{X}^{(G, s, 1/2 + \varepsilon)}_{i-1}\right)\right)\vert \mathbf{X}^{(G, s, 1/2 + \varepsilon)}_{i-1}\right] = \\
&= \underbrace{\frac{1}{2}\left(h\left(\left(\mathbf{X}^{(G, s, 1/2 + \varepsilon)}_{i-1}\right)_+\right) + h\left(\left(\mathbf{X}^{(G, s, 1/2 + \varepsilon)}_{i-1}\right)_-\right)\right) - h\left(\mathbf{X}^{(G, s, 1/2 + \varepsilon)}_{i-1}\right)}_{= 0 \text{ because } h \text{ is harmonic}} + \\
& \quad\quad + \varepsilon \underbrace{\left(h\left(\left(\mathbf{X}^{(G, s, 1/2 + \varepsilon)}_{i-1}\right)_+\right) - h\left(\left(\mathbf{X}^{(G, s, 1/2 + \varepsilon)}_{i-1}\right)_-\right)\right)}_{:= \Delta_G\left(\mathbf{X}^{(G, s, 1/2 + \varepsilon)}_{i-1}\right)} \\
&= \varepsilon \Delta_G\left(\mathbf{X}^{(G, s, 1/2 + \varepsilon)}_{i-1}\right).
\end{align*}
$$

Finally we can compute the derivative of the probability of winning as the limit of the perturbation argument.

$$
\begin{align*}
    \frac{d}{dp} \mathbb{P}\left(O^{(G, s, p)} = W\right)  \Big\vert_{p=1/2} &= \lim_{\varepsilon \to 0} \frac{\mathbb{P}\left(O^{(G, s, 1/2 + \varepsilon)} = W\right) - \mathbb{P}\left(O^{(G, s, 1/2)} = W\right)}{\varepsilon} \\
    &= \lim_{\varepsilon \to 0} \frac{\frac{1}{2} + \varepsilon \mathbb{E}\left[\sum_{i=0}^{\tau^{(G, s, 1/2 + \varepsilon)}} \Delta_G\left(\mathbf{X}^{(G, s, 1/2 + \varepsilon)}_{i-1}\right)\right] - \frac{1}{2}}{\varepsilon} \\
    &= \lim_{\varepsilon \to 0} \mathbb{E}\left[\sum_{i=0}^{\tau^{(G, s, 1/2 + \varepsilon)}} \Delta_G\left(\mathbf{X}^{(G, s, 1/2 + \varepsilon)}_{i-1}\right)\right] \\
    &= \mathbb{E}\left[\sum_{i=0}^{\tau^{(G, s, 1/2)}} \Delta_G\left(\mathbf{X}^{(G, s, 1/2)}_{i-1}\right)\right].
\end{align*}
$$

Thus, finally we have

$$
\mathcal{F}(G, s) = \mathbb{E}\left[\sum_{i=0}^{\infty} \Delta_G\left(\mathbf{X}^{(G, s, 1/2)}_{i-1}\right)\right]. \tag{1}
$$

This concludes the perturbation argument.

### Variance identity

In the perturbation argument we have focused on the expected value of $M^{(G, s, p)}$ over the course of the match. Now we focus on the variance instead. Similar to the perturbation argument, we will use a telescoping argument to express the variance of $M^{(G, s, 1/2)}$ over the course of the match, together with a key insight on the variance of the $M^{(G, s, 1/2)}_{\tau^{(G, s, 1/2)}}$ function.

The match is balanced, thus we know that the probability of winning the match is $1/2$, thus $M^{(G, s, 1/2)}_{\tau^{(G, s, 1/2)}} \sim \text{Bernoulli}(1/2)$. Thus

$$
\mathbb{V}\left(M^{(G, s, 1/2)}_{\tau^{(G, s, 1/2)}}\right) = \mathbb{V}\left(\text{Bernoulli}(1/2)\right) =  \frac{1}{4}. \tag{2}
$$

Now we can use the telescoping argument to express the variance of $M^{(G, s, 1/2)}$ over the course of the match. Let $b_i$ be a random variable that is $1$ if at step $i$ the process goes on the $+$ edge, and $-1$ if it goes on the $-$ edge. Note that $b_i\overset{\text{i.i.d.}}{\sim} \text{Rademacher}(1/2)$ (i.e., $\mathbb{P}(b_i = 1) = \mathbb{P}(b_i = -1) = 1/2$), and the independence follows from the second assumption we made. Then:

$$
\begin{align*}
    \mathbb{V} \left(M^{(G, s, 1/2)}_{\tau^{(G, s, 1/2)}}\right) &= \mathbb{V}\left(M^{(G, s, 1/2)}_0 + \sum_{i=1}^{\tau^{(G, s, 1/2)}} \left(M^{(G, s, 1/2)}_i - M^{(G, s, 1/2)}_{i-1}\right)\right) \\
    &\overset{(1)}{=}\mathbb{V}\left(\sum_{i=1}^{\tau^{(G, s, 1/2)}} \left(M^{(G, s, 1/2)}_i - M^{(G, s, 1/2)}_{i-1}\right)\right)
\end{align*}
$$

Where $(1)$ is due to the fact that $M^{(G, s, 1/2)}_0 = 1/2$, and translations do not contribute to the variance. Now we focus on $M^{(G, s, 1/2)}_i - M^{(G, s, 1/2)}_{i-1}$.

$$
\begin{align*}
    M^{(G, s, 1/2)}_i - M^{(G, s, 1/2)}_{i-1} &= h\left(\mathbf{X}^{(G, s, 1/2)}_i\right) - h\left(\mathbf{X}^{(G, s, 1/2)}_{i - 1}\right) \\
    &= \begin{cases}
    h\left(\left(\mathbf{X}^{(G, s, 1/2)}_{i - 1}\right)_+\right) - h\left(\mathbf{X}^{(G, s, 1/2)}_{i - 1}\right) & \text{if win}\\
    h\left(\left(\mathbf{X}^{(G, s, 1/2)}_{i - 1}\right)_-\right) - h\left(\mathbf{X}^{(G, s, 1/2)}_{i - 1}\right) & \text{if lose}
    \end{cases}\\
    &= \begin{cases}
    h\left(\left(\mathbf{X}^{(G, s, 1/2)}_{i - 1}\right)_+\right) - \frac{1}{2}\left(h\left(\left(\mathbf{X}^{(G, s, 1/2)}_{i - 1}\right)_+\right) + h\left(\left(\mathbf{X}^{(G, s, 1/2)}_{i - 1}\right)_-\right)\right) & \text{if win}\\
    h\left(\left(\mathbf{X}^{(G, s, 1/2)}_{i - 1}\right)_-\right) - \frac{1}{2}\left(h\left(\left(\mathbf{X}^{(G, s, 1/2)}_{i - 1}\right)_+\right) + h\left(\left(\mathbf{X}^{(G, s, 1/2)}_{i - 1}\right)_-\right)\right)  & \text{if lose}
    \end{cases}\\
    &= \begin{cases}
    \frac{1}{2} \Delta_G\left(\mathbf{X}^{(G, s, 1/2)}_{i - 1}\right) & \text{if win}\\
    \frac{1}{2} -\Delta_G\left(\mathbf{X}^{(G, s, 1/2)}_{i - 1}\right)  & \text{if lose}
    \end{cases}\\
    &= \frac{1}{2} \Delta_G\left(\mathbf{X}^{(G, s, 1/2)}_{i - 1}\right)b_i,
\end{align*}
$$

Where $b_i = 1$ if the first player manages to claim the $i$-th point, otherwise $-1$. Because the match is balanced, $b_i \sim \text{Rademacher}(1/2)$. Furgthermore, notice that $b_i \perp b_j\;\forall i \neq j$ due to the the second assumption of the theorem.

$$
\begin{align*}
    \mathbb{V} \left(M^{(G, s, 1/2)}_{\tau^{(G, s, 1/2)}}\right) &= \mathbb{V}\left(M^{(G, s, 1/2)}_0 + \sum_{i=1}^{\tau^{(G, s, 1/2)}} \left(M^{(G, s, 1/2)}_i - M^{(G, s, 1/2)}_{i-1}\right)\right) \\
    &=\mathbb{V}\left(\sum_{i=1}^{\tau^{(G, s, 1/2)}}\frac{1}{2} \Delta_G\left(\mathbf{X}^{(G, s, 1/2)}_{i-1}\right)  b_{i-1} \right)\\
    &=\frac{1}{4} \sum_{i=1}^{\tau^{(G, s, 1/2)}} \sum_{j=1}^{\tau^{(G, s, 1/2)}}\text{Cov}\left(\Delta_G\left(\mathbf{X}^{(G, s, 1/2)}_{i-1}\right)  b_{i-1}, \Delta_G\left(\mathbf{X}^{(G, s, 1/2)}_{j-1}\right)  b_{j-1} \right)\\
    &\overset{(1)}{=}\frac{1}{4} \sum_{i=1}^{\tau^{(G, s, 1/2)}} \mathbb{V}\left(\Delta_G\left(\mathbf{X}^{(G, s, 1/2)}_{i-1}\right) b_{i-1}\right)\\
    &\overset{(2)}{=}\frac{1}{4} \sum_{i=1}^{\tau^{(G, s, 1/2)}} \mathbb{E}\left[\Delta_G\left(\mathbf{X}^{(G, s, 1/2)}_{i-1}\right)^2\right]
\end{align*}
$$

In $(1)$ we use the aformentioned indipendence to cancel out all the cross-terms. In $(2)$ we use the fact $\mathbb{E}[b_i] = 0$, thus the variance is the expected value of the square.

Combining this with (2) we get

$$
\sum_{i=1}^{\tau^{(G, s, 1/2)}} \mathbb{E}\left[\Delta_G\left(\mathbf{X}^{(G, s, 1/2)}_{i-1}\right)^2\right] = 1 \tag{3}
$$

This concludes the variance identity.

### Cauchy–Schwarz inequality

The last step to complete the proof is to use the Cauchy–Schwarz inequality:

$$
\begin{align*}
    \mathcal{F}(G, s)^2 &\overset{(1)}{=} \mathbb{E}\left[\sum_{i=0}^{\tau^{(G, s, 1/2)}} \Delta_G\left(\mathbf{X}^{(G, s, 1/2)}_{i-1}\right)\right]^2 \\
    &= \mathbb{E}\left[\sum_{i=0}^{\tau^{(G, s, 1/2)}} 1 \;\Delta_G\left(\mathbf{X}^{(G, s, 1/2)}_{i-1}\right)\right]^2 \\
    &\overset{(2)}{\leq}\underbrace{\mathbb{E}\left[\sum_{i=0}^{\tau^{(G, s, 1/2)}} \Delta_G\left(\mathbf{X}^{(G, s, 1/2)}_{i-1}\right)^2\right]}_{= 1 \text{ due to variance identity}} \mathbb{E}\left[\sum_{i=0}^{\tau^{(G, s, 1/2)}} 1\right]\\
    &= \mathbb{E}\left[\sum_{i=0}^{\tau^{(G, s, 1/2)}} 1\right] = \mathbb{E}\left[\tau^{(G, s, 1/2)}\right].
\end{align*}
$$

Where $(1)$ is due to equation (1) in the perturbation argument, and $(2)$ to the Cauchy–Schwarz inequality for sum of random variables. This concludes the proof. $\square$
