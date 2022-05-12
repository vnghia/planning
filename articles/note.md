# Introduction

In standard learning problems, full information is obtained on the state of the system, as such, allowing algorithms to converge to an optimal policy. What we plan to address in this internship, is instead to understand how to learn an efficient policy when we **DO NOT** have full information on the state of the system. To do so, we propose to study a Markov decision problem (MDP) with state $X(t)$ that lives in a modulated environment $D(t)$ (and this modulated environment behaves independent from $X(t)$). Here the state $X(t)$ can be observed, however, the state of the environment $D(t)$ cannot be observed. Note that the transition rates of $X(t)$ do depend on the state the environment $D(t)$ is in.

Consider for example a scheduling problem with two queues. These two queues are modulated, meaning that their parameters (for example departure rate and arrival rates) depend on the state of a so called modulating process (for example, depends on the weather or other environmental effects).

The scheduler, observes the state of the two queues, but not of the modulating process. Here, it's not clear how to define the optimal control. It should be a combination of MDP taking into account that the states of the queues are observed, and a learning part to infer the state of the environment. This is not clear yet, and it will be the objective of the research.

# MDP Problem

## Restless bandit problem

We often work on a particular class of MDP problems, that is, the class of restless bandit problems. Those problems are important, since there exist the so-called Whittle index policy that is proved to be asymptotically optimal for a restless bandit problem. In particular, the problem for this internship is motivated by the following Sigmetrics paper.

This paper studies a restless bandit problem that lives in a modulated environment. We found the Whittle index policy when the environment $D(t)$ changes very fast. Having a fast changing environment allowed us to ignore the learning aspect.

Also, in the numerics we observed an apparent paradox. If we apply the theory of Partially Observable MDP's, we end up with an MDP in which the parameters of the queue are averaged with that of the modulating process. However, we also had heuristics that gave a better performance than this solution of the POMDP (the latter are not presented in the paper).

![[articles/Asymptotic Optimal Control of Markov-Modulated Restless Bandits.pdf]]

### Notions

#### Bandits

- $K$ classes of bandits.
- $N_k$ class-$k$ bandits in the system.
- $N \coloneqq \sum_k N_k$
- $\gamma_k \coloneqq N_k / N$
- $\{1, 2, \dots, J_k\}$ possible states of class-$k$.

#### At a moment

- $a = 0$ (inactive) or $a = 1$ (active).
- At most $\alpha N$ bandits can be made active at a time.

#### Environment

- $D_k(t)$: background positive recurrent Markov process with countable state $\mathcal{Z} = \{1,\dots,d,\dots\}$.
- $\overrightarrow{D} = (D_1,\dots,D_K)$
- $\phi(\vec{d})$: stationary probability vector that $\overrightarrow{D}$ is in state $\vec{d}$.
- $\phi_k(d)$: marginal probability vector that $D_k$ is in state $d$.
- $r(\vec{d'}|\vec{d})$: transition rate of $\overrightarrow{D}(t)$ from $\vec{d}$ to $\vec{d'}$.
- $\sum_{\vec{d'}}r(\vec{d'}|\vec{d}) < C_1$

#### Action

Action $a$ performed on a class-$k$ in state $i$. The environment of this bandit is in state $d$.

- $\frac{1}{N} q_k^{(d)}(j|i,a)$: transition rate to state $j$.
- The $\frac{1}{N}$ to keep the evolution of the state of a bandit is relatively slow compared to that of its environment.
- Evolution of each bandit is independent from other bandits.
- $\bar{q}_k(j|i,a) \coloneqq \sum_{d \in \mathcal{Z}} \phi_k(d)q_k^{(d)}(j|i,a)$
- $\sum_{j=1}^{J_k}q_k^{(d)}(j|i,a) < C_2 \ \forall \ a,d,i,k$

#### Policy

A *policy* of which $\alpha N$ bandits are made active.

- Decision epoch is the moment when one of the bandit changes state.
- Policies base only on propotion of bandits.
- Policies does not rely on the environment and does not try to learn either.
- $x_{j,k}$: propotion of class-$k$ bandits in state $j$ (both activated and none).
- $\vec{x} \coloneqq (x_{j,k};k=1,\dots,K;j=1,\dots,J_k)$
- $\vec{x} \in \mathcal{B} \coloneqq \{\vec{x}: 0 \le x_{j,k} \le 1 \ \forall \ j,k; \sum_j x_{j,k} = \gamma_k \}$

Given $\pi$ a policy,

- $y^{\pi,1}: \mathcal{B} \rightarrow [0, 1]^{\sum_{k=1}^K J_k}$
- $y^{\pi,1}_{j,k}(\vec{x})$: proportion of class-$k$ bandits in state $j$ that are active given $\vec{x}$.
- $y^{\pi,1}_{j,k}(\vec{x}) \le x_{j,k}$
- $y^{\pi,1}_{j,k}(\vec{x}) \le \alpha$
- $y^{\pi,0}_{j,k}(\vec{x}) \coloneqq x_{j,k} - y^{\pi,1}_{j,k}(\vec{x})$: proportion of class-$k$ bandits in state $j$ that are inactive given $\vec{x}$.

We assume $y^{\pi,1}(\cdot)$ is continous.

#### Stable

- $X^{N,\pi}_{j,k}(t)$: the number of class-$k$ bandits that are in state $j$ at time $t$.
- $\overrightarrow{X}^{N,\pi}(t) \coloneqq (X^{N,\pi}_{j,k}(t);k=1,\dots,K;j=1,\dots,J_k)$
- stable $\iff \vec{X^{N,\pi}}(t)$ has a unique invariant probability distribution.
- $\overrightarrow{X}^{N,\pi}$ and $X^{N,\pi}_{j,k}(t)$ the random variables following the steady state distribution (the distribution of these steady states^[https://en.wikipedia.org/wiki/Steady_state]).
- Finite state space ,  $\overrightarrow{X}^{N,\pi}(t)$ is unichain $\implies$ stable.
- Infinite state space, depend on policy.

We will be interested in the set of stable policies.

#### Cost

- $C^{(d)}_k(j, a)$: cost per unit of time for having a class-$k$ bandit in state $j$ under action $a$ and in environment $d$. (negative for reward).
- $\overline{C_k}(j, a) \coloneqq \sum_{d \in \mathcal{Z}}\phi_{k}(d) C^{(d)}_k(j, a)$: average cost over all environment states.

#### Value functions

Initial proportion of bandits $\frac{\overrightarrow{X}^{N,\pi}(0)}{N} = \vec{x} \in \mathcal{B}$.

$$

V_{\_}^{N,\pi}(\vec{x}) \coloneqq \lim_{T \to \infty} \inf \frac{1}{T}\mathbb{E}\left( \int_0^T \sum_{k=1}^{K} \sum_{j=1}^{J_k} \sum_{a=0}^{1} C_k^{(D_k(t))}(j,a) y_{j,k}^{\pi,a} \left(\frac{\overrightarrow{X}^{N,\pi}(t)}{N} \right) dt \right)

$$
$$

V_{+}^{N,\pi}(\vec{x}) \coloneqq \lim_{T \to \infty} \sup \frac{1}{T}\mathbb{E}\left( \int_0^T \sum_{k=1}^{K} \sum_{j=1}^{J_k} \sum_{a=0}^{1} C_k^{(D_k(t))}(j,a) y_{j,k}^{\pi,a} \left(\frac{\overrightarrow{X}^{N,\pi}(t)}{N} \right) dt \right)

$$

Breakdown:

- $\frac{\overrightarrow{X}^{N,\pi}(t)}{N} = \vec{x}$
- Intuitively, the cost per one unit of time all over the system.

if $V_{\_}^{N,\pi}(\vec{x}) = V_{+}^{N,\pi}(\vec{x}) \ \forall \ \vec{x}$, we define $V^{N,\pi}(\vec{x}) = V_{\_}^{N,\pi}(\vec{x})$

#### Objective

Find policy $\pi^{*}$, that is:

$$

V_{+}^{N,\pi^{*}}(\vec{x}) \le V_{\_}^{N,\pi}(\vec{x}), \ \forall \ \vec{x} \ , \forall \ \pi

$$

Under constraint at most $\alpha N$ bandits are active, i.e

$$
\sum_{k=1}^K \sum_{j=1}^{J_k} y_{j,k}^{\pi,a} \left(\frac{\overrightarrow{X}^{N,\pi}(t)}{N} \right) \le \alpha \ , \forall \ t
$$

## General MDP with partial observable state space

This paper studies a general MDP with partial observable state space (so this is a much more general model then ours). They combine Partial Observable MDP with that of Q-learning, and derive several results.

![[articles/Hidden Markov Model Estimation-Based Q-learning for Partially Observable Markov Decision Process.pdf]]

## Reinforcement Learning for Finite-Horizon Restless Multi-Armed Multi-Action Bandits

![[articles/Reinforcement Learning for Finite-Horizon Restless Multi-Armed Multi-Action Bandits.pdf]]

## Queueing Sysems

![[articles/Queueing Systems.pdf]]

## Introduction to Stochastic Dynamic Programming

![[articles/Introduction to Stochastic Dynamic Programming.pdf]]
