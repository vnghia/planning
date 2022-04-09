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

## General MDP with partial observable state space

This paper studies a general MDP with partial observable state space (so this is a much more general model then ours). They combine Partial Observable MDP with that of Q-learning, and derive several results.

![[articles/Hidden Markov Model Estimation-Based Q-learning for Partially Observable Markov Decision Process.pdf]]