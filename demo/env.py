from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors


class Env:
    def __init__(self, c1, c2, p1_u, p2_u, p1_d, p2_d, max_L1, max_L2, seed=42):
        self.seed = seed

        self.c1 = c1
        self.c2 = c2

        self.p1_u = p1_u
        self.p2_u = p2_u
        self.p1_d = p1_d
        self.p2_d = p2_d

        self.actions = [0, 1]
        self.n_action = len(self.actions)
        self.max_L1 = max_L1
        self.max_L2 = max_L2
        self.dim_state = (self.max_L1 + 1, self.max_L2 + 1)
        self.n_state = (self.max_L1 + 1) * (self.max_L2 + 1) - 1

        # [p1_u, p2_u, p1_d, p2_d, p]
        self.transitions = [(1, 0), (0, 1), (-1, 0), (0, -1), (0, 0)]

        self.reset()

    def optimal(self):
        if self.c1 / self.p1_d > self.c2 / self.p2_d:
            return 0
        else:
            return 1

    def reset(self):
        self.rng = np.random.default_rng(self.seed)
        self.L1 = 0
        self.L2 = 0
        self.state = (self.L1, self.L2)
        self.rewards = [self.reward()]
        self.n_visit = defaultdict(int)

        return self.state, self.rewards[-1], False

    def reward(self):
        return -(self.c1 * self.L1 + self.c2 * self.L2)

    def step(self, action):
        self.rewards.append(self.reward())

        p1_u = self.p1_u * (self.L1 < self.max_L1)
        p2_u = self.p2_u * (self.L2 < self.max_L2)
        p1_d = self.p1_d * (action == 0) * (self.L1 > 0)
        p2_d = self.p2_d * (action == 1) * (self.L2 > 0)

        next_trans = self.rng.choice(
            len(self.transitions),
            1,
            p=[p1_u, p2_u, p1_d, p2_d, 1 - p1_u - p1_d - p2_u - p2_d],
        )
        self.L1 += self.transitions[next_trans[0]][0]
        self.L2 += self.transitions[next_trans[0]][1]
        self.state = (self.L1, self.L2)
        return self.state, self.rewards[-1], False

    def random_action(self):
        return self.rng.choice(self.actions)

    def train(
        self,
        epoch=1,
        gamma=0.9,
        eps=0.01,
        decay=0.5,
        ls=1000000,
        lr_pow=0.51,
        save_Q=False,
    ):
        Q = np.zeros(self.dim_state + (self.n_action,))
        Q_avg = np.empty([2, epoch * ls])
        if save_Q:
            Qs = np.empty(self.dim_state + (self.n_action, epoch * ls))

        for i in range(epoch):
            state, reward, _ = self.reset()
            for j in range(ls):
                if state[0] == 0:
                    action = 1
                elif state[1] == 0:
                    action = 0
                elif self.rng.uniform() < eps:
                    action = self.random_action()
                else:
                    action = np.argmax(Q[state])

                next_state, reward, _ = self.step(action)

                next_q = 0
                if next_state[0] == 0:
                    next_q = Q[next_state][1]
                elif next_state[1] == 0:
                    next_q = Q[next_state][0]
                else:
                    next_q = np.max(Q[next_state])

                lr = 1 / (self.n_visit[state + (action,)] + 1) ** lr_pow

                if state == (0, 0):
                    Q[state][0] += lr * (reward + gamma * next_q - Q[state][0])
                    Q[state][1] = Q[state][0]
                else:
                    Q[state][action] += lr * (
                        reward + gamma * next_q - Q[state][action]
                    )
                if save_Q:
                    Qs[:, :, :, i * ls + j] = Q.copy()
                Q_avg[:, i * ls + j] = np.mean(Q, axis=(0, 1))

                self.n_visit[state + (action,)] += 1
                state = next_state

            eps -= decay * eps
        Q[Q == 0] = -np.inf
        policy = np.argmax(Q, axis=-1)

        self.Q = Q
        if save_Q:
            self.Qs = Qs
        self.Q_avg = Q_avg
        self.policy = policy

    def show_policy(self, ax=None, info=""):
        ax = ax or plt.axes()
        fig = ax.get_figure()
        cmap = colors.ListedColormap(["lightgray", "black"])
        bounds = [0, 1, 2]
        norm = colors.BoundaryNorm(bounds, cmap.N)
        mat = ax.matshow(self.policy, cmap=cmap, norm=norm)
        fig.colorbar(mat, cmap=cmap, boundaries=bounds, ticks=bounds[:-1])

        ax.set_ylabel("L1")
        ax.set_xlabel("L2")
        ax.get_xaxis().set_label_position("top")

        ax.set_yticks(np.arange(-0.5, self.policy.shape[0], 1), minor=True)
        ax.set_xticks(np.arange(-0.5, self.policy.shape[1], 1), minor=True)
        ax.grid(which="minor", color="w", linestyle="-", linewidth=1)

        ax.set_title(f"Optimal policy - Optimal action {self.optimal()}{info}")

    def show_Q(self, ax=None, info=""):
        ax = ax or plt.axes()
        ax.plot(self.Q_avg[0], label="action 0", alpha=0.5)
        ax.plot(self.Q_avg[1], label="action 1", alpha=0.5)
        ax.set_title(f"Q-learning value - Optimal action {self.optimal()}{info}")
        ax.legend()
