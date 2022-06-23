import io
from multiprocessing import Pool

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import plotly.subplots as sp
from matplotlib import colors

from .planning_ext import *


class Env:
    def __init__(self, ls, cost, param, prob=None, type="linear", save_qs=False):
        self.ls = ls
        self.cost = np.array(cost).reshape((-1, 2))
        self.param = np.array(param).reshape((-1, 2, 2))
        self.n_env = self.param.shape[0]
        self.prob = (
            np.array(prob).reshape((-1, 2)) if self.n_env > 1 else np.zeros((1, 2))
        )
        self.type = type
        self.save_qs = save_qs
        self.__env = vars(planning_ext)[
            f"{type}_env_{self.n_env}_{int(save_qs)}_{ls[0]}_{ls[1]}"
        ](self.cost, self.param, self.prob)
        self._policy = None

    def __repr__(self):
        return (
            f"{self.type}_env: "
            f"ls: {self.ls} "
            f"cost: {self.cost} "
            f"param: {self.param} "
            f"prob: {self.prob}"
        )

    def __getstate__(self):
        state = io.BytesIO()
        np.savez_compressed(
            state,
            ls=self.ls,
            cost=self.cost,
            param=self.param,
            prob=self.prob,
            type=self.type,
            save_qs=self.save_qs,
            q=self.q,
            n_visit=self.n_visit,
            qs=self.qs,
            policy=self.policy,
        )
        return state.getvalue()

    def __setstate__(self, state):
        data = np.load(io.BytesIO(state), allow_pickle=True)
        self.__init__(
            data["ls"],
            data["cost"],
            data["param"],
            data["prob"],
            data["type"].item(),
            data["save_qs"].item(),
        )

        q = data["q"]
        n_visit = data["n_visit"]
        qs = (
            np.zeros(shape=(0,) + q.shape, dtype=q.dtype)
            if not self.save_qs
            else data["qs"]
        )

        self.__env.from_array(q, n_visit, qs, qs.size)
        self._policy = data["policy"]

    def train(self, gamma=0.9, eps=0.01, decay=0.5, epoch=1, ls=20000000, lr_pow=0.51):
        self.__env.train(gamma, eps, decay, epoch, ls, lr_pow)
        self._policy = np.argmax(self.q, axis=-1)

    @property
    def q(self):
        return self.__env.q

    @property
    def n_visit(self):
        return self.__env.n_visit

    @property
    def qs(self):
        return self.__env.qs if self.save_qs else None

    @property
    def reward_mat(self):
        return self.__env.reward_mat

    @property
    def policy(self):
        return self._policy

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

        ax.set_yticks(np.arange(-0.5, self.policy.shape[0], 1), minor=True)
        ax.set_xticks(np.arange(-0.5, self.policy.shape[1], 1), minor=True)
        ax.grid(which="minor", color="w", linestyle="-", linewidth=1)

        ax.set_title(f"policy{info}")

    def show_qs(self, info="", index=None):
        qs = self.qs.transpose(1, 2, 3, 0)
        qs = qs if index is None else qs[index]
        qs.shape = (1,) * (4 - qs.ndim) + qs.shape
        for i, q_row in enumerate(qs):
            for j, q_cell in enumerate(q_row):
                plt.plot(q_cell[0], label="action 0", alpha=0.5)
                plt.plot(q_cell[1], label="action 1", alpha=0.5)
                plt.legend()
                l1 = index[0] if index else i
                l2 = index[1] if index and len(index) > 1 else j
                plt.title(f"L1 = {l1} - L2 = {l2}{info}")
                plt.show()

    def show_cost(self, ax=None, info=""):
        ax = ax or plt.axes()
        x1 = np.linspace(0, self.ls[0], 100)
        y1 = x1
        x2 = np.linspace(0, self.ls[1], 100)
        y2 = x2
        if self.type == "convex":
            y2 = x2**2

        for i in range(self.n_env):
            ax.plot(x1, self.cost[i, 0] * y1, label=f"env {i} cost 0")
            ax.plot(x2, self.cost[i, 1] * y2, label=f"env {i} cost 1")

        ax.legend()
        ax.set_title(f"cost{info}")

    def show_ratio(self, ax=None, info=""):
        ax = ax or plt.axes()
        x1 = np.linspace(0, self.ls[0], 100)
        y1 = x1
        x2 = np.linspace(0, self.ls[1], 100)
        y2 = x2
        if self.type == "convex":
            y2 = x2**2

        for i in range(self.n_env):
            r1 = self.cost[i, 0] / self.param[i, 0, 1]
            r2 = self.cost[i, 1] / self.param[i, 1, 1]
            ax.plot(x1, r1 * y1, label=f"env {i} ratio 0")
            ax.plot(x2, r2 * y2, label=f"env {i} ratio 1")

        ax.legend()
        ax.set_title(f"ratio{info}")

    def show_n_visit(self, info=""):
        a1 = self.n_visit[..., 0].ravel()
        a2 = self.n_visit[..., 1].ravel()

        _x = np.arange(self.ls[0] + 1)
        _y = np.arange(self.ls[1] + 1)
        _xx, _yy = np.meshgrid(_x, _y, indexing="ij")
        x, y = _xx.ravel(), _yy.ravel()

        fig = sp.make_subplots(
            rows=1,
            cols=2,
            specs=[[{"type": "surface"}, {"type": "surface"}]],
            subplot_titles=["Action 1", "Action 2"],
        )
        for i, j, z1, z2 in zip(x, y, a1, a2):
            fig.add_trace(
                go.Scatter3d(
                    {
                        "hoverinfo": "text",
                        "hovertext": f"L1 = {i}<br />L2 = {j}<br />n_visit = {z1}",
                        "line": {"color": "blue", "width": 10},
                        "mode": "lines",
                        "showlegend": False,
                        "x": [i, i],
                        "y": [j, j],
                        "z": [z1, 0],
                    }
                ),
                row=1,
                col=1,
            )

            fig.add_trace(
                go.Scatter3d(
                    {
                        "hoverinfo": "text",
                        "hovertext": f"L1 = {i}<br />L2 = {j}<br />n_visit = {z2}",
                        "line": {"color": "orange", "width": 10},
                        "mode": "lines",
                        "showlegend": False,
                        "x": [i, i],
                        "y": [j, j],
                        "z": [z2, 0],
                    }
                ),
                row=1,
                col=2,
            )

        fig.update_layout(title_text=info)
        fig.show()

    @classmethod
    def init_and_train(
        cls, ls, param, prob, type, save_qs, gamma, eps, decay, epoch, learns, lr_pow
    ):
        env = cls(ls, param, prob, type, save_qs)
        env.train(gamma, eps, decay, epoch, learns, lr_pow)
        return env

    @staticmethod
    def get_param(
        ls,
        param,
        prob=None,
        type="linear",
        save_qs=False,
        gamma=0.9,
        eps=0.01,
        decay=0.5,
        epoch=1,
        learns=20000000,
        lr_pow=0.51,
    ):
        return (
            ls,
            param,
            prob,
            type,
            save_qs,
            gamma,
            eps,
            decay,
            epoch,
            learns,
            lr_pow,
        )

    @classmethod
    def train_parallel(cls, parameters):
        with Pool() as p:
            return p.starmap(cls.init_and_train, parameters)
