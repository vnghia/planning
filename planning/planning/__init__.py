import io
from multiprocessing import Pool

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import plotly.subplots as sp
from matplotlib import colors

from .planning_ext import *


class Env:
    def __init__(
        self,
        lens,
        cost,
        param,
        prob=None,
        C=None,
        type=None,
        cost_eps=None,
        save_qs=None,
    ):
        self.lens = lens
        self.cost = np.array(cost).reshape((-1, 2))
        self.param = np.array(param).reshape((-1, 2, 2))
        self.n_env = self.param.shape[0]
        self.prob = (
            np.array(prob).reshape((-1, 2)) if self.n_env > 1 else np.zeros((1, 2))
        )
        self.C = C or 1
        self.type = type or "linear"
        self.cost_eps = cost_eps if cost_eps is not None else 1
        self.save_qs = bool(save_qs)
        self.__env = vars(planning_ext)[
            f"{self.type}_env_{self.n_env}_{int(self.save_qs)}_{self.lens[0]}_{self.lens[1]}"
        ](self.cost, self.param / self.C, self.prob / self.C, self.cost_eps)
        self._policy = None

    def __repr__(self):
        return (
            f"{self.type}_env:"
            f" lens: {self.lens}"
            f" cost: {self.cost}"
            f" param: {self.param}"
            f" prob: {self.prob}"
            f" C: {self.C}"(
                "" if self.type != "convex" else f" cost_eps: {self.cost_eps}"
            )
        )

    def __getstate__(self):
        state = io.BytesIO()
        np.savez_compressed(
            state,
            lens=self.lens,
            cost=self.cost,
            param=self.param,
            prob=self.prob,
            C=self.C,
            type=self.type,
            cost_eps=self.cost_eps,
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
            data["lens"],
            data["cost"],
            data["param"],
            data["prob"],
            data["C"].item(),
            data["type"].item(),
            data["cost_eps"].item(),
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

    def train(
        self,
        gamma=None,
        eps=None,
        decay=None,
        epoch=None,
        ls=None,
        lr_pow=None,
        seed=None,
    ):
        gamma = gamma or 0.9
        eps = eps or 0.8
        decay = decay or 0.5
        epoch = epoch or 1
        ls = ls or 100000000
        lr_pow = lr_pow or 1
        seed = seed or 42

        self.__env.train(gamma, eps, decay, epoch, ls, lr_pow, seed)
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
        x1 = np.linspace(0, self.lens[0], 100)
        y1 = x1
        x2 = np.linspace(0, self.lens[1], 100)
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
        x1 = np.linspace(0, self.lens[0], 100)
        y1 = x1
        x2 = np.linspace(0, self.lens[1], 100)
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

        _x = np.arange(self.lens[0] + 1)
        _y = np.arange(self.lens[1] + 1)
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
        cls,
        lens,
        cost,
        param,
        prob=None,
        C=None,
        type=None,
        cost_eps=None,
        save_qs=None,
        gamma=None,
        eps=None,
        decay=None,
        epoch=None,
        ls=None,
        lr_pow=None,
        seed=None,
    ):
        env = cls(lens, cost, param, prob, C, type, cost_eps, save_qs)
        env.train(gamma, eps, decay, epoch, ls, lr_pow, seed)
        return env

    @staticmethod
    def get_param(
        lens,
        cost,
        param,
        prob=None,
        C=None,
        type=None,
        cost_eps=None,
        save_qs=None,
        gamma=None,
        eps=None,
        decay=None,
        epoch=None,
        ls=None,
        lr_pow=None,
        seed=None,
    ):
        return (
            lens,
            cost,
            param,
            prob,
            C,
            type,
            cost_eps,
            save_qs,
            gamma,
            eps,
            decay,
            epoch,
            ls,
            lr_pow,
            seed,
        )

    @classmethod
    def train_parallel(cls, parameters):
        with Pool() as p:
            return p.starmap(cls.init_and_train, parameters)
