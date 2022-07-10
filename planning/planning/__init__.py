import io
from multiprocessing import Pool

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.subplots as sp
from matplotlib import colors

from planning import planning_ext
from planning.planning_ext import Reward


class Env:
    def __init__(
        self,
        lens,
        cost,
        arrival,
        departure,
        prob=None,
        C=None,
        env_type=None,
        cost_eps=None,
        save_qs=None,
    ):
        self.lens = lens

        self.cost = np.array(cost).reshape((2, -1))
        self.arrival = np.array(arrival).reshape((2, -1))
        self.departure = np.array(departure).reshape((2, -1))
        self.prob = (
            np.array(prob).reshape((2, -1)) if prob is not None else np.zeros((2, 1))
        )

        self.n_queue = 2
        self.n_env = self.cost.shape[1]
        self.dims_queue = tuple(np.array(self.lens) + 1)

        self.C = C or 1
        self.env_type = env_type or Reward.linear_2
        self.cost_eps = cost_eps if cost_eps is not None else 1
        self.save_qs = bool(save_qs)

        self.__env = vars(planning_ext)[
            f"env_{self.n_env}"
            f"_{int(self.save_qs)}"
            f"_{self.lens[0]}"
            f"_{self.lens[1]}"
        ](
            self.cost.ravel("F"),
            self.arrival.ravel("F") / self.C,
            self.departure.ravel("F") / self.C,
            self.prob.ravel("F") / self.C,
            self.env_type,
            cost_eps=self.cost_eps,
        )

        self._policy_q = None

        self.summary = pd.DataFrame()
        self.summary["name"] = (
            [f"Cost {i + 1}" for i in range(self.n_queue)]
            + [f"Arrival {i + 1}" for i in range(self.n_queue)]
            + [f"Departure {i + 1}" for i in range(self.n_queue)]
            + [f"Probability {i + 1}" for i in range(self.n_queue)]
        )
        for i in range(self.n_env):
            self.summary[f"Env {i + 1}"] = (
                self.cost[:, i].tolist()
                + (self.arrival[:, i] / self.C).tolist()
                + (self.departure[:, i] / self.C).tolist()
                + (self.prob[:, i] / self.C).tolist()
            )

    def __repr__(self):
        return (
            f"{self.env_type}_env:"
            f" lens: {self.lens}"
            f" cost: {self.cost}"
            f" arrival: {self.arrival}"
            f" departure: {self.departure}"
            f" prob: {self.prob}"
            f" C: {self.C}"
            ""
            if self.env_type != "convex"
            else f" cost_eps: {self.cost_eps}"
        )

    def __getstate__(self):
        state = io.BytesIO()
        np.savez_compressed(
            state,
            lens=self.lens,
            cost=self.cost,
            arrival=self.arrival,
            departure=self.departure,
            prob=self.prob,
            C=self.C,
            env_type=self.env_type,
            cost_eps=self.cost_eps,
            save_qs=self.save_qs,
            q=self.q,
            n_visit=self.n_visit,
            qs=self.qs,
            policy_q=self.policy_q,
        )
        return state.getvalue()

    def __setstate__(self, state):
        data = np.load(io.BytesIO(state), allow_pickle=True)
        self.__init__(
            data["lens"],
            data["cost"],
            data["arrival"],
            data["departure"],
            data["prob"],
            data["C"].item(),
            data["env_type"].item(),
            data["cost_eps"].item(),
            data["save_qs"].item(),
        )

        self._q = data["q"]
        self._n_visit = data["n_visit"]
        self._qs = data["qs"] if self.save_qs else None
        self._policy_q = data["policy_q"]

    def train_q(
        self,
        gamma=None,
        eps=None,
        decay=None,
        epoch=None,
        ls=None,
        seed=None,
    ):
        gamma = gamma or 0.9
        eps = eps or 0.8
        decay = decay or 0.5
        epoch = epoch or 1
        ls = ls or 100000000
        seed = seed or 42

        self.__env.train_q(gamma, eps, decay, epoch, ls, seed)

        shape = self.dims_queue + (self.n_queue,)
        self._q = np.reshape(np.ravel(self.__env.q, order="C"), shape, order="F")
        self._n_visit = np.reshape(
            np.ravel(self.__env.n_visit, order="C"), shape, order="F"
        )
        if self.save_qs:
            self._qs = np.reshape(
                np.ravel(self.__env.qs, order="C"), (-1,) + shape, order="F"
            )
        self._policy_q = np.argmax(self.q, axis=-1)

    def train_v(self, gamma=None, ls=None):
        gamma = gamma or 0.9
        ls = ls or 1000

        self.__env.train_v(gamma, ls)

        self._v = np.reshape(
            np.ravel(self.__env.v, order="C"), self.dims_queue, order="F"
        )
        self._policy_v = np.reshape(
            np.ravel(self.__env.policy_v, order="C"), self.dims_queue, order="F"
        )

    @property
    def q(self):
        return self._q

    @property
    def n_visit(self):
        return self._n_visit

    @property
    def qs(self):
        return self._qs if self.save_qs else None

    @property
    def policy_q(self):
        return self._policy_q

    @property
    def v(self):
        return self._v

    @property
    def policy_v(self):
        return self._policy_v

    def show_policy(self, algo="q", ax=None, info=""):
        policy = self.policy_q if algo == "q" else self.policy_v

        ax = ax or plt.axes()
        fig = ax.get_figure()
        cmap = colors.ListedColormap(["lightgray", "black"])
        bounds = [0, 1, 2]
        norm = colors.BoundaryNorm(bounds, cmap.N)
        mat = ax.matshow(policy, cmap=cmap, norm=norm)
        fig.colorbar(mat, cmap=cmap, boundaries=bounds, ticks=bounds[:-1])

        ax.set_ylabel("L1")
        ax.set_xlabel("L2")

        ax.set_yticks(np.arange(-0.5, policy.shape[0], 1), minor=True)
        ax.set_xticks(np.arange(-0.5, policy.shape[1], 1), minor=True)
        ax.grid(which="minor", color="w", linestyle="-", linewidth=1)

        ax.set_title(f"policy_{algo}{info}")

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
        if self.env_type == "convex":
            y2 = x2**2

        for i in range(self.n_env):
            ax.plot(x1, self.cost[0, i] * y1, label=f"env {i} cost 0")
            ax.plot(x2, self.cost[1, i] * y2, label=f"env {i} cost 1")

        ax.legend()
        ax.set_title(f"cost{info}")

    def show_ratio(self, ax=None, info=""):
        ax = ax or plt.axes()
        x1 = np.linspace(0, self.lens[0], 100)
        y1 = x1
        x2 = np.linspace(0, self.lens[1], 100)
        y2 = x2
        if self.env_type == "convex":
            y2 = x2**2

        for i in range(self.n_env):
            r1 = self.cost[0, i] / self.departure[0, i]
            r2 = self.cost[1, i] / self.departure[1, i]
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
    def init_and_train_q(
        cls,
        lens,
        cost,
        arrival,
        departure,
        prob=None,
        C=None,
        env_type=None,
        cost_eps=None,
        save_qs=None,
        gamma=None,
        eps=None,
        decay=None,
        epoch=None,
        ls=None,
        seed=None,
    ):
        env = cls(lens, cost, arrival, departure, prob, C, env_type, cost_eps, save_qs)
        env.train_q(gamma, eps, decay, epoch, ls, seed)
        return env

    @staticmethod
    def get_param(
        lens,
        cost,
        arrival,
        departure,
        prob=None,
        C=None,
        env_type=None,
        cost_eps=None,
        save_qs=None,
        gamma=None,
        eps=None,
        decay=None,
        epoch=None,
        ls=None,
        seed=None,
    ):
        return (
            lens,
            cost,
            arrival,
            departure,
            prob,
            C,
            env_type,
            cost_eps,
            save_qs,
            gamma,
            eps,
            decay,
            epoch,
            ls,
            seed,
        )

    @classmethod
    def train_parallel(cls, parameters):
        with Pool() as p:
            return p.starmap(cls.init_and_train_q, parameters)
