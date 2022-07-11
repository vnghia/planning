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


class System:
    def __init__(
        self,
        limits,
        costs,
        arrivals,
        departures,
        env_trans_probs=None,
        reward_type=None,
        save_qs=None,
        **kwargs,
    ):
        self.limits = limits
        self.n_class = len(self.limits)

        self.costs = np.array(costs).reshape((self.n_class, -1))
        self.n_env = self.costs.shape[1]

        self.arrivals = np.array(arrivals).reshape((self.n_class, self.n_env))
        self.departures = np.array(departures).reshape((self.n_class, self.n_env))

        self.env_trans_probs = (
            np.array(env_trans_probs).reshape((self.n_class, self.n_env, self.n_env))
            if self.n_env > 1
            else np.zeros((self.n_class, 1, 1))
        )

        self.class_dims = tuple(np.array(self.limits) + 1)

        self.reward_type = reward_type or Reward.linear_2
        self.save_qs = bool(save_qs)
        self.kwargs = kwargs

        self.cpp_type = (
            f"system_{self.n_env}"
            f"_{int(self.save_qs)}"
            f"_{self.limits[0]}"
            f"_{self.limits[1]}"
        )

        self.__sys = vars(planning_ext)[self.cpp_type](
            self.costs.ravel("F"),
            self.arrivals.ravel("F"),
            self.departures.ravel("F"),
            self.env_trans_probs.ravel("F"),
            self.reward_type,
            **self.kwargs,
        )

        self._policy_q = None

        self.summary = pd.DataFrame()
        self.summary["Param"] = (
            [f"Cost {i + 1}" for i in range(self.n_class)]
            + [f"Arrival {i + 1}" for i in range(self.n_class)]
            + [f"Departure {i + 1}" for i in range(self.n_class)]
        )
        for i in range(self.n_env):
            self.summary[f"Env {i + 1}"] = (
                self.costs[:, i].tolist()
                + (self.arrivals[:, i]).tolist()
                + (self.departures[:, i]).tolist()
            )

    def __repr__(self):
        return self.cpp_type

    def __getstate__(self):
        state = io.BytesIO()
        np.savez_compressed(
            state,
            limits=self.limits,
            costs=self.costs,
            arrivals=self.arrivals,
            departures=self.departures,
            env_trans_probs=self.env_trans_probs,
            reward_type=self.reward_type,
            save_qs=self.save_qs,
            kwargs=self.kwargs,
            q=self.q,
            n_visit=self.n_visit,
            qs=self.qs,
            policy_q=self.policy_q,
        )
        return state.getvalue()

    def __setstate__(self, state):
        data = np.load(io.BytesIO(state), allow_pickle=True)
        self.__init__(
            data["limits"],
            data["costs"],
            data["arrivals"],
            data["departures"],
            data["env_trans_probs"],
            data["reward_type"].item(),
            data["save_qs"].item(),
            **data["kwargs"].item(),
        )

        self._q = data["q"]
        self._n_visit = data["n_visit"]
        self._qs = data["qs"] if self.save_qs else None
        self._policy_q = data["policy_q"]

    @classmethod
    def __to_c_major(_, data, shape):
        return np.reshape(np.ravel(data, order="C"), shape, order="F")

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

        self.__sys.train_q(gamma, eps, decay, epoch, ls, seed)

        shape = self.class_dims + (self.n_class,)
        self._q = self.__to_c_major(self.__sys.q, shape)
        self._n_visit = self.__to_c_major(self.__sys.n_visit, shape)
        self._qs = (
            self.__to_c_major(self.__sys.qs, (-1,) + shape) if self.save_qs else None
        )
        self._policy_q = np.argmax(self.q, axis=-1)

    def train_v(self, gamma=None, ls=None):
        gamma = gamma or 0.9
        ls = ls or 1000

        self.__sys.train_v(gamma, ls)

        self._v = self.__to_c_major(self.__sys.v, self.class_dims)
        self._policy_v = self.__to_c_major(self.__sys.policy_v, self.class_dims)

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

    def show_policy(self, algo="q", info=""):
        if self.n_class != 2:
            return

        policy = self.policy_q if algo == "q" else self.policy_v

        ax = plt.axes()
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

        ax.set_title(f"policy_{algo}{info}")
        plt.show()

    def show_qs(self, index=None, info=""):
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

    def show_ratio(self, info=""):
        if self.n_class != 2:
            return

        ax = plt.axes()
        x1 = np.linspace(0, self.limits[0], 100)
        y1 = x1
        x2 = np.linspace(0, self.limits[1], 100)
        y2 = x2
        if self.reward_type == Reward.convex_2:
            y2 = x2**2

        for i in range(self.n_env):
            r1 = self.costs[0, i] / self.departures[0, i]
            r2 = self.costs[1, i] / self.departures[1, i]
            ax.plot(x1, r1 * y1, label=f"env {i} ratio 0")
            ax.plot(x2, r2 * y2, label=f"env {i} ratio 1")

        ax.legend()
        ax.set_title(f"ratio{info}")
        plt.show()

    def show_n_visit(self, info=""):
        if self.n_class != 2:
            return

        a1 = self.n_visit[..., 0].ravel()
        a2 = self.n_visit[..., 1].ravel()

        _x = np.arange(self.limits[0] + 1)
        _y = np.arange(self.limits[1] + 1)
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
    def __init_and_train_q(
        cls,
        limits,
        costs,
        arrivals,
        departures,
        env_trans_probs,
        reward_type,
        save_qs,
        gamma,
        eps,
        decay,
        epoch,
        ls,
        seed,
        kwargs,
    ):
        sys = cls(
            limits,
            costs,
            arrivals,
            departures,
            env_trans_probs,
            reward_type,
            save_qs,
            **kwargs,
        )
        sys.train_q(gamma, eps, decay, epoch, ls, seed)
        return sys

    @staticmethod
    def get_param(
        limits,
        costs,
        arrivals,
        departures,
        env_trans_probs=None,
        reward_type=None,
        save_qs=None,
        gamma=None,
        eps=None,
        decay=None,
        epoch=None,
        ls=None,
        seed=None,
        **kwargs,
    ):
        return (
            limits,
            costs,
            arrivals,
            departures,
            env_trans_probs,
            reward_type,
            save_qs,
            gamma,
            eps,
            decay,
            epoch,
            ls,
            seed,
            kwargs,
        )

    @classmethod
    def train_parallel(cls, parameters):
        with Pool() as p:
            return p.starmap(cls.__init_and_train_q, parameters)
