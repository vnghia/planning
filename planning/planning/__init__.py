import matplotlib.pyplot as plt
import numpy as np
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
        env_trans_mats=None,
        reward_type=None,
        save_qs=None,
        **kwargs,
    ):
        # init
        self.limits = limits
        self.n_class = len(self.limits)

        self.costs = np.array(costs).reshape((self.n_class, -1))
        self.n_env = self.costs.shape[1]

        self.arrivals = np.array(arrivals).reshape((self.n_class, self.n_env))
        self.departures = np.array(departures).reshape((self.n_class, self.n_env))

        self.env_trans_mats = (
            np.array(env_trans_mats).reshape((self.n_class, self.n_env, self.n_env))
            if self.n_env > 1
            else np.zeros((self.n_class, 1, 1))
        )

        self.reward_type = reward_type or Reward.linear_2
        self.save_qs = bool(save_qs)
        self.kwargs = kwargs

        self.cpp_type = (
            f"system_{self.n_env}"
            f"_{int(self.save_qs)}"
            f"_{self.limits[0]}"
            f"_{self.limits[1]}"
        )

        self._sys = vars(planning_ext)[self.cpp_type](
            self.costs.ravel("F"),
            self.arrivals.ravel("F"),
            self.departures.ravel("F"),
            self.env_trans_mats.ravel("F"),
            self.reward_type,
            **self.kwargs,
        )

        # dimensions

        self.cls_dims = tuple(np.array(self.limits) + 1)
        self.env_dims = (self.n_env,) * self.n_class

        # constexpr state types
        self.states = self._sys.states
        self.cls_states = self._sys.cls_states
        self.env_states = self._sys.env_states
        self.n_state = len(self.states)
        self.n_cls_state = len(self.cls_states)
        self.n_env_state = len(self.env_states)

        # system transitions
        self.trans_probs = self._sys.trans_probs

        # rewards
        self.rewards = self._sys.rewards

        # additional precomputed probabilities
        self.state_cls_trans_probs = self._sys.state_cls_trans_probs
        self.env_trans_probs = self.__to_c_major(
            self._sys.env_trans_probs, (self.n_env_state, self.n_env_state)
        )

        # q learning
        self.train_q = self._sys.train_q
        self.q_shape = self.cls_dims + (self.n_class,)
        self.q = self.__to_c_major(self._sys.q, self.q_shape)
        self.q_n_visit = self.__to_c_major(self._sys.q_n_visit, self.q_shape)
        self.q_policy = self.__to_c_major(self._sys.q_policy, self.cls_dims)
        self.qs = (
            self.__to_c_major(self._sys.qs, (-1,) + self.q_shape)
            if self.save_qs
            else None
        )

        # value iteration
        self.train_v_s = self._sys.train_v_s
        self.v = self.__to_c_major(self._sys.v, self.cls_dims)
        self.v_policy = self.__to_c_major(self._sys.v_policy, self.cls_dims)

        # tilde
        self.train_t = self._sys.train_t
        self.train_v_t = self._sys.train_v_t
        self.t_env_probs = self._sys.t_env_probs
        self.t_cls_trans_probs = self._sys.t_cls_trans_probs
        self.t_cls_rewards = self._sys.t_cls_rewards

    def __repr__(self):
        return self.cpp_type

    @classmethod
    def __to_c_major(_, data, shape):
        return np.reshape(np.ravel(data, order="C"), shape, order="F")

    def show_policy(self, algo="q", info=""):
        if self.n_class != 2:
            return

        policy = self.q_policy if algo == "q" else self.v_policy

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

    def show_q_n_visit(self, info=""):
        if self.n_class != 2:
            return

        a1 = self.q_n_visit[..., 0].ravel()
        a2 = self.q_n_visit[..., 1].ravel()

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
