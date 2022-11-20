import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import plotly.subplots as sp
from matplotlib import colors
from planning.planning_ext import Reward

from planning import planning_ext

index_type = planning_ext.index_type().dtype
float_type = planning_ext.float_type().dtype


def add_system_helper(original_cls):
    def from_file(cls, path):
        return cls(super(original_cls, cls).from_file(path))

    original_cls.from_file = classmethod(from_file)

    def from_str(cls, content):
        return cls(super(original_cls, cls).from_str(content))

    original_cls.from_str = classmethod(from_str)

    def show_policy(self, algo="q", info=""):
        if self.n_cls != 2:
            return

        policy = self.q_policy if algo == "q" else self.v_policy

        ax = plt.axes()
        fig = ax.get_figure()
        cmap = colors.ListedColormap(["#e5e5e5", "black", "#7f7f7f"])
        bounds = [0, 1, 2, 3]
        norm = colors.BoundaryNorm(bounds, cmap.N)
        mat = ax.matshow(policy, cmap=cmap, norm=norm)
        fig.colorbar(mat, cmap=cmap, boundaries=bounds, ticks=bounds[:-1])

        ax.set_ylabel("L1")
        ax.set_xlabel("L2")

        ax.set_yticks(np.arange(-0.5, policy.shape[0], 1), minor=True)
        ax.set_xticks(np.arange(-0.5, policy.shape[1], 1), minor=True)

        ax.set_title(f"policy_{algo}{info}")
        plt.show()

    original_cls.show_policy = show_policy

    def show_qs(self, index=None, info=""):
        if self.n_cls != 2:
            return

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
                plt.title(f"L1 = {l1} & L2 = {l2}{info}")
                plt.show()

    original_cls.show_qs = show_qs

    def show_n_cls_visit(self, info=""):
        if self.n_cls != 2:
            return

        a1 = self.n_cls_visit[..., 0].ravel()
        a2 = self.n_cls_visit[..., 1].ravel()

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
                go.Mesh3d(
                    x=[
                        i + 0.25,
                        i + 0.25,
                        i + 0.75,
                        i + 0.75,
                        i + 0.25,
                        i + 0.25,
                        i + 0.75,
                        i + 0.75,
                    ],
                    y=[
                        j + 0.25,
                        j + 0.75,
                        j + 0.75,
                        j + 0.25,
                        j + 0.25,
                        j + 0.75,
                        j + 0.75,
                        j + 0.25,
                    ],
                    z=[0, 0, 0, 0, z1, z1, z1, z1],
                    alphahull=0,
                    hoverinfo="text",
                    hovertext=f"L1 = {i}<br />L2 = {j}<br />n_visit = {z1}",
                    color="blue",
                ),
                row=1,
                col=1,
            )

            fig.add_trace(
                go.Mesh3d(
                    x=[
                        i + 0.25,
                        i + 0.25,
                        i + 0.75,
                        i + 0.75,
                        i + 0.25,
                        i + 0.25,
                        i + 0.75,
                        i + 0.75,
                    ],
                    y=[
                        j + 0.25,
                        j + 0.75,
                        j + 0.75,
                        j + 0.25,
                        j + 0.25,
                        j + 0.75,
                        j + 0.75,
                        j + 0.25,
                    ],
                    z=[0, 0, 0, 0, z2, z2, z2, z2],
                    alphahull=0,
                    hoverinfo="text",
                    hovertext=f"L1 = {i}<br />L2 = {j}<br />n_visit = {z2}",
                    color="orange",
                ),
                row=1,
                col=2,
            )

        fig.update_layout(title_text=info)
        fig.show()

    original_cls.show_n_cls_visit = show_n_cls_visit

    def __eq__(self, rhs):
        return super(original_cls, self).__eq__(rhs)

    original_cls.__eq__ = __eq__

    return original_cls


@add_system_helper
class Queuing(planning_ext.Queuing):
    def __init__(self, *nargs, **kwargs):
        super().__init__(*nargs, **kwargs)

    @classmethod
    def new(
        cls,
        n_env,
        n_cls,
        limits,
        costs,
        arrivals,
        departures,
        env_trans_mats,
        reward_type=Reward.linear_2,
        **kwargs,
    ):
        return cls(
            n_env,
            n_cls,
            np.asarray(limits, dtype=index_type),
            np.asarray(costs, dtype=float_type),
            np.asarray(arrivals, dtype=float_type),
            np.asarray(departures, dtype=float_type),
            np.asarray(env_trans_mats, dtype=float_type),
            reward_type,
            **kwargs,
        )


@add_system_helper
class LoadBalance(planning_ext.LoadBalance):
    def __init__(self, *nargs, **kwargs):
        super().__init__(*nargs, **kwargs)

    @classmethod
    def new(
        cls,
        n_env,
        n_cls,
        limits,
        costs,
        arrival,
        departures,
        env_trans_mats,
        reward_type=Reward.linear_2,
        **kwargs,
    ):
        return cls(
            n_env,
            n_cls,
            np.asarray(limits, dtype=index_type),
            np.asarray(costs, dtype=float_type),
            arrival,
            np.asarray(departures, dtype=float_type),
            np.asarray(env_trans_mats, dtype=float_type),
            reward_type,
            **kwargs,
        )
