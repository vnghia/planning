import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import plotly.subplots as sp
from matplotlib import colors

from planning import planning_ext
from planning.planning_ext import Reward

index_type = planning_ext.index_type().dtype
float_type = planning_ext.float_type().dtype


class System(planning_ext.System):
    def __init__(self, *nargs, **kwargs):
        super().__init__(*nargs, **kwargs)

    @classmethod
    def new(
        cls,
        n_env,
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
            np.asarray(limits, dtype=index_type),
            np.asarray(costs, dtype=float_type),
            np.asarray(arrivals, dtype=float_type),
            np.asarray(departures, dtype=float_type),
            np.asarray(env_trans_mats, dtype=float_type),
            reward_type,
            **kwargs,
        )

    @classmethod
    def from_file(cls, path):
        return cls(super().from_file(path))

    @classmethod
    def from_str(cls, content):
        return cls(super().from_str(content))

    def show_policy(self, algo="q", info=""):
        if self.n_cls != 2:
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
