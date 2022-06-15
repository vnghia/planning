import ctypes
import io

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors

from .planning_ext import *


class Env:
    def __init__(self, ls, cs, pus=None, pds=None, type="linear", save_q=False):
        self.ls = ls
        self.cs = cs
        self.pus = pus if pus is not None else [0.1, 0.1]
        self.pds = pds if pds is not None else [0.3, 0.3]
        self.type = type
        self.save_q = save_q
        self.env = vars(planning_ext)[f"{type}_env_{int(save_q)}_{ls[0]}_{ls[1]}"](
            cs, self.pus, self.pds
        )
        self._policy = None

    def __repr__(self):
        return (
            f"{self.type}_env: "
            f"ls: {self.ls} cs: {self.cs} pus: {self.pus} pds: {self.pds} save_q: {self.save_q}"
        )

    def __getstate__(self):
        state = io.BytesIO()
        np.savez_compressed(
            state,
            ls=self.ls,
            cs=self.cs,
            pus=self.pus,
            pds=self.pds,
            type=self.type,
            save_q=self.save_q,
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
            data["cs"],
            data["pus"],
            data["pds"],
            data["type"].item(),
            data["save_q"].item(),
        )

        q = data["q"]
        n_visit = data["n_visit"]
        qs = (
            np.zeros(shape=(0,) + q.shape, dtype=q.dtype)
            if not self.save_q
            else data["qs"]
        )

        self.env.from_array(q, n_visit, qs, qs.size)
        self._policy = data["policy"]

    def train(self, gamma=0.9, eps=0.01, decay=0.5, epoch=1, ls=1000000, lr_pow=0.51):
        self.env.train(gamma, eps, decay, epoch, ls, lr_pow)
        self._policy = np.argmax(self.q, axis=-1)

    @property
    def q(self):
        return self.env.q

    @property
    def n_visit(self):
        return self.env.n_visit

    @property
    def qs(self):
        return self.env.qs if self.save_q else None

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
                l2 = index[1] if len(index) > 1 else j
                plt.title(f"L1 = {l1} - L2 = {l2}{info}")
                plt.show()
