import io
from multiprocessing import Pool

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors

from .planning_ext import *


class Env:
    def __init__(self, ls, param, prob=None, type="linear", save_qs=False):
        self.ls = ls
        self.param = np.array(param).reshape((-1, 2, 3))
        self.n_env = self.param.shape[0]
        self.prob = (
            np.array(prob).reshape((-1, 2)) if self.n_env > 1 else np.zeros((1, 2))
        )
        self.type = type
        self.save_qs = save_qs
        self.__env = vars(planning_ext)[
            f"{type}_env_{self.n_env}_{int(save_qs)}_{ls[0]}_{ls[1]}"
        ](self.param, self.prob)
        self._policy = None

    def __repr__(self):
        return f"{self.type}_env: " f"ls: {self.ls} param: {self.param}"

    def __getstate__(self):
        state = io.BytesIO()
        np.savez_compressed(
            state,
            ls=self.ls,
            param=self.param,
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
            data["param"],
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
                l2 = index[1] if len(index) > 1 else j
                plt.title(f"L1 = {l1} - L2 = {l2}{info}")
                plt.show()

    def show_cost(self, ax=None, info=""):
        ax = ax or plt.axes()
        x1 = np.linspace(0, self.ls[0], 100)
        x2 = np.linspace(0, self.ls[1], 100)
        ax.plot(x1, self.cs[0] * x1, label="cost 0")
        if self.type == "convex":
            y2 = self.cs[1] * x2**2
        else:
            y2 = self.cs[1] * x2
        ax.plot(x2, y2, label="cost 1")
        ax.legend()
        ax.set_title(f"cost{info}")

    def show_ratio(self, ax=None, info=""):
        ax = ax or plt.axes()
        x1 = np.linspace(0, self.ls[0], 100)
        x2 = np.linspace(0, self.ls[1], 100)

        r1 = self.param[0, 0] / self.param[0, 2]
        ax.plot(x1, r1 * x1, label="ratio 0")

        r2 = self.param[1, 0] / self.param[1, 2]
        if self.type == "convex":
            y2 = r2 * x2**2
        else:
            y2 = r2 * x2
        ax.plot(x2, y2, label="ratio 1")

        ax.legend()
        ax.set_title(f"ratio {r1/r2}{info}")

    def show_n_visit(self, info=""):
        fig = plt.figure(figsize=plt.figaspect(0.5))
        _x = np.arange(self.ls[0] + 1)
        _y = np.arange(self.ls[1] + 1)
        _xx, _yy = np.meshgrid(_x, _y, indexing="ij")
        x, y = _xx.ravel(), _yy.ravel()

        ax1 = fig.add_subplot(1, 2, 1, projection="3d")
        dz1 = self.n_visit.transpose(2, 0, 1)[0].ravel()
        ax1.bar3d(
            x,
            y,
            np.zeros_like(dz1),
            np.ones_like(x) * 0.75,
            np.ones_like(x) * 0.75,
            dz1,
            shade=True,
            color="blue",
        )
        ax1.set_xlabel("L1")
        ax1.set_xticks(_x)
        ax1.set_ylabel("L2")
        ax1.set_yticks(_y)
        ax1.set_title("action 0")
        ax1.set_ylim(ax1.get_ylim()[::-1])

        ax2 = fig.add_subplot(1, 2, 2, projection="3d")
        dz2 = self.n_visit.transpose(2, 0, 1)[1].ravel()
        ax2.bar3d(
            x,
            y,
            np.zeros_like(dz2),
            np.ones_like(x) * 0.75,
            np.ones_like(x) * 0.75,
            dz2,
            shade=True,
            color="orange",
        )
        ax2.set_xlabel("L1")
        ax2.set_xticks(_x)
        ax2.set_ylabel("L2")
        ax2.set_yticks(_y)
        ax2.set_title("action 1")
        ax2.set_ylim(ax2.get_ylim()[::-1])

        fig.suptitle(f"n_visit{info}")

    @classmethod
    def init_and_train(
        cls, ls, param, type, save_qs, gamma, eps, decay, epoch, learns, lr_pow
    ):
        env = cls(ls, param, type, save_qs)
        env.train(gamma, eps, decay, epoch, learns, lr_pow)
        return env

    @staticmethod
    def get_param(
        ls,
        param,
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
