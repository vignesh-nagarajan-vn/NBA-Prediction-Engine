"""Small histogram gradient-boosted regression trees in pure numpy (squared loss).

Features are bucketed into quantile bins once; each tree is grown depth-first to
`max_depth`, choosing at every node the (feature, bin) split with the largest gain from
per-bin gradient sums. Leaves are L2-regularized means of the residuals. This keeps the
dependency footprint to numpy, which matters on machines where compiled ML wheels are
blocked.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class _Node:
    feature: int = -1
    bin: int = -1
    left: int = -1
    right: int = -1
    value: float = 0.0


@dataclass
class GBMRegressor:
    n_estimators: int = 300
    learning_rate: float = 0.04
    max_depth: int = 3
    min_samples_leaf: int = 80
    l2: float = 1.0
    n_bins: int = 32
    subsample: float = 0.8
    random_state: int = 0
    edges_: list[np.ndarray] = field(default_factory=list)
    trees_: list[list[_Node]] = field(default_factory=list)
    base_: float = 0.0

    # ---- binning -------------------------------------------------------------------------
    def _fit_bins(self, X: np.ndarray) -> None:
        qs = np.linspace(0, 1, self.n_bins + 1)[1:-1]
        self.edges_ = [np.unique(np.nanquantile(X[:, j], qs)) for j in range(X.shape[1])]

    def _bin(self, X: np.ndarray) -> np.ndarray:
        B = np.empty(X.shape, dtype=np.int16)
        for j, e in enumerate(self.edges_):
            B[:, j] = np.searchsorted(e, np.nan_to_num(X[:, j]), side="right")
        return B

    # ---- training ------------------------------------------------------------------------
    def fit(self, X: np.ndarray, y: np.ndarray) -> GBMRegressor:
        X = np.asarray(X, float)
        y = np.asarray(y, float)
        rng = np.random.default_rng(self.random_state)
        self._fit_bins(X)
        B = self._bin(X)
        self.base_ = float(y.mean())
        pred = np.full(len(y), self.base_)
        self.trees_ = []
        for _ in range(self.n_estimators):
            resid = y - pred
            idx = (rng.random(len(y)) < self.subsample).nonzero()[0]
            tree: list[_Node] = []
            self._grow(B, resid, idx, 0, tree)
            self.trees_.append(tree)
            pred += self.learning_rate * self._predict_tree(tree, B)
        return self

    def _grow(self, B, r, idx, depth, tree) -> int:
        node_id = len(tree)
        node = _Node(value=float(r[idx].sum() / (len(idx) + self.l2)))
        tree.append(node)
        if depth >= self.max_depth or len(idx) < 2 * self.min_samples_leaf:
            return node_id
        g_tot, n_tot = r[idx].sum(), len(idx)
        parent = g_tot**2 / (n_tot + self.l2)
        best = (0.0, -1, -1)
        for j in range(B.shape[1]):
            nb = len(self.edges_[j]) + 1
            if nb < 2:
                continue
            bj = B[idx, j]
            g = np.bincount(bj, weights=r[idx], minlength=nb)
            n = np.bincount(bj, minlength=nb)
            gl, nl = np.cumsum(g)[:-1], np.cumsum(n)[:-1]
            gr, nr = g_tot - gl, n_tot - nl
            ok = (nl >= self.min_samples_leaf) & (nr >= self.min_samples_leaf)
            if not ok.any():
                continue
            gain = gl**2 / (nl + self.l2) + gr**2 / (nr + self.l2) - parent
            gain[~ok] = -np.inf
            k = int(np.argmax(gain))
            if gain[k] > best[0]:
                best = (float(gain[k]), j, k)
        if best[1] < 0:
            return node_id
        _, j, k = best
        go_left = B[idx, j] <= k
        node.feature, node.bin = j, k
        node.left = self._grow(B, r, idx[go_left], depth + 1, tree)
        node.right = self._grow(B, r, idx[~go_left], depth + 1, tree)
        return node_id

    @staticmethod
    def _predict_tree(tree: list[_Node], B: np.ndarray) -> np.ndarray:
        out = np.empty(len(B))
        stack = [(0, np.arange(len(B)))]
        while stack:
            nid, rows = stack.pop()
            node = tree[nid]
            if node.feature < 0:
                out[rows] = node.value
                continue
            left = B[rows, node.feature] <= node.bin
            stack.append((node.left, rows[left]))
            stack.append((node.right, rows[~left]))
        return out

    def predict(self, X: np.ndarray) -> np.ndarray:
        B = self._bin(np.asarray(X, float))
        pred = np.full(len(B), self.base_)
        for tree in self.trees_:
            pred += self.learning_rate * self._predict_tree(tree, B)
        return pred

    def feature_gain(self, n_features: int) -> np.ndarray:
        """How often each feature is used for a split (a rough importance measure)."""
        counts = np.zeros(n_features)
        for tree in self.trees_:
            for node in tree:
                if node.feature >= 0:
                    counts[node.feature] += 1
        return counts / max(1, counts.sum())
