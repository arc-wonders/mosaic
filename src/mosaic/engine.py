"""Batched, device-resident (GPU) version of Mosaic.classify.

The registry's tree is packed into a few stacked tensors: group descriptors,
leaf descriptors, and every leaf's image embeddings, each with a segment id.
A batch of queries then runs the whole pipeline as tensor ops:

    L1 group scores   = segment-max(Q @ G^T)        -> top-k groups, route gate
    L2 leaf scores    = segment-max(Q @ L^T), masked to children of kept groups -> top-k leaves
    k-NN              = top-k(Q @ X^T), masked to entries of kept leaves
    vote + leaf gate  = scatter-add over (leaf, label) pairs, per-leaf accept threshold

It gives the same answers as Mosaic.classify (up to float ties) and is
rebuilt automatically when modules are plugged in or out. Supports trees
whose leaves are all at depth 2 ("group/leaf") or all at depth 1 (flat).

Note: leaf scores and k-NN similarities are computed densely for every leaf and
then masked; on a GPU one big matmul is cheaper than gathering per-query blocks
at this scale. For very large trees, swap the k-NN stage for a FAISS GPU index.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from mosaic.registry import Registry


@dataclass
class BatchPrediction:
    labels: list[str | None]   # None -> unknown
    modules: list[str | None]
    votes: np.ndarray
    nearest: np.ndarray        # similarity of the closest stored example in the winning leaf
    reasons: list[str]         # "known" | "router" | "match_threshold" | "leaf_threshold" | "negative_margin"


def _segment_max(scores: torch.Tensor, seg: torch.Tensor, n: int) -> torch.Tensor:
    out = torch.full((scores.shape[0], n), -torch.inf, device=scores.device, dtype=scores.dtype)
    return out.scatter_reduce(1, seg.expand(scores.shape[0], -1), scores, reduce="amax", include_self=True)


class BatchEngine:
    def __init__(self, registry: Registry, device: str | None = None, dtype: torch.dtype = torch.float32):
        self.registry = registry
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.dtype = dtype
        self._version = None

    # ---- packing -----------------------------------------------------------------
    def _pack(self) -> None:
        reg = self.registry
        t = lambda a: torch.as_tensor(np.asarray(a), device=self.device)
        top = reg.children()
        self.flat = all(not reg.is_group(n) for n in top)
        if self.flat:
            self.groups, self.leaves = [], list(top)
        else:
            if not all(reg.is_group(n) and all(not reg.is_group(c) for c in reg.children(n)) for n in top):
                raise NotImplementedError("BatchEngine supports flat trees or group/leaf trees of depth 2")
            self.groups = top
            self.leaves = [c for g in top for c in reg.children(g)]
            G = [reg.descriptors(g) for g in self.groups]
            self.G = t(np.concatenate(G)).to(self.dtype)
            self.G_seg = t(np.repeat(np.arange(len(G)), [len(d) for d in G]))
            self.leaf_group = t([self.groups.index(n.rsplit("/", 1)[0]) for n in self.leaves])

        L = [reg.descriptors(n) for n in self.leaves]
        self.L = t(np.concatenate(L)).to(self.dtype)
        self.L_seg = t(np.repeat(np.arange(len(L)), [len(d) for d in L]))

        X, x_leaf, x_pair, self.pairs = [], [], [], []
        pair_id: dict[tuple[int, str], int] = {}
        for i, name in enumerate(self.leaves):
            m = reg.get(name)
            mask = m._kind_masks.get("image", np.zeros(len(m), bool))
            X.append(m.embeddings[mask])
            for e in (e for e, keep in zip(m.entries, mask) if keep):
                key = (i, e["label"])
                if key not in pair_id:
                    pair_id[key] = len(self.pairs)
                    self.pairs.append(key)
                x_pair.append(pair_id[key])
            x_leaf += [i] * int(mask.sum())
        self.X = t(np.concatenate(X)).to(self.dtype)
        self.x_leaf, self.x_pair = t(x_leaf), t(x_pair)
        self.pair_leaf = t([p[0] for p in self.pairs])
        accept = [reg.get(n).accept_threshold for n in self.leaves]
        self.accept = t([-np.inf if a is None else a for a in accept]).to(self.dtype)

        # negatives: counter-examples per leaf, and the margin positives must beat them by
        N, n_leaf = [], []
        for i, name in enumerate(self.leaves):
            m = reg.get(name)
            mask = m._kind_masks.get("negative")
            if mask is not None and mask.any():
                N.append(m.embeddings[mask])
                n_leaf += [i] * int(mask.sum())
        self.N = t(np.concatenate(N)).to(self.dtype) if N else None
        self.n_leaf = t(n_leaf) if N else None
        margin = [reg.get(n).reject_margin for n in self.leaves]
        self.margin = t([-np.inf if m_ is None else m_ for m_ in margin]).float()
        self._version = reg.version

    def refresh(self) -> None:
        if self._version != self.registry.version:
            self._pack()

    # ---- inference ---------------------------------------------------------------
    @torch.no_grad()
    def classify(
        self,
        queries: np.ndarray | torch.Tensor,
        top_k: int | list[int] = (1, 3),
        route_threshold: float = -1.0,
        match_threshold: float = -1.0,
        k: int = 5,
        leaf_gate: bool = True,
        batch_size: int = 1024,
    ) -> BatchPrediction:
        self.refresh()
        top_k = [top_k] if isinstance(top_k, int) else list(top_k)
        Q = torch.as_tensor(queries, device=self.device).to(self.dtype)
        Q = Q / Q.norm(dim=1, keepdim=True).clamp_min(1e-12)
        parts = [self._classify(Q[i : i + batch_size], top_k, route_threshold, match_threshold, k, leaf_gate)
                 for i in range(0, len(Q), batch_size)]
        winner, votes, nearest, reason = (torch.cat(x).cpu().numpy() for x in zip(*parts))
        names = ["known", "router", "match_threshold", "leaf_threshold", "negative_margin"]
        labels, modules = [], []
        for w, r in zip(winner, reason):
            leaf, label = self.pairs[w] if w >= 0 else (None, None)
            modules.append(self.leaves[leaf] if leaf is not None else None)
            labels.append(label if r == 0 else None)
        return BatchPrediction(labels, modules, votes, nearest, [names[r] for r in reason])

    def _classify(self, Q, top_k, route_threshold, match_threshold, k, leaf_gate):
        B, n_leaves = len(Q), len(self.leaves)
        leaf_scores = _segment_max(Q @ self.L.T, self.L_seg, n_leaves)

        if self.flat:
            kl = min(top_k[-1], n_leaves)
            topv, topi = leaf_scores.topk(kl, dim=1)
            ok = topv >= route_threshold
            routed = ok.any(dim=1)
        else:
            gs = _segment_max(Q @ self.G.T, self.G_seg, len(self.groups))
            gv, gi = gs.topk(min(top_k[0], len(self.groups)), dim=1)
            gkeep = torch.zeros_like(gs, dtype=torch.bool).scatter_(1, gi, gv >= route_threshold)
            routed = gkeep.any(dim=1)
            leaf_scores = leaf_scores.masked_fill(~gkeep[:, self.leaf_group], -torch.inf)
            kl = min(top_k[1] if len(top_k) > 1 else top_k[0], n_leaves)
            topv, topi = leaf_scores.topk(kl, dim=1)
            ok = torch.isfinite(topv)
        leaf_keep = torch.zeros((B, n_leaves), dtype=torch.bool, device=self.device).scatter_(1, topi, ok)

        S = Q @ self.X.T
        S = S.masked_fill(~leaf_keep[:, self.x_leaf] | (S < match_threshold), -torch.inf)
        sv, si = S.topk(min(k, S.shape[1]), dim=1)
        valid = torch.isfinite(sv)
        pairs = self.x_pair[si]

        n_pairs = len(self.pairs)
        counts = torch.zeros((B, n_pairs), device=self.device).scatter_add_(1, pairs, valid.float())
        best = torch.full((B, n_pairs), -torch.inf, device=self.device).scatter_reduce(
            1, pairs, sv.float().masked_fill(~valid, -torch.inf), reduce="amax", include_self=True)
        key = torch.where(counts > 0, counts * 4 + best, torch.full_like(best, -torch.inf))
        winner = key.argmax(dim=1)                      # most votes, ties -> highest similarity
        votes = counts.gather(1, winner[:, None]).squeeze(1).long()
        win_leaf = self.pair_leaf[winner]
        nearest = sv.float().masked_fill(~valid | (self.x_leaf[si] != win_leaf[:, None]), -torch.inf).amax(dim=1)

        reason = torch.zeros(B, dtype=torch.long, device=self.device)
        if leaf_gate:
            if self.N is not None:
                neg = (Q @ self.N.T).float().masked_fill(self.n_leaf[None, :] != win_leaf[:, None], -torch.inf)
                reason[nearest - neg.amax(dim=1) < self.margin[win_leaf]] = 4
            reason[nearest < self.accept[win_leaf].float()] = 3
        has_match = valid.any(dim=1)
        reason[~has_match] = 2
        reason[~routed] = 1
        winner = torch.where(routed & has_match, winner, torch.full_like(winner, -1))
        return winner, votes, nearest, reason
