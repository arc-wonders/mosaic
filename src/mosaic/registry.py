"""Registry: a plug-in / plug-out tree of knowledge modules.

Modules are leaves named by path, e.g. "fruits/apple". Every proper prefix
("fruits") is a *group*: it holds no data, and its routing descriptors are a
summary of its children's, recomputed whenever a child is attached or detached.
A module named without "/" is simply a leaf at the top level.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from mosaic.module import KnowledgeModule, kmeans


class EncoderMismatchError(ValueError):
    """A module was built with a different encoder than the running one."""


def _parent(name: str) -> str:
    return name.rsplit("/", 1)[0] if "/" in name else ""


class Registry:
    def __init__(self, encoder_id: str, dim: int, max_group_descriptors: int | None = 16):
        self.encoder_id = encoder_id
        self.dim = dim
        self.max_group_descriptors = max_group_descriptors
        self._modules: dict[str, KnowledgeModule] = {}
        self._group_descriptors: dict[str, np.ndarray] = {}
        self._children: dict[str, list[str]] | None = None  # tree index, rebuilt lazily
        self.version = 0  # bumped on every plug-in / plug-out

    # ---- plug in / plug out ------------------------------------------------------
    def add(self, module: KnowledgeModule) -> None:
        name = module.name
        if module.encoder_id != self.encoder_id or module.dim != self.dim:
            raise EncoderMismatchError(
                f"module '{name}' built with {module.encoder_id} (dim {module.dim}); "
                f"registry expects {self.encoder_id} (dim {self.dim})"
            )
        if self.is_group(name):
            raise ValueError(f"'{name}' is already a group of modules; a node cannot be both")
        clash = next((p for p in self._ancestors(name) if p in self._modules), None)
        if clash:
            raise ValueError(f"'{clash}' is a leaf module; cannot attach '{name}' beneath it")
        self._modules[name] = module
        self._invalidate(name)

    def remove(self, name: str) -> list[str]:
        """Detach a leaf, or every leaf under a group. Returns the leaves removed."""
        removed = sorted(n for n in self._modules if n == name or n.startswith(name + "/"))
        if not removed:
            raise KeyError(name)
        for n in removed:
            del self._modules[n]
            self._invalidate(n)
        return removed

    def load_dir(self, root: str | Path) -> list[str]:
        """Attach every module found under `root` (at any depth). Returns the names loaded."""
        loaded = []
        for manifest in sorted(Path(root).rglob("manifest.json")):
            module = KnowledgeModule.load(manifest.parent)
            self.add(module)
            loaded.append(module.name)
        return loaded

    # ---- tree queries --------------------------------------------------------------
    def get(self, name: str) -> KnowledgeModule:
        return self._modules[name]

    def is_group(self, name: str) -> bool:
        return name in self._tree()

    def children(self, parent: str = "") -> list[str]:
        """Immediate child nodes (leaves or groups) of `parent`; "" is the root."""
        return self._tree().get(parent, [])

    def leaves(self, under: str = "") -> list[str]:
        return sorted(n for n in self._modules if not under or n == under or n.startswith(under + "/"))

    def descriptors(self, name: str) -> np.ndarray:
        """Routing descriptors of a leaf, or the (cached) summary of a group's children."""
        if name in self._modules:
            return self._modules[name].descriptors
        if name not in self._group_descriptors:
            stacked = np.concatenate([self.descriptors(c) for c in self.children(name)])
            cap = self.max_group_descriptors
            self._group_descriptors[name] = kmeans(stacked, cap) if cap and len(stacked) > cap else stacked
        return self._group_descriptors[name]

    def __contains__(self, name: str) -> bool:
        return name in self._modules or (name != "" and self.is_group(name))

    def __len__(self) -> int:
        return len(self._modules)

    def __iter__(self):
        return iter(self._modules.values())

    # ---- internals ---------------------------------------------------------------
    @staticmethod
    def _ancestors(name: str) -> list[str]:
        out = []
        while "/" in name:
            name = _parent(name)
            out.append(name)
        return out

    def _tree(self) -> dict[str, list[str]]:
        """group name -> sorted children; "" is the root."""
        if self._children is None:
            kids: dict[str, set[str]] = {"": set()}
            for name in self._modules:
                parts = name.split("/")
                for depth in range(len(parts)):
                    kids.setdefault("/".join(parts[:depth]), set()).add("/".join(parts[: depth + 1]))
            self._children = {g: sorted(c) for g, c in kids.items()}
        return self._children

    def _invalidate(self, name: str) -> None:
        self._children = None
        self.version += 1
        for group in self._ancestors(name):
            self._group_descriptors.pop(group, None)
