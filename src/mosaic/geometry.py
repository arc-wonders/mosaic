"""Body geometry: skeletons, anatomical signatures of parts, and geometry rules.

Three things live here, all plain numpy and all computed from keypoints (see `mosaic.pose`):

Skeleton     one person's joints in image pixels, plus derived points (neck, mid_hip,
             left_hand / right_hand extrapolated past the wrist along the forearm).

Signature    where a body-part leaf sits on a skeleton, learned from the leaf's own examples
             and stored in its manifest: which joints fall inside its box (a head holds the
             nose, eyes and ears; a hand holds a wrist but no elbow), how large it is relative
             to its person, and how far down the person it is. At identify time it is a
             log-likelihood that separates parts that look alike but sit differently
             (head vs face, arm vs hand, leg vs foot).

Rules        named expressions over joints, e.g.

                 left_elbow_angle   angle(left_shoulder, left_elbow, left_wrist)
                 left_arm_L_shape   goes(left_shoulder, left_elbow, "down") and between(left_elbow_angle, 60, 120)
                 hands_up           above(left_wrist, nose) and above(right_wrist, nose)

             Rules are data (JSON), can refer to other rules and to objects (obj_3), and are
             evaluated with three-valued logic: a rule that needs a joint the skeleton could not
             see is `None` (unknown) and reports which joints were missing, never a guess.
             Coordinates follow the image: "up" is toward the top of the image, "left" is the
             viewer's left; left_* / right_* joints are the person's own sides.
"""

from __future__ import annotations

import ast
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from mosaic.pose import KEYPOINTS

MIN_CONFIDENCE = 0.3
INDEX = {name: i for i, name in enumerate(KEYPOINTS)}
GROUPS = {"nose": ["nose"], "eyes": ["left_eye", "right_eye"], "ears": ["left_ear", "right_ear"],
          "shoulders": ["left_shoulder", "right_shoulder"], "elbows": ["left_elbow", "right_elbow"],
          "wrists": ["left_wrist", "right_wrist"], "hips": ["left_hip", "right_hip"],
          "knees": ["left_knee", "right_knee"], "ankles": ["left_ankle", "right_ankle"]}


class Missing(Exception):
    """A rule needed a joint (or object) that is not available."""

    def __init__(self, name: str):
        super().__init__(name)
        self.name = name


# ---- skeleton ---------------------------------------------------------------------------------
@dataclass
class Skeleton:
    keypoints: np.ndarray  # (17, 3): x, y, confidence in image pixels
    box: list[float]       # the person box the keypoints were estimated in
    min_confidence: float = MIN_CONFIDENCE

    def visible(self, name: str) -> bool:
        try:
            self.point(name)
            return True
        except Missing:
            return False

    def point(self, name: str) -> np.ndarray:
        """(x, y) of a joint or derived point; raises Missing if it cannot be seen."""
        if name in INDEX:
            x, y, c = self.keypoints[INDEX[name]]
            if c < self.min_confidence:
                raise Missing(name)
            return np.array([x, y], dtype=float)
        if name in DERIVED:
            return DERIVED[name](self)
        raise KeyError(name)

    def _mid(self, a: str, b: str) -> np.ndarray:
        return (self.point(a) + self.point(b)) / 2

    @property
    def scale(self) -> float:
        """Body length unit: the torso (neck to mid-hip); else 1.6 x shoulder width; else a third of the box height."""
        try:
            return max(1.0, float(np.linalg.norm(self.point("neck") - self.point("mid_hip"))))
        except Missing:
            pass
        try:
            return max(1.0, 1.6 * float(np.linalg.norm(self.point("left_shoulder") - self.point("right_shoulder"))))
        except Missing:
            return max(1.0, (self.box[3] - self.box[1]) / 3)

    def summary(self) -> dict:
        return {n: [round(float(x)), round(float(y)), round(float(c), 2)]
                for n, (x, y, c) in zip(KEYPOINTS, self.keypoints) if c >= self.min_confidence}

    @property
    def n_visible(self) -> int:
        return int((self.keypoints[:, 2] >= self.min_confidence).sum())


def _hand(side: str) -> Callable[[Skeleton], np.ndarray]:
    def f(s: Skeleton) -> np.ndarray:  # palm centre: a third of a forearm past the wrist
        wrist, elbow = s.point(f"{side}_wrist"), s.point(f"{side}_elbow")
        return wrist + 0.35 * (wrist - elbow)
    return f


DERIVED: dict[str, Callable[[Skeleton], np.ndarray]] = {
    "neck": lambda s: s._mid("left_shoulder", "right_shoulder"),
    "mid_hip": lambda s: s._mid("left_hip", "right_hip"),
    "left_hand": _hand("left"),
    "right_hand": _hand("right"),
}
POINT_NAMES = set(KEYPOINTS) | set(DERIVED)


# ---- anatomical signature of a part -------------------------------------------------------------
def part_features(box, skeleton: Skeleton) -> dict:
    """Where `box` sits on `skeleton`: visible joints inside it per group, and its size, height and distance from
    the midline relative to the person box. (Measuring position in torso lengths from the neck instead is more
    robust to people cut off by the frame, but was less accurate on held-out Open Images parts: 71.3% vs 74.5%.)"""
    x0, y0, x1, y1 = box
    px, py = 0.05 * (x1 - x0), 0.05 * (y1 - y0)
    counts = {}
    for group, names in GROUPS.items():
        seen = [skeleton.point(n) for n in names if skeleton.visible(n)]
        if seen:  # a group with no visible joint tells us nothing
            counts[group] = int(sum(x0 - px <= x <= x1 + px and y0 - py <= y <= y1 + py for x, y in seen))
    bx0, by0, bx1, by1 = skeleton.box
    return {
        "counts": counts,
        "log_size": math.log(math.sqrt(max(1.0, (x1 - x0) * (y1 - y0)) / max(1.0, (bx1 - bx0) * (by1 - by0)))),
        "rel_y": ((y0 + y1) / 2 - by0) / max(1.0, by1 - by0),
        "rel_x": abs((x0 + x1) / 2 - (bx0 + bx1) / 2) / max(1.0, bx1 - bx0),
    }


FEATURES = ("log_size", "rel_y", "rel_x")


def fit_signature(features: list[dict]) -> dict:
    """Per-leaf signature from its examples' features (Laplace-smoothed counts, Gaussians for the rest)."""
    counts = {}
    for group, names in GROUPS.items():
        observed = [f["counts"][group] for f in features if group in f["counts"]]
        hist = np.bincount(np.asarray(observed, dtype=int), minlength=len(names) + 1)[: len(names) + 1] + 1.0
        counts[group] = [round(float(v), 4) for v in hist / hist.sum()]
    out = {"counts": counts, "n": len(features)}
    for key in FEATURES:
        v = np.array([f[key] for f in features if key in f])
        out[key] = [round(float(v.mean()), 4), round(float(max(v.std(), 0.05)), 4)] if len(v) else None
    return out


def signature_loglik(signature: dict, features: dict) -> float:
    ll = 0.0
    for group, c in features["counts"].items():
        probs = signature["counts"][group]
        ll += math.log(probs[min(c, len(probs) - 1)])
    for key in FEATURES:
        if key in features and signature.get(key):
            mu, sd = signature[key]
            ll += -0.5 * ((features[key] - mu) / sd) ** 2 - math.log(sd)
    return ll


def owner(box, skeletons: list[Skeleton], min_inside: float = 0.5) -> Skeleton | None:
    """The skeleton whose person box contains most of `box` (None if none holds at least `min_inside` of it)."""
    area = max(1.0, (box[2] - box[0]) * (box[3] - box[1]))
    best, best_f = None, min_inside
    for s in skeletons:
        iw = max(0.0, min(box[2], s.box[2]) - max(box[0], s.box[0]))
        ih = max(0.0, min(box[3], s.box[3]) - max(box[1], s.box[1]))
        if iw * ih / area >= best_f:
            best, best_f = s, iw * ih / area
    return best


# ---- rules ----------------------------------------------------------------------------------------
DIRECTIONS = {"right": 0, "up-right": 45, "up": 90, "up-left": 135, "left": 180,
              "down-left": 225, "down": 270, "down-right": 315}

FUNCTIONS_DOC = {
    "angle(a, b, c)": "angle at b in degrees (180 = straight line a-b-c)",
    "heading(a, b)": "direction a -> b in degrees: 0 right, 90 up, 180 left, 270 down (image frame)",
    "direction(a, b)": "direction a -> b as a word: up, up-right, right, down-right, down, down-left, left, up-left",
    "goes(a, b, word, tol=45)": "true if a -> b points within tol degrees of the direction word",
    "turn(a, b, c)": "at b, which way the path a -> b -> c bends as seen in the image: left / right / straight",
    "toward(a, b, c)": "true if moving a -> b brings you closer to c",
    "above(a, b) / below / left_of / right_of": "comparisons in the image frame",
    "dist(a, b)": "distance in body units (1 = torso length, neck to mid-hip)",
    "px(a, b)": "distance in pixels",
    "x(a) / y(a)": "position within the person box, 0..1 (y: 0 = top)",
    "visible(a)": "whether the joint was seen (never unknown)",
    "between(v, lo, hi)": "lo <= v <= hi",
    "touches(a, obj) / inside(a, obj)": "point a within / near an object's mask (obj_N from find_objects)",
    "abs, min, max, round": "as in Python",
}


class RuleBook:
    """Named geometry rules, evaluated against one skeleton (and the session's objects)."""

    def __init__(self, rules: list[dict] | None = None):
        self.rules: dict[str, dict] = {}
        for r in rules or []:
            self.add(r["name"], r["expression"], r.get("description", ""), r.get("kind", "fact"), check=False)

    @classmethod
    def load(cls, *paths: str | Path) -> RuleBook:
        rules = []
        for p in paths:
            if Path(p).exists():
                rules += json.loads(Path(p).read_text())
        return cls(rules)

    def save(self, path: str | Path, only: set[str] | None = None) -> None:
        rows = [{"name": n, **r} for n, r in self.rules.items() if only is None or n in only]
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(rows, indent=2))

    def add(self, name: str, expression: str, description: str = "", kind: str = "fact", check: bool = True) -> None:
        if not name.isidentifier() or name in POINT_NAMES or name in _FUNCTIONS:
            raise ValueError(f"'{name}' cannot be a rule name (must be an identifier, not a joint or function)")
        if kind not in ("fact", "constraint"):
            raise ValueError("kind must be 'fact' or 'constraint'")
        tree = ast.parse(expression, mode="eval")
        _validate(tree)
        if check:
            unknown = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)} - POINT_NAMES - set(_FUNCTIONS) \
                - set(self.rules) - {name, "True", "False"}
            unknown = {u for u in unknown if not u.startswith("obj_")}
            if unknown:
                raise ValueError(f"unknown names {sorted(unknown)}; use joints, other rules, or obj_N")
        self.rules[name] = {"expression": expression, "description": description, "kind": kind}

    def remove(self, name: str) -> None:
        del self.rules[name]

    def evaluate(self, expression: str, skeleton: Skeleton, objects: Callable[[str], Any] | None = None) -> dict:
        """{"value": ..., "missing": [...]}; value is None when a needed joint was not seen."""
        ctx = _Context(self, skeleton, objects)
        try:
            value = ctx.eval(ast.parse(expression, mode="eval").body)
        except Missing as m:
            return {"value": None, "missing": sorted(ctx.missing | {m.name})}
        return {"value": _jsonable(value), "missing": sorted(ctx.missing)}

    def evaluate_all(self, skeleton: Skeleton, objects=None, names: list[str] | None = None) -> dict:
        facts, broken, unknown = {}, [], {}
        for name in names or self.rules:
            rule = self.rules[name]
            r = self.evaluate(rule["expression"], skeleton, objects)
            if r["value"] is None:
                unknown[name] = r["missing"]
            elif rule["kind"] == "constraint":
                if r["value"] is False:
                    broken.append(name)
            else:
                facts[name] = r["value"]
        return {"facts": facts, "constraints_broken": broken, "unknown": unknown}

    def describe(self) -> list[dict]:
        return [{"name": n, **r} for n, r in self.rules.items()]


def _jsonable(v):
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, (int, float, np.floating, np.integer)):
        return round(float(v), 2)
    if isinstance(v, np.ndarray):
        return [round(float(x), 1) for x in v]
    return v


_ALLOWED = (ast.Expression, ast.BoolOp, ast.And, ast.Or, ast.UnaryOp, ast.Not, ast.USub, ast.UAdd,
            ast.BinOp, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Compare, ast.Eq, ast.NotEq, ast.Lt,
            ast.LtE, ast.Gt, ast.GtE, ast.Call, ast.Name, ast.Load, ast.Constant, ast.IfExp, ast.keyword)


def _validate(tree: ast.AST) -> None:
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED):
            raise ValueError(f"not allowed in a rule: {type(node).__name__}")
        if isinstance(node, ast.Call) and not (isinstance(node.func, ast.Name) and node.func.id in _FUNCTIONS):
            raise ValueError(f"unknown function in rule: {ast.unparse(node.func)}")


def _heading(a, b) -> float:
    d = b - a
    return math.degrees(math.atan2(-d[1], d[0])) % 360  # image y grows downward


def _angle_diff(a: float, b: float) -> float:
    return abs((a - b + 180) % 360 - 180)


class _Context:
    def __init__(self, book: RuleBook, skeleton: Skeleton, objects):
        self.book, self.s, self.objects = book, skeleton, objects
        self.missing: set[str] = set()
        self.stack: list[str] = []

    # names: joints, derived points, objects, other rules
    def name(self, n: str):
        if n in ("True", "False"):
            return n == "True"
        if n in POINT_NAMES:
            return self.s.point(n)
        if n.startswith("obj_"):
            if self.objects is None:
                raise Missing(n)
            return self.objects(n)
        if n in self.book.rules:
            if n in self.stack:
                raise ValueError(f"rule '{n}' refers to itself")
            self.stack.append(n)
            try:
                return self.eval(ast.parse(self.book.rules[n]["expression"], mode="eval").body)
            finally:
                self.stack.pop()
        raise ValueError(f"unknown name '{n}'")

    def point(self, v) -> np.ndarray:
        if isinstance(v, np.ndarray):
            return v
        if hasattr(v, "center"):  # an object
            return np.asarray(v.center, dtype=float)
        raise ValueError("expected a joint, point or object")

    def eval(self, node):
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name):
            return self.name(node.id)
        if isinstance(node, ast.BoolOp):  # three-valued: unknown only if it could change the answer
            values = []
            for v in node.values:
                try:
                    values.append(bool(self.eval(v)))
                except Missing as m:
                    self.missing.add(m.name)
                    values.append(None)
            decisive = isinstance(node.op, ast.Or)  # True decides an `or`, False decides an `and`
            if decisive in values:
                return decisive
            if None in values:
                raise Missing(sorted(self.missing)[0] if self.missing else "?")
            return not decisive
        if isinstance(node, ast.UnaryOp):
            v = self.eval(node.operand)
            return (not v) if isinstance(node.op, ast.Not) else (-v if isinstance(node.op, ast.USub) else v)
        if isinstance(node, ast.BinOp):
            a, b = self.eval(node.left), self.eval(node.right)
            if isinstance(node.op, ast.Add):
                return a + b
            if isinstance(node.op, ast.Sub):
                return a - b
            if isinstance(node.op, ast.Mult):
                return a * b
            return a / b if b else float("inf")
        if isinstance(node, ast.Compare):
            left = self.eval(node.left)
            for op, right_node in zip(node.ops, node.comparators):
                right = self.eval(right_node)
                ok = {ast.Eq: lambda a, b: a == b, ast.NotEq: lambda a, b: a != b, ast.Lt: lambda a, b: a < b,
                      ast.LtE: lambda a, b: a <= b, ast.Gt: lambda a, b: a > b,
                      ast.GtE: lambda a, b: a >= b}[type(op)](left, right)
                if not ok:
                    return False
                left = right
            return True
        if isinstance(node, ast.IfExp):
            return self.eval(node.body) if self.eval(node.test) else self.eval(node.orelse)
        if isinstance(node, ast.Call):
            fn = node.func.id
            if fn == "visible":  # asks about visibility, so it never raises Missing
                arg = node.args[0]
                if not isinstance(arg, ast.Name):
                    raise ValueError("visible() takes a joint name")
                try:
                    self.name(arg.id)
                    return True
                except Missing:
                    return False
            args = [self.eval(a) for a in node.args]
            kwargs = {k.arg: self.eval(k.value) for k in node.keywords}
            return _FUNCTIONS[fn](self, *args, **kwargs)
        raise ValueError(f"cannot evaluate {ast.dump(node)}")


def _goes(c, a, b, word, tol=45):
    if word not in DIRECTIONS:
        raise ValueError(f"direction must be one of {list(DIRECTIONS)}")
    return _angle_diff(_heading(c.point(a), c.point(b)), DIRECTIONS[word]) <= tol


def _direction(c, a, b):
    h = _heading(c.point(a), c.point(b))
    return min(DIRECTIONS, key=lambda w: _angle_diff(h, DIRECTIONS[w]))


def _turn(c, a, b, cc, straight=20):
    h1, h2 = _heading(c.point(a), c.point(b)), _heading(c.point(b), c.point(cc))
    d = (h2 - h1 + 180) % 360 - 180  # positive = counter-clockwise in the image = a left turn
    return "straight" if abs(d) <= straight else ("left" if d > 0 else "right")


def _angle(c, a, b, cc):
    v1, v2 = c.point(a) - c.point(b), c.point(cc) - c.point(b)
    cos = float(v1 @ v2 / max(1e-9, np.linalg.norm(v1) * np.linalg.norm(v2)))
    return math.degrees(math.acos(max(-1.0, min(1.0, cos))))


def _toward(c, a, b, target):
    pa, pb, pt = c.point(a), c.point(b), c.point(target)
    return float(np.linalg.norm(pb - pt)) < float(np.linalg.norm(pa - pt))


def _rel(c, a, axis):
    p = c.point(a)
    x0, y0, x1, y1 = c.s.box
    return (p[0] - x0) / max(1.0, x1 - x0) if axis == 0 else (p[1] - y0) / max(1.0, y1 - y0)


def _touches(c, a, obj, tol=0.15):
    """Point a lies on the object's mask, or within tol body units of it."""
    if not hasattr(obj, "distance_to"):
        raise ValueError("touches() needs an object (obj_N) as its second argument")
    return obj.distance_to(c.point(a)) <= tol * c.s.scale


def _inside(c, a, obj):
    if not hasattr(obj, "distance_to"):
        raise ValueError("inside() needs an object (obj_N) as its second argument")
    return obj.distance_to(c.point(a)) == 0


_FUNCTIONS: dict[str, Callable] = {
    "angle": _angle,
    "heading": lambda c, a, b: _heading(c.point(a), c.point(b)),
    "direction": _direction,
    "goes": _goes,
    "turn": _turn,
    "toward": _toward,
    "above": lambda c, a, b: c.point(a)[1] < c.point(b)[1],
    "below": lambda c, a, b: c.point(a)[1] > c.point(b)[1],
    "left_of": lambda c, a, b: c.point(a)[0] < c.point(b)[0],
    "right_of": lambda c, a, b: c.point(a)[0] > c.point(b)[0],
    "dist": lambda c, a, b: float(np.linalg.norm(c.point(a) - c.point(b))) / c.s.scale,
    "px": lambda c, a, b: float(np.linalg.norm(c.point(a) - c.point(b))),
    "x": lambda c, a: _rel(c, a, 0),
    "y": lambda c, a: _rel(c, a, 1),
    "visible": None,  # handled in _Context.eval
    "between": lambda c, v, lo, hi: lo <= v <= hi,
    "touches": _touches,
    "inside": _inside,
    "abs": lambda c, v: abs(v),
    "min": lambda c, *v: min(v),
    "max": lambda c, *v: max(v),
    "round": lambda c, v, n=0: round(v, int(n)),
}

DEFAULT_RULES = Path(__file__).with_name("people_rules.json")
