"""MCP server: MOSAIC as the eyes of a text-only reasoning model.

Run:  python -m mosaic.server            (stdio transport; or the `mosaic-mcp` script)
Env:  MOSAIC_MODULES   module directories, ':'-separated (default: <repo>/modules_people).
                       Gate thresholds and the geometry weight come from the first one's registry.json.
      MOSAIC_TAUGHT    where modules and rules taught in a session are saved (default: <repo>/taught)

Models load on the first tool call, so the server starts instantly.
"""

from __future__ import annotations

import os
from pathlib import Path

from mcp.server.mcpserver import MCPServer

REPO = Path(__file__).resolve().parents[2]

INSTRUCTIONS = """\
MOSAIC gives you sight without giving you pixels. It finds objects without knowing what they are, looks each
one up in a registry of plug-in knowledge modules (the people domain: person, head, face, hair, eye, nose,
mouth, ear, arm, hand, leg, foot, plus anything taught in a session), and measures bodies with a skeleton and
geometry rules. You do the reasoning. Everything comes back as numbers and names you can check.
All coordinates are [x0, y0, x1, y1] in pixels of the original image, origin top-left.

Typical loop:
  1. load_image -> image_id; find_objects(image_id) -> object ids with boxes
  2. identify(object_id) -> module path, or known=false with the nearest candidates. Unknown is a real answer:
     the module's own gate says the object isn't typical of anything stored. Don't guess past it.
  3. pose(image_id) -> each person's joints and ~45 facts from the geometry rules (elbow angles, arm raised,
     sitting, facing, hands' estimated positions ...). Facts whose joints weren't seen are listed as unknown.
  4. measure(person_id, "angle(left_shoulder, left_elbow, left_wrist)") for any geometry you need;
     define_rule(...) to name a concept once and have it evaluated on every person from then on.
  5. Look closer: find_parts(object_id), find_objects(image_id, region=[...]) (zooms in), search(image_id,
     "people/limbs/hand", region=...) to scan for a module, relate(a, b) for touching / overlap / containment.
     For "is she holding something": take hands_estimated_at from pose, zoom there, then relate or measure
     touches(left_hand, obj_N).
  6. explain(object_id) shows why: the routing scores, the gate, and the stored examples it matched.
  7. teach(label, [object_ids]) plugs in a new module from objects you've found (the user's own mug, a
     specific tool); forget(name) plugs it out again and deletes it. Both take effect immediately.
Only modules from list_modules() can be recognised; free-text search() is a weaker fallback.
"""

mcp = MCPServer("mosaic", instructions=INSTRUCTIONS, log_level="WARNING")
_session = None


def session():
    global _session
    if _session is None:
        from mosaic.capability import VisionSession

        modules = os.environ.get("MOSAIC_MODULES", str(REPO / "modules_people")).split(":")
        _session = VisionSession(modules, taught_dir=os.environ.get("MOSAIC_TAUGHT", str(REPO / "taught")))
    return _session


@mcp.tool()
def load_image(path: str) -> dict:
    """Load an image file. Returns image_id, width and height (pixels)."""
    return session().load_image(path)


@mcp.tool()
def find_objects(image_id: str, region: list[int] | None = None, max_objects: int = 8) -> dict:
    """Find objects without identifying them (class-agnostic).

    region: optional [x0, y0, x1, y1] to zoom into; the region is upscaled before detection, so small things
    (a hand, something held in a hand) become findable. Returns object_id, box, mask_box (tight box of the
    segmentation mask), area_fraction and objectness for each object. Ids are stable: the same thing found
    again keeps its id.
    """
    return session().find_objects(image_id, region, max_objects)


@mcp.tool()
def identify(object_id: str | None = None, image_id: str | None = None, box: list[int] | None = None) -> dict:
    """What is this? Pass an object_id, or an image_id plus a box [x0, y0, x1, y1] for any region.

    Returns module (e.g. "people/limbs/hand"), known (bool), reason, similarity to the closest stored example,
    part_of, the top-3 candidate modules with whether each would accept it, and for body parts a geometry block:
    which joints of the owner's skeleton fall inside the box, and whether the skeleton changed the answer
    (e.g. appearance said face, but the box holds both ears and the top of the head, so it's a head).
    """
    return session().identify(object_id, image_id, box)


@mcp.tool()
def find_parts(object_id: str, max_parts: int = 8) -> dict:
    """Zoom into an object and find the objects inside it (e.g. a person's hands, face, arms, legs, or
    things they hold), each identified."""
    return session().find_parts(object_id, max_parts)


@mcp.tool()
def search(image_id: str, query: str, region: list[int] | None = None, scan: bool = True) -> dict:
    """Look for something, optionally only inside region [x0, y0, x1, y1].

    query: a registry module path from list_modules (a leaf like "people/limbs/hand", or a group like
    "people/face_region"), or free text (the encoder's text tower gives a match probability; weaker).
    For module queries the region is also scanned with windows scored against that module, so this finds
    things the detector merged into something else or missed. "match" is true only when the registry itself
    identifies the result as the queried module.
    """
    return session().search(image_id, query, region, scan)


@mcp.tool()
def relate(object_a: str, object_b: str) -> dict:
    """Spatial relation between two objects in the same image, from their masks: overlap fractions,
    containment, touching, direction of b relative to a, and centre distance."""
    return session().relate(object_a, object_b)


@mcp.tool()
def list_modules(path: str = "") -> dict:
    """The registry tree: groups and leaf modules with example counts, what each part is part of, and
    which were taught in a session. Pass a group path to list only its subtree."""
    return session().list_modules(path)


@mcp.tool()
def pose(image_id: str | None = None, object_id: str | None = None) -> dict:
    """Skeletons: for every person in the image (or the one person object given), the visible joints
    [x, y, confidence], the estimated centre of each hand, and every geometry rule evaluated:
      facts               e.g. left_elbow_angle 92, left_upper_arm "down", left_elbow_turn "right",
                          left_arm_L true, hands_up false, facing "camera", sitting true
      constraints_broken  anatomy checks that failed (limb proportions ...): the skeleton may be wrong there
      unknown             rules that needed a joint that wasn't seen, with the missing joints
    Directions are in the image frame (up = top of image); left_*/right_* joints are the person's own sides.
    """
    return session().pose(image_id, object_id)


@mcp.tool()
def measure(person_id: str, expression: str) -> dict:
    """Evaluate a geometry expression on one person's skeleton (see list_rules for joints and functions).

    Examples: "angle(right_hip, right_knee, right_ankle)", "dist(left_hand, right_hand)",
    "direction(left_elbow, left_wrist)", "touches(right_hand, obj_7)", "above(left_wrist, nose) and left_arm_straight".
    Returns value (None if a needed joint wasn't seen) and missing (the joints that weren't).
    """
    return session().measure(person_id, expression)


@mcp.tool()
def define_rule(name: str, expression: str, description: str = "", kind: str = "fact") -> dict:
    """Name a geometry concept so pose() evaluates it on every person from now on, and other rules can use it.
    kind: "fact" (reported as a value) or "constraint" (reported only when it fails). Saved across sessions.
    Example: define_rule("waving", "left_arm_raised and left_elbow_angle < 150 or right_arm_raised and
    right_elbow_angle < 150", "a raised, bent arm"). Returns its value on the people seen so far."""
    return session().define_rule(name, expression, description, kind)


@mcp.tool()
def list_rules() -> dict:
    """The geometry vocabulary: joint and derived point names, functions (angle, direction, goes, turn,
    toward, dist, above, touches ...) and every named rule with its expression."""
    return session().list_rules()


@mcp.tool()
def teach(label: str, object_ids: list[str], part_of: str | None = None, description: str = "") -> dict:
    """Plug in a new module from objects found in this session: e.g. the user says "that's my mug" ->
    teach("my_mug", ["obj_5"]). More examples (other photos) make its gate better calibrated; teaching the
    same label again adds examples. A bare label goes under "taught/"; a path ("people/limbs/tattoo") places
    it in the tree. It's recognised from the next call on, and saved across sessions. Returns its gate,
    whether the examples are recognised again, and the closest existing modules (possible confusions)."""
    return session().teach(label, object_ids, part_of, description)


@mcp.tool()
def forget(name: str) -> dict:
    """Plug out a module or a whole group ("taught/my_mug", "taught"), or drop a rule. Taught modules and
    rules are deleted from disk too; nothing learned from them remains. Built-in modules are plugged out for
    this session only."""
    return session().forget(name)


@mcp.tool()
def explain(object_id: str) -> dict:
    """Why identify() answered what it did: routing scores at each level of the tree, the domain gate,
    the module's own accept gate, the 5 stored examples it matched (source photo and box), the runner-up,
    and what the skeleton contributed."""
    return session().explain(object_id)


def main():
    mcp.run("stdio")


if __name__ == "__main__":
    main()
