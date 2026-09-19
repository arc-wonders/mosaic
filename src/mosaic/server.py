"""MCP server: MOSAIC perception as tools for a reasoning model.

Run:  python -m mosaic.server            (stdio transport)
Env:  MOSAIC_MODULES          module directories, ':'-separated (default: <repo>/modules_masks)
      MOSAIC_ROUTE_THRESHOLD  domain gate (default: "route_threshold" in the first directory's registry.json)

Models load on the first tool call, so the server starts instantly.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from mcp.server.mcpserver import Image, MCPServer

REPO = Path(__file__).resolve().parents[2]

INSTRUCTIONS = """\
MOSAIC is a perception tool. It finds objects without knowing what they are, then looks each one up in a
registry of plug-in knowledge modules (e.g. people/hand, handheld/phone, animals/dog).
All coordinates are [x0, y0, x1, y1] in pixels of the original image, origin top-left.

Typical reasoning loop:
  1. load_image -> image_id
  2. find_objects(image_id) -> object ids with boxes
  3. identify(object_id) -> what it is (module path), or unknown with nearest candidates
  4. find_parts(object_id) to look inside something (a person -> hands, face, arms, legs)
  5. find_objects(image_id, region=[...]) or search(image_id, "handheld", region=[...]) to zoom in on an area,
     e.g. around a hand, and relate(hand_id, other_id) to check touching/overlap/containment.
     If find_parts misses something (a hand merged with the thing it holds), search(image_id, "people/hand",
     region=person_box) asks the registry to scan for it directly.
  6. render(image_id) to see the annotated image if you can view images
identify() answers "unknown" when the closest module does not consider the object typical; the candidates
list still shows the nearest modules and whether each would accept it. Only modules listed by list_modules()
can be recognised; free-text search() is a weaker fallback.
"""

mcp = MCPServer("mosaic", instructions=INSTRUCTIONS, log_level="WARNING")
_session = None


def session():
    global _session
    if _session is None:
        from mosaic.capability import VisionSession

        modules = os.environ.get("MOSAIC_MODULES", str(REPO / "modules_masks")).split(":")
        threshold = os.environ.get("MOSAIC_ROUTE_THRESHOLD")
        if threshold is None:
            config = Path(modules[0]) / "registry.json"
            threshold = json.loads(config.read_text())["route_threshold"] if config.exists() else 0.5
        _session = VisionSession(modules, route_threshold=float(threshold))
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

    Returns module (e.g. "people/hand"), label, known (bool), reason, similarity to the closest stored
    example, part_of (e.g. a hand is part of "people/person"), and the top-3 candidate modules with
    whether each would accept this object.
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

    query: a registry module path from list_modules (a class like "people/hand" or "handheld/phone", or a
    group like "handheld" for anything in it), or free text (CLIP text matching; weaker).
    For module queries the region is also scanned with windows scored against that module, so this finds
    things the detector merged into something else (a hand holding a phone) or missed. Every result is
    re-identified; "match" is true only when the registry identifies it as the queried module.
    """
    return session().search(image_id, query, region, scan)


@mcp.tool()
def relate(object_a: str, object_b: str) -> dict:
    """Spatial relation between two objects in the same image, from their masks: overlap fractions,
    containment, touching, direction of b relative to a, and centre distance."""
    return session().relate(object_a, object_b)


@mcp.tool()
def list_modules(path: str = "") -> dict:
    """The registry tree: groups (e.g. "people", "handheld") and leaf modules with example counts and,
    for parts, what they are part of. Pass a group path to list only its subtree."""
    return session().list_modules(path)


@mcp.tool()
def render(image_id: str, object_ids: list[str] | None = None) -> Image:
    """The image with objects outlined and labelled (default: all objects found so far). Green tint =
    identified as known."""
    return Image(data=session().render(image_id, object_ids), format="png")


def main():
    mcp.run("stdio")


if __name__ == "__main__":
    main()
