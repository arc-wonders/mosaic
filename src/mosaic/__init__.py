"""MOSAIC: Modular Object-agnostic Semantic Architecture with Indexed Concepts.

Plug-in visual knowledge and body geometry, served to a text-only LLM as MCP tools (see mosaic.server).
"""

from mosaic.encoder import Encoder, normalize
from mosaic.matcher import Match, Matcher
from mosaic.module import KnowledgeModule
from mosaic.perceiver import Perceiver, Region
from mosaic.pipeline import Mosaic, Prediction
from mosaic.registry import EncoderMismatchError, Registry
from mosaic.router import Route, Router

__all__ = [
    "Encoder",
    "EncoderMismatchError",
    "KnowledgeModule",
    "Match",
    "Matcher",
    "Mosaic",
    "Perceiver",
    "Prediction",
    "Region",
    "Registry",
    "Route",
    "Router",
    "normalize",
]
