"""
Pipeline assembly module: 3-stage meter reading inference.
Image in, reading out. Supports visualization and auto-labeling.
"""

from .reader import PipelineReader, ReadingResult
from .visualizer import draw_pipeline_result
from .auto_labeler import AutoLabeler
from .manual_labeler import ManualLabeler

__all__ = [
    "PipelineReader",
    "ReadingResult",
    "AutoLabeler",
    "ManualLabeler",
    "draw_pipeline_result",
]
