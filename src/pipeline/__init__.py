"""
Pipeline assembly module: 3-stage meter reading inference.
Image in, reading out. Supports visualization and auto-labeling.
"""

from .reader import PipelineReader, ReadingResult
from .visualizer import draw_pipeline_result
from .auto_labeler import AutoLabeler
from .label_sorter import LabelSorter
from .manual_labeler import ManualLabeler

__all__ = [
    "PipelineReader",
    "ReadingResult",
    "AutoLabeler",
    "LabelSorter",
    "ManualLabeler",
    "draw_pipeline_result",
]
