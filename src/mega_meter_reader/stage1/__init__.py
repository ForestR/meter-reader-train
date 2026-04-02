"""Stage 1: dial + decimal_section segmentation and DIAL ROI post-processing."""

from mega_meter_reader.stage1.predict import run_stage1
from mega_meter_reader.stage1.types import Stage1Output

__all__ = ["Stage1Output", "run_stage1"]
