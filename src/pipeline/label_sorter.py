"""
Label Sorter: Rule-based routing of auto-label output.
Routes labeled images by reading length; copies skipped (low-conf) images for manual review.
"""

import shutil
from pathlib import Path
from typing import Optional, Union

import yaml


class LabelSorter:
    """
    Sorts auto-label output by rules from a YAML config.
    """

    def __init__(
        self,
        rules_path: Union[str, Path] = "configs/labeling_rules.yaml",
        workspace_root: Optional[Union[str, Path]] = None,
    ):
        """
        Args:
            rules_path: Path to labeling_rules.yaml
            workspace_root: Project root for resolving relative rules_path
        """
        root = Path(workspace_root or Path.cwd())
        path = Path(rules_path)
        if not path.is_absolute():
            path = root / path
        with open(path) as f:
            cfg = yaml.safe_load(f)
        reading = cfg.get("reading", {})
        sort_cfg = cfg.get("sort", {})
        self.min_len = int(reading.get("normal_length_min", 5))
        self.max_len = int(reading.get("normal_length_max", 6))
        self.pending_review_dir = sort_cfg.get("pending_review_dir", "pending_review")
        self.atypical_dir = sort_cfg.get("atypical_dir", "atypical")

    def is_normal_reading(self, reading: str) -> bool:
        """True if len(reading) in [normal_length_min, normal_length_max]."""
        n = len(reading)
        return self.min_len <= n <= self.max_len

    def copy_pending_review(
        self,
        img_path: Path,
        dst: Path,
        dry_run: bool = False,
    ) -> None:
        """Copy image to dst/pending_review/images/ for manual annotation."""
        out_dir = dst / self.pending_review_dir / "images"
        if not dry_run:
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / img_path.name
            shutil.copy2(img_path, out_path)

    def move_to_atypical(
        self,
        img_path: Path,
        label_path: Path,
        dst: Path,
        dry_run: bool = False,
    ) -> None:
        """Move image and label from images/labels to atypical/images and atypical/labels."""
        atypical_images = dst / self.atypical_dir / "images"
        atypical_labels = dst / self.atypical_dir / "labels"
        if not dry_run:
            atypical_images.mkdir(parents=True, exist_ok=True)
            atypical_labels.mkdir(parents=True, exist_ok=True)
            shutil.move(str(img_path), str(atypical_images / img_path.name))
            shutil.move(str(label_path), str(atypical_labels / label_path.name))
