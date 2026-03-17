"""
Manifest-based dataset loader for YOLO training.
Enables "Physical Isolation, Logical Mixing" of datasets.
"""

import os
import yaml
import glob
import random
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

try:
    from PIL import Image as PILImage
except ImportError:
    PILImage = None


@dataclass
class DataSourceConfig:
    """Configuration for a single data source in the manifest."""
    source: str
    weight: float
    label_map: Optional[str] = None

    def is_negative_sample(self) -> bool:
        """Check if this source contains negative samples (no labels)."""
        return self.label_map == 'empty'

    def is_digit_to_dial(self) -> bool:
        """Check if this source has digit-level labels to convert to dial-level."""
        return self.label_map == 'digit_to_dial'

    def is_digit_to_position(self) -> bool:
        """Check if this source has digit labels to convert to class-agnostic positions."""
        return self.label_map == 'digit_to_position'


@dataclass
class ManifestConfig:
    """Configuration parsed from manifest YAML file."""
    train_policy: List[DataSourceConfig] = field(default_factory=list)
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'ManifestConfig':
        """Load manifest configuration from YAML file."""
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
        
        train_policy = []
        for item in data.get('train_policy', []):
            item = {k: v for k, v in item.items() if k in ('source', 'weight', 'label_map')}
            train_policy.append(DataSourceConfig(**item))
        
        return cls(train_policy=train_policy)


class ManifestLoader:
    """
    Loads and processes dataset manifests for YOLO training.
    Handles weighted sampling, negative samples, and data validation.
    """
    
    def __init__(self, manifest_path: str, workspace_root: Optional[str] = None):
        """
        Initialize the manifest loader.
        
        Args:
            manifest_path: Path to the manifest YAML file
            workspace_root: Root directory for resolving relative paths (defaults to manifest parent)
        """
        self.manifest_path = Path(manifest_path)
        if workspace_root is None:
            # Use the parent of the manifest's parent (assumes datasets/ folder)
            self.workspace_root = self.manifest_path.parent.parent
        else:
            self.workspace_root = Path(workspace_root)
        
        self.config = ManifestConfig.from_yaml(str(self.manifest_path))
        self.image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    
    def resolve_path(self, relative_path: str) -> Path:
        """Resolve a path relative to the workspace root."""
        path = Path(relative_path)
        if path.is_absolute():
            return path
        return self.workspace_root / path

    def _ensure_digit_to_dial_cache(self, source_config: DataSourceConfig) -> Path:
        """
        Convert digit-level labels (classes 0-9) to a single class-0 dial bbox.
        Writes converted labels to a sibling '<source>_dial/' directory.
        Idempotent: skips conversion for labels that already exist.
        Returns the path to the converted source root.
        """
        src_path = self.resolve_path(source_config.source)
        dst_path = src_path.parent / (src_path.name + '_dial')
        dst_labels = dst_path / 'labels'
        dst_images = dst_path / 'images'
        src_labels = src_path / 'labels'

        if not src_labels.exists():
            raise FileNotFoundError(f"Labels directory not found: {src_labels}")

        dst_labels.mkdir(parents=True, exist_ok=True)
        if not dst_images.exists():
            dst_images.symlink_to((src_path / 'images').resolve())

        for src_label in sorted(src_labels.glob('*.txt')):
            dst_label = dst_labels / src_label.name
            if dst_label.exists():
                continue
            boxes = []
            for line in src_label.read_text().splitlines():
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 5:
                    continue
                try:
                    cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                    x1 = cx - w / 2
                    x2 = cx + w / 2
                    y1 = cy - h / 2
                    y2 = cy + h / 2
                    boxes.append((x1, y1, x2, y2))
                except (ValueError, IndexError):
                    continue
            if not boxes:
                dst_label.touch()
            else:
                x1_min = min(b[0] for b in boxes)
                y1_min = min(b[1] for b in boxes)
                x2_max = max(b[2] for b in boxes)
                y2_max = max(b[3] for b in boxes)
                cx = (x1_min + x2_max) / 2
                cy = (y1_min + y2_max) / 2
                w = x2_max - x1_min
                h = y2_max - y1_min
                dst_label.write_text(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

        return dst_path

    def _ensure_digit_to_position_cache(self, source_config: DataSourceConfig) -> Path:
        """
        Crop images to dial ROI and re-normalize digit labels into crop space.
        ROI is derived from the union of all digit bounding boxes (same as digit_to_dial).
        Writes cropped images and re-normalized labels to '<source>_pos/'.
        Idempotent: skips images whose output label already exists.
        Returns the path to the converted source root.
        """
        if PILImage is None:
            raise ImportError("PIL/Pillow required for digit_to_position. Install with: pip install Pillow")

        src_path = self.resolve_path(source_config.source)
        dst_path = src_path.parent / (src_path.name + '_pos')
        dst_labels = dst_path / 'labels'
        dst_images = dst_path / 'images'
        src_images_dir = src_path / 'images'
        src_labels = src_path / 'labels'

        if not src_images_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {src_images_dir}")
        if not src_labels.exists():
            raise FileNotFoundError(f"Labels directory not found: {src_labels}")

        dst_labels.mkdir(parents=True, exist_ok=True)
        dst_images.mkdir(parents=True, exist_ok=True)

        padding = 0.05

        images = []
        for ext in self.image_extensions:
            images.extend(src_images_dir.glob(f'*{ext}'))
            images.extend(src_images_dir.glob(f'*{ext.upper()}'))
        images = sorted(set(images))

        for img_path in images:
            stem = img_path.stem
            dst_label = dst_labels / f"{stem}.txt"
            if dst_label.exists():
                continue

            digit_boxes = self._read_digit_labels(src_labels / f"{stem}.txt")
            if not digit_boxes:
                continue

            x1_min = min(cx - w / 2 for cx, cy, w, h in digit_boxes)
            y1_min = min(cy - h / 2 for cx, cy, w, h in digit_boxes)
            x2_max = max(cx + w / 2 for cx, cy, w, h in digit_boxes)
            y2_max = max(cy + h / 2 for cx, cy, w, h in digit_boxes)
            roi_cx = (x1_min + x2_max) / 2
            roi_cy = (y1_min + y2_max) / 2
            roi_w = x2_max - x1_min
            roi_h = y2_max - y1_min

            img = PILImage.open(img_path).convert("RGB")
            img_w, img_h = img.size

            cx, cy, w, h = roi_cx, roi_cy, roi_w, roi_h
            pad_w = w * padding
            pad_h = h * padding
            x1_n = max(0.0, cx - w / 2 - pad_w)
            y1_n = max(0.0, cy - h / 2 - pad_h)
            x2_n = min(1.0, cx + w / 2 + pad_w)
            y2_n = min(1.0, cy + h / 2 + pad_h)

            x1_px = int(x1_n * img_w)
            y1_px = int(y1_n * img_h)
            x2_px = int(x2_n * img_w)
            y2_px = int(y2_n * img_h)

            crop_w = x2_px - x1_px
            crop_h = y2_px - y1_px
            if crop_w < 1 or crop_h < 1:
                continue

            cropped = img.crop((x1_px, y1_px, x2_px, y2_px))

            out_labels = []
            for (dcx, dcy, dw, dh) in digit_boxes:
                px1, py1, px2, py2 = self._norm_to_pixel(dcx, dcy, dw, dh, img_w, img_h)
                clip_x1 = max(px1, x1_px)
                clip_y1 = max(py1, y1_px)
                clip_x2 = min(px2, x2_px)
                clip_y2 = min(py2, y2_px)
                if clip_x2 <= clip_x1 or clip_y2 <= clip_y1:
                    continue
                rel_x1 = clip_x1 - x1_px
                rel_y1 = clip_y1 - y1_px
                rel_x2 = clip_x2 - x1_px
                rel_y2 = clip_y2 - y1_px
                ncx, ncy, nw, nh = self._pixel_to_norm(rel_x1, rel_y1, rel_x2, rel_y2, crop_w, crop_h)
                if nw < 0.001 or nh < 0.001:
                    continue
                out_labels.append(f"0 {ncx:.6f} {ncy:.6f} {nw:.6f} {nh:.6f}\n")

            if not out_labels:
                continue

            out_img = dst_images / f"{stem}{img_path.suffix}"
            cropped.save(out_img)
            dst_label.write_text("".join(out_labels))

        return dst_path

    def _read_digit_labels(self, path: Path) -> List[Tuple[float, float, float, float]]:
        """Read digit boxes from label file. Returns list of (cx, cy, w, h) in normalized coords."""
        boxes = []
        if not path.exists():
            return boxes
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            try:
                boxes.append((float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])))
            except (ValueError, IndexError):
                continue
        return boxes

    def _norm_to_pixel(self, cx: float, cy: float, w: float, h: float, img_w: int, img_h: int) -> Tuple[float, float, float, float]:
        """Convert normalized YOLO box to pixel corners (x1, y1, x2, y2)."""
        x1 = (cx - w / 2) * img_w
        y1 = (cy - h / 2) * img_h
        x2 = (cx + w / 2) * img_w
        y2 = (cy + h / 2) * img_h
        return (x1, y1, x2, y2)

    def _pixel_to_norm(self, x1: float, y1: float, x2: float, y2: float, crop_w: float, crop_h: float) -> Tuple[float, float, float, float]:
        """Convert pixel corners to normalized YOLO box (cx, cy, w, h)."""
        cx = (x1 + x2) / 2 / crop_w
        cy = (y1 + y2) / 2 / crop_h
        w = (x2 - x1) / crop_w
        h = (y2 - y1) / crop_h
        return (cx, cy, w, h)

    def get_images_from_source(self, source_config: DataSourceConfig) -> List[Path]:
        """
        Get all image files from a data source.
        
        Args:
            source_config: Configuration for the data source
            
        Returns:
            List of image file paths
        """
        if source_config.is_digit_to_dial():
            source_path = self._ensure_digit_to_dial_cache(source_config)
        elif source_config.is_digit_to_position():
            source_path = self._ensure_digit_to_position_cache(source_config)
        else:
            source_path = self.resolve_path(source_config.source)
        images_dir = source_path / 'images'
        
        if not images_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {images_dir}")
        
        images = []
        for ext in self.image_extensions:
            images.extend(images_dir.glob(f'*{ext}'))
            images.extend(images_dir.glob(f'*{ext.upper()}'))
        
        return sorted(images)
    
    def validate_source(self, source_config: DataSourceConfig) -> Tuple[int, int, List[str]]:
        """
        Validate a data source and check for missing labels.
        
        Args:
            source_config: Configuration for the data source
            
        Returns:
            Tuple of (num_images, num_labels, missing_labels)
        """
        images = self.get_images_from_source(source_config)
        num_images = len(images)
        
        if source_config.is_negative_sample():
            # Negative samples don't need labels
            return num_images, 0, []

        if source_config.is_digit_to_dial():
            source_path = self._ensure_digit_to_dial_cache(source_config)
        elif source_config.is_digit_to_position():
            source_path = self._ensure_digit_to_position_cache(source_config)
        else:
            source_path = self.resolve_path(source_config.source)
        labels_dir = source_path / 'labels'
        
        if not labels_dir.exists():
            raise FileNotFoundError(f"Labels directory not found: {labels_dir}")
        
        missing_labels = []
        num_labels = 0
        
        for img_path in images:
            label_path = labels_dir / f"{img_path.stem}.txt"
            if label_path.exists():
                num_labels += 1
            else:
                missing_labels.append(str(img_path))
        
        return num_images, num_labels, missing_labels
    
    def generate_weighted_image_list(self, split: str = 'train', 
                                     val_split: float = 0.2,
                                     seed: int = 42) -> Tuple[List[str], List[str]]:
        """
        Generate weighted image lists for training and validation.
        
        Args:
            split: 'train' or 'val' (for future use)
            val_split: Fraction of data to use for validation
            seed: Random seed for reproducibility
            
        Returns:
            Tuple of (train_image_paths, val_image_paths)
        """
        random.seed(seed)
        
        # Pass 1: Labeled sources -> 80/20 train/val split
        labeled_images = []
        for source_config in self.config.train_policy:
            if source_config.is_negative_sample():
                continue
            images = self.get_images_from_source(source_config)
            weight = source_config.weight
            base_count = int(weight)
            fractional_weight = weight - base_count
            for img_path in images:
                for _ in range(base_count):
                    labeled_images.append(str(img_path))
                if random.random() < fractional_weight:
                    labeled_images.append(str(img_path))
        
        random.shuffle(labeled_images)
        split_idx = int(len(labeled_images) * (1 - val_split))
        train_images = list(labeled_images[:split_idx])
        val_images = list(labeled_images[split_idx:])
        
        # Pass 2: Negative samples -> train only (create empty label stubs, exclude from val)
        for source_config in self.config.train_policy:
            if not source_config.is_negative_sample():
                continue
            images = self.get_images_from_source(source_config)
            weight = source_config.weight
            base_count = int(weight)
            fractional_weight = weight - base_count
            for img_path in images:
                for _ in range(base_count):
                    self.create_empty_label(str(img_path))
                    train_images.append(str(img_path))
                if random.random() < fractional_weight:
                    self.create_empty_label(str(img_path))
                    train_images.append(str(img_path))
        
        return (train_images, val_images)
    
    def create_empty_label(self, image_path: str) -> str:
        """
        Create an empty label file path for negative samples.
        Places the label co-located with the images so Ultralytics can resolve it
        (e.g. .../negative_samples/images/foo.jpg -> .../negative_samples/labels/foo.txt).
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Path where empty label was created
        """
        img_path = Path(image_path)
        # Derive co-located labels dir from image path
        labels_dir = img_path.parent.parent / 'labels'
        labels_dir.mkdir(parents=True, exist_ok=True)
        label_path = labels_dir / f"{img_path.stem}.txt"
        if not label_path.exists():
            label_path.touch()
        return str(label_path)
    
    def get_statistics(self) -> Dict:
        """
        Get statistics about the manifest datasets.
        
        Returns:
            Dictionary with dataset statistics
        """
        stats = {
            'sources': [],
            'total_images': 0,
            'total_labels': 0,
            'weighted_total': 0
        }
        
        for source_config in self.config.train_policy:
            num_images, num_labels, missing = self.validate_source(source_config)
            
            source_stats = {
                'path': source_config.source,
                'weight': source_config.weight,
                'images': num_images,
                'labels': num_labels,
                'missing_labels': len(missing),
                'is_negative': source_config.is_negative_sample(),
                'is_digit_to_dial': source_config.is_digit_to_dial(),
                'is_digit_to_position': source_config.is_digit_to_position(),
                'weighted_contribution': num_images * source_config.weight
            }
            
            stats['sources'].append(source_stats)
            stats['total_images'] += num_images
            stats['total_labels'] += num_labels
            stats['weighted_total'] += source_stats['weighted_contribution']
        
        return stats
    
    def print_statistics(self):
        """Print dataset statistics in a human-readable format."""
        stats = self.get_statistics()
        
        print("=" * 80)
        print("DATASET MANIFEST STATISTICS")
        print("=" * 80)
        print(f"Manifest: {self.manifest_path}")
        print(f"Workspace Root: {self.workspace_root}")
        print()
        
        for i, source in enumerate(stats['sources'], 1):
            print(f"Source {i}: {source['path']}")
            print(f"  Weight: {source['weight']}")
            print(f"  Images: {source['images']}")
            print(f"  Labels: {source['labels']}")
            if source['missing_labels'] > 0:
                print(f"  ⚠ Missing Labels: {source['missing_labels']}")
            if source['is_negative']:
                print(f"  ℹ Negative Samples (no labels expected)")
            if source.get('is_digit_to_dial'):
                print(f"  ℹ Digit-to-dial conversion (labels cached to <source>_dial/)")
            if source.get('is_digit_to_position'):
                print(f"  ℹ Digit-to-position (crop+renorm) conversion (labels cached to <source>_pos/)")
            print(f"  Weighted Contribution: {source['weighted_contribution']:.1f}")
            print()
        
        print("-" * 80)
        print(f"Total Physical Images: {stats['total_images']}")
        print(f"Total Labels: {stats['total_labels']}")
        print(f"Effective Dataset Size (weighted): {stats['weighted_total']:.1f}")
        print("=" * 80)


def test_manifest_loader():
    """Test function for the manifest loader."""
    manifest_path = "datasets/mix_v1_robust.yaml"
    
    print("Testing Manifest Loader...")
    print()
    
    try:
        loader = ManifestLoader(manifest_path)
        loader.print_statistics()
        
        print("\nGenerating weighted image lists...")
        train_imgs, val_imgs = loader.generate_weighted_image_list()
        print(f"Training images: {len(train_imgs)}")
        print(f"Validation images: {len(val_imgs)}")
        print(f"First 3 training images:")
        for img in train_imgs[:3]:
            print(f"  - {img}")
        
    except FileNotFoundError as e:
        print(f"⚠ Warning: {e}")
        print("This is expected if data directories don't exist yet.")
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    test_manifest_loader()
