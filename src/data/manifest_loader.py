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


@dataclass
class DataSourceConfig:
    """Configuration for a single data source in the manifest."""
    source: str
    weight: float
    label_map: Optional[str] = None
    
    def is_negative_sample(self) -> bool:
        """Check if this source contains negative samples (no labels)."""
        return self.label_map == 'empty'


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
    
    def get_images_from_source(self, source_config: DataSourceConfig) -> List[Path]:
        """
        Get all image files from a data source.
        
        Args:
            source_config: Configuration for the data source
            
        Returns:
            List of image file paths
        """
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
