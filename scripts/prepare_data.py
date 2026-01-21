#!/usr/bin/env python3
"""
Data Preparation and Validation Script
Validates data structure and manifest sources before training.
"""

import os
import sys
from pathlib import Path
from typing import List, Tuple

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.manifest_loader import ManifestLoader


class DataValidator:
    """
    Validates YOLO dataset structure and manifest configuration.
    """
    
    def __init__(self, workspace_root: str = None):
        """
        Initialize the validator.
        
        Args:
            workspace_root: Root directory for the project
        """
        if workspace_root is None:
            self.workspace_root = Path.cwd()
        else:
            self.workspace_root = Path(workspace_root)
        
        self.errors = []
        self.warnings = []
    
    def check_directory_structure(self) -> bool:
        """
        Check if required directories exist.
        
        Returns:
            True if structure is valid
        """
        print("Checking directory structure...")
        
        required_dirs = [
            'datasets',
            'configs',
            'src/data',
            'scripts'
        ]
        
        all_exist = True
        for dir_path in required_dirs:
            full_path = self.workspace_root / dir_path
            if full_path.exists():
                print(f"  ✓ {dir_path}")
            else:
                print(f"  ✗ {dir_path} (missing)")
                self.errors.append(f"Required directory missing: {dir_path}")
                all_exist = False
        
        return all_exist
    
    def check_yolo_format(self, images_dir: Path, labels_dir: Path, 
                          is_negative: bool = False) -> Tuple[int, List[str]]:
        """
        Check if dataset follows YOLO format.
        
        Args:
            images_dir: Directory containing images
            labels_dir: Directory containing labels
            is_negative: If True, labels are optional
            
        Returns:
            Tuple of (num_valid, list_of_issues)
        """
        issues = []
        
        if not images_dir.exists():
            issues.append(f"Images directory not found: {images_dir}")
            return 0, issues
        
        # Get image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        images = []
        for ext in image_extensions:
            images.extend(images_dir.glob(f'*{ext}'))
            images.extend(images_dir.glob(f'*{ext.upper()}'))
        
        if len(images) == 0:
            issues.append(f"No images found in {images_dir}")
            return 0, issues
        
        # For negative samples, labels are optional
        if is_negative:
            return len(images), issues
        
        # Check labels
        if not labels_dir.exists():
            issues.append(f"Labels directory not found: {labels_dir}")
            return 0, issues
        
        num_valid = 0
        missing_labels = []
        invalid_labels = []
        
        for img_path in images:
            label_path = labels_dir / f"{img_path.stem}.txt"
            
            if not label_path.exists():
                missing_labels.append(img_path.name)
                continue
            
            # Validate label format
            try:
                with open(label_path, 'r') as f:
                    lines = f.readlines()
                    
                    # Empty label file is valid (background)
                    if len(lines) == 0:
                        num_valid += 1
                        continue
                    
                    # Check each line
                    for line_num, line in enumerate(lines, 1):
                        line = line.strip()
                        if not line:
                            continue
                        
                        parts = line.split()
                        if len(parts) < 5:
                            invalid_labels.append(
                                f"{label_path.name}:{line_num} - "
                                f"Expected 5+ values (class x y w h), got {len(parts)}"
                            )
                            continue
                        
                        # Validate numeric values
                        try:
                            class_id = int(parts[0])
                            x, y, w, h = map(float, parts[1:5])
                            
                            # Check ranges
                            if not (0 <= x <= 1 and 0 <= y <= 1):
                                invalid_labels.append(
                                    f"{label_path.name}:{line_num} - "
                                    f"Center coordinates must be in [0, 1]: x={x}, y={y}"
                                )
                            if not (0 < w <= 1 and 0 < h <= 1):
                                invalid_labels.append(
                                    f"{label_path.name}:{line_num} - "
                                    f"Width/height must be in (0, 1]: w={w}, h={h}"
                                )
                        except ValueError:
                            invalid_labels.append(
                                f"{label_path.name}:{line_num} - "
                                f"Invalid numeric values"
                            )
                    
                    num_valid += 1
                    
            except Exception as e:
                invalid_labels.append(f"{label_path.name} - Error reading: {e}")
        
        # Report issues
        if missing_labels:
            issues.append(f"Missing labels for {len(missing_labels)} images")
            if len(missing_labels) <= 5:
                for name in missing_labels:
                    issues.append(f"  - {name}")
            else:
                for name in missing_labels[:3]:
                    issues.append(f"  - {name}")
                issues.append(f"  ... and {len(missing_labels) - 3} more")
        
        if invalid_labels:
            issues.append(f"Invalid label format in {len(invalid_labels)} cases")
            for issue in invalid_labels[:5]:
                issues.append(f"  - {issue}")
            if len(invalid_labels) > 5:
                issues.append(f"  ... and {len(invalid_labels) - 5} more")
        
        return num_valid, issues
    
    def validate_manifest(self, manifest_path: str) -> bool:
        """
        Validate manifest and all referenced data sources.
        
        Args:
            manifest_path: Path to manifest YAML file
            
        Returns:
            True if manifest is valid
        """
        print(f"\nValidating manifest: {manifest_path}")
        print("-" * 80)
        
        try:
            loader = ManifestLoader(manifest_path, str(self.workspace_root))
            stats = loader.get_statistics()
            
            all_valid = True
            
            for source in stats['sources']:
                print(f"\nSource: {source['path']}")
                print(f"  Weight: {source['weight']}")
                print(f"  Is Negative: {source['is_negative']}")
                
                # Resolve paths
                source_path = loader.resolve_path(source['path'])
                images_dir = source_path / 'images'
                labels_dir = source_path / 'labels'
                
                # Validate YOLO format
                num_valid, issues = self.check_yolo_format(
                    images_dir, 
                    labels_dir,
                    is_negative=source['is_negative']
                )
                
                if issues:
                    print(f"  ✗ Issues found:")
                    for issue in issues:
                        print(f"    {issue}")
                        self.errors.append(f"{source['path']}: {issue}")
                    all_valid = False
                else:
                    print(f"  ✓ Valid: {num_valid} images")
            
            # Print summary
            print("\n" + "=" * 80)
            if all_valid:
                print("✓ Manifest validation PASSED")
                print(f"  Total images: {stats['total_images']}")
                print(f"  Total labels: {stats['total_labels']}")
                print(f"  Weighted dataset size: {stats['weighted_total']:.0f}")
            else:
                print("✗ Manifest validation FAILED")
                print(f"  Errors: {len(self.errors)}")
            print("=" * 80)
            
            return all_valid
            
        except FileNotFoundError as e:
            print(f"\n✗ Error: {e}")
            self.errors.append(str(e))
            return False
        except Exception as e:
            print(f"\n✗ Unexpected error: {e}")
            import traceback
            traceback.print_exc()
            self.errors.append(str(e))
            return False
    
    def create_sample_structure(self):
        """
        Create sample data directory structure for reference.
        """
        print("\nCreating sample data structure...")
        
        sample_sources = [
            'data/raw_xuzhou_2023',
            'data/raw_changzhou_2024',
            'data/raw_negative_samples',
            'data/raw_hard_cases'
        ]
        
        for source in sample_sources:
            source_path = self.workspace_root / source
            images_dir = source_path / 'images'
            labels_dir = source_path / 'labels'
            
            images_dir.mkdir(parents=True, exist_ok=True)
            
            # Only create labels for non-negative samples
            if 'negative' not in source:
                labels_dir.mkdir(parents=True, exist_ok=True)
            
            print(f"  ✓ Created: {source}/")
        
        # Create sample README
        readme_path = self.workspace_root / 'data' / 'README.md'
        if not readme_path.exists():
            with open(readme_path, 'w') as f:
                f.write("""# Data Directory Structure

This directory contains physically separated datasets that are logically mixed via manifests.

## Structure

```
data/
├── raw_xuzhou_2023/       # City-specific data
│   ├── images/            # .jpg, .png images
│   └── labels/            # .txt YOLO format labels
├── raw_changzhou_2024/    # Another city-specific data
│   ├── images/
│   └── labels/
├── raw_negative_samples/  # Background images (no meters)
│   └── images/            # Only images, no labels needed
└── raw_hard_cases/        # Manually curated failure cases
    ├── images/
    └── labels/
```

## YOLO Label Format

Each label file (.txt) has the same name as its corresponding image:
- `image_001.jpg` → `image_001.txt`

Label format (one line per object):
```
<class_id> <x_center> <y_center> <width> <height>
```

All values are normalized to [0, 1]:
- `class_id`: Integer class ID (0-indexed)
- `x_center`, `y_center`: Center of bounding box (0-1)
- `width`, `height`: Box dimensions (0-1)

Example:
```
0 0.5 0.5 0.8 0.4
```

## Negative Samples

The `raw_negative_samples/` directory contains images WITHOUT meters:
- Only requires `images/` folder
- No `labels/` folder needed
- Used to teach model what is NOT a meter (reduces false positives)
""")
            print(f"  ✓ Created: data/README.md")
        
        print("\n✓ Sample structure created!")
        print("  Add your images and labels to these directories.")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Validate data structure and manifest for YOLO training'
    )
    parser.add_argument(
        '--manifest',
        type=str,
        default='datasets/mix_v1_robust.yaml',
        help='Path to dataset manifest YAML'
    )
    parser.add_argument(
        '--create-structure',
        action='store_true',
        help='Create sample data directory structure'
    )
    parser.add_argument(
        '--workspace',
        type=str,
        default=None,
        help='Workspace root directory (defaults to current directory)'
    )
    
    args = parser.parse_args()
    
    # Create validator
    validator = DataValidator(workspace_root=args.workspace)
    
    print("=" * 80)
    print("DATA PREPARATION & VALIDATION")
    print("=" * 80)
    
    # Check directory structure
    if not validator.check_directory_structure():
        print("\n⚠ Some required directories are missing.")
        print("  Run with --create-structure to create sample structure.")
    
    # Create sample structure if requested
    if args.create_structure:
        validator.create_sample_structure()
        return
    
    # Validate manifest
    manifest_valid = validator.validate_manifest(args.manifest)
    
    # Exit with appropriate code
    if manifest_valid:
        print("\n✓ Data validation complete! Ready to train.")
        sys.exit(0)
    else:
        print("\n✗ Data validation failed. Please fix errors and try again.")
        print("\nCommon issues:")
        print("  - Missing images/ or labels/ directories")
        print("  - Label files don't match image filenames")
        print("  - Invalid YOLO format (values must be normalized to [0, 1])")
        print("  - Missing data sources referenced in manifest")
        sys.exit(1)


if __name__ == '__main__':
    main()
