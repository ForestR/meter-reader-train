"""
Generate Ultralytics-compatible data YAML files from manifest configuration.
"""

import os
import yaml
from pathlib import Path
from typing import List, Dict, Optional
from .manifest_loader import ManifestLoader


class DataYAMLGenerator:
    """
    Generates Ultralytics-compatible data.yaml files from manifest configuration.
    """
    
    def __init__(self, manifest_loader: ManifestLoader):
        """
        Initialize the generator.
        
        Args:
            manifest_loader: ManifestLoader instance with loaded manifest
        """
        self.loader = manifest_loader
        self.workspace_root = manifest_loader.workspace_root
    
    def write_image_list_file(self, image_paths: List[str], output_path: str):
        """
        Write a text file containing image paths (one per line).
        
        Args:
            image_paths: List of absolute image paths
            output_path: Path to output text file
        """
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            for img_path in image_paths:
                f.write(f"{img_path}\n")
    
    def generate_data_yaml(self, 
                          output_path: str,
                          num_classes: int = 1,
                          class_names: Optional[List[str]] = None,
                          val_split: float = 0.2,
                          seed: int = 42) -> Dict:
        """
        Generate a complete Ultralytics data.yaml configuration.
        
        Args:
            output_path: Path where data.yaml will be saved
            num_classes: Number of classes in the dataset
            class_names: List of class names (defaults to ['meter_display'])
            val_split: Fraction of data for validation
            seed: Random seed for train/val split
            
        Returns:
            Dictionary containing the data configuration
        """
        if class_names is None:
            if num_classes == 1:
                class_names = ['meter_display']
            else:
                class_names = [f'class_{i}' for i in range(num_classes)]
        
        # Generate weighted image lists
        train_images, val_images = self.loader.generate_weighted_image_list(
            val_split=val_split,
            seed=seed
        )
        
        # Write image list files
        output_dir = Path(output_path).parent
        train_list_path = output_dir / 'train_images.txt'
        val_list_path = output_dir / 'val_images.txt'
        
        self.write_image_list_file(train_images, str(train_list_path))
        self.write_image_list_file(val_images, str(val_list_path))
        
        # Create data.yaml configuration
        data_config = {
            'path': str(self.workspace_root.absolute()),
            'train': str(train_list_path.absolute()),
            'val': str(val_list_path.absolute()),
            'nc': num_classes,
            'names': class_names
        }
        
        # Write data.yaml
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            yaml.dump(data_config, f, default_flow_style=False, sort_keys=False)
        
        print(f"✓ Generated data configuration: {output_file}")
        print(f"  - Training images: {len(train_images)}")
        print(f"  - Validation images: {len(val_images)}")
        print(f"  - Classes: {num_classes} ({', '.join(class_names)})")
        
        return data_config
    
    def generate_for_pipeline_stage1(self, output_path: str = 'configs/pipeline/stage1_dial/data.yaml') -> Dict:
        """
        Generate data.yaml for Pipeline Stage 1 (dial segmentation) training.
        Stage 1 segments dial ROI and decimal_section (nc=2).
        
        Args:
            output_path: Path where data.yaml will be saved
            
        Returns:
            Dictionary containing the data configuration
        """
        return self.generate_data_yaml(
            output_path=output_path,
            num_classes=2,
            class_names=['dial', 'decimal_section'],
            val_split=0.2,
            seed=42
        )

    def generate_for_pipeline_stage2(self, output_path: str = 'configs/pipeline/stage2_digit/data.yaml') -> Dict:
        """
        Generate data.yaml for Pipeline Stage 2 (digit position detection) training.
        Stage 2 detects digit bounding boxes within dial ROI as a single class (digit).
        
        Args:
            output_path: Path where data.yaml will be saved
            
        Returns:
            Dictionary containing the data configuration
        """
        return self.generate_data_yaml(
            output_path=output_path,
            num_classes=1,
            class_names=['digit'],
            val_split=0.2,
            seed=42
        )

    def generate_for_pipeline_stage(self,
                                     stage: str,
                                     num_classes: int,
                                     class_names: List[str],
                                     output_path: Optional[str] = None) -> Dict:
        """
        Generate data.yaml for a specific pipeline stage.
        
        Args:
            stage: Pipeline stage ('dial', 'digit', or 'cls')
            num_classes: Number of classes for this stage
            class_names: List of class names
            output_path: Path where data.yaml will be saved (auto-generated if None)
            
        Returns:
            Dictionary containing the data configuration
        """
        if output_path is None:
            output_path = f'configs/data_{stage}.yaml'
        
        return self.generate_data_yaml(
            output_path=output_path,
            num_classes=num_classes,
            class_names=class_names,
            val_split=0.2,
            seed=42
        )


def generate_stage1_data_yaml(manifest_path: str = 'datasets/mix_v1_robust.yaml',
                               output_path: str = 'configs/pipeline/stage1_dial/data.yaml'):
    """
    Convenience function to generate Pipeline Stage 1 data.yaml from manifest.
    
    Args:
        manifest_path: Path to manifest YAML file
        output_path: Path where data.yaml will be saved
    """
    print(f"Generating data configuration from manifest: {manifest_path}")
    print()
    
    try:
        # Load manifest
        loader = ManifestLoader(manifest_path)
        loader.print_statistics()
        
        # Generate data.yaml
        print("\nGenerating Ultralytics data configuration...")
        generator = DataYAMLGenerator(loader)
        config = generator.generate_for_pipeline_stage1(output_path)
        
        print()
        print("✓ Data configuration generated successfully!")
        print(f"  Use this config in training: data={output_path}")
        
        return config
        
    except FileNotFoundError as e:
        print(f"✗ Error: {e}")
        print("\nPlease ensure data directories exist:")
        print("  data/raw_xuzhou_2023/")
        print("  data/raw_changzhou_2024/")
        print("  data/raw_negative_samples/")
        print("  data/raw_hard_cases/")
        return None
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return None


def generate_stage2_data_yaml(manifest_path: str = 'datasets/mix_v1_stage2.yaml',
                               output_path: str = 'configs/pipeline/stage2_digit/data.yaml'):
    """
    Convenience function to generate Pipeline Stage 2 data.yaml from manifest.
    
    Args:
        manifest_path: Path to manifest YAML file
        output_path: Path where data.yaml will be saved
    """
    print(f"Generating data configuration from manifest: {manifest_path}")
    print()
    
    try:
        loader = ManifestLoader(manifest_path)
        loader.print_statistics()
        print("\nGenerating Ultralytics data configuration...")
        generator = DataYAMLGenerator(loader)
        config = generator.generate_for_pipeline_stage2(output_path)
        print()
        print("✓ Data configuration generated successfully!")
        print(f"  Use this config in training: data={output_path}")
        return config
    except FileNotFoundError as e:
        print(f"✗ Error: {e}")
        print("\nPlease run scripts/prepare_stage2_data.py first to create data/basic_stage2")
        return None
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == '__main__':
    generate_stage1_data_yaml()
