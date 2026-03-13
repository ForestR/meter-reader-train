#!/usr/bin/env python3
"""
Pipeline Stage 3 (Digit Classification) Training Script
Trains a YOLO classification model to recognize digit values (0-9) from cropped digit images.
Single-phase training. Uses yolo26n-cls backbone, imgsz=80.
"""

import sys
import yaml
import argparse
import shutil
from pathlib import Path
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False
    print("⚠ Warning: Ultralytics not installed. Install with: pip install ultralytics")


class Stage3ClsTrainer:
    """
    Orchestrates Pipeline Stage 3 (digit classification) training.
    Classifies each digit crop into class 0-9.
    """

    def __init__(self,
                 data_root: str = 'data/digit_crops',
                 workspace_root: str = None):
        """
        Initialize the trainer.

        Args:
            data_root: Path to digit crops directory (train/{0-9}/, val/{0-9}/)
            workspace_root: Root directory for the project
        """
        if workspace_root is None:
            self.workspace_root = Path.cwd()
        else:
            self.workspace_root = Path(workspace_root)

        self.data_root = self.workspace_root / data_root

        # Paths
        self.configs_dir = self.workspace_root / 'configs'
        self.stage_config_dir = self.configs_dir / 'pipeline' / 'stage3_cls'
        self.checkpoints_dir = self.workspace_root / 'checkpoints' / 'pipeline' / 'stage3_cls'
        self.results_dir = self.workspace_root / 'results'

        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        print("=" * 80)
        print("PIPELINE STAGE 3: DIGIT CLASSIFICATION TRAINER")
        print("=" * 80)
        print(f"Workspace: {self.workspace_root}")
        print(f"Data root: {self.data_root}")
        print()

    def prepare_data(self):
        """Verify digit crops exist and write data.yaml for classification."""
        print("STEP 1: Preparing Data Configuration")
        print("-" * 80)

        train_dir = self.data_root / 'train'
        val_dir = self.data_root / 'val'

        if not train_dir.exists():
            raise FileNotFoundError(
                f"Training data not found: {train_dir}\n"
                "Please run scripts/prepare_stage3_data.py first."
            )
        if not val_dir.exists():
            raise FileNotFoundError(
                f"Validation data not found: {val_dir}\n"
                "Please run scripts/prepare_stage3_data.py first."
            )

        # Verify class folders 0-9 exist
        for cls in range(10):
            if not (train_dir / str(cls)).exists():
                raise FileNotFoundError(
                    f"Class folder not found: {train_dir / str(cls)}\n"
                    "Please run scripts/prepare_stage3_data.py first."
                )

        # Write data.yaml for Ultralytics classification
        data_config = {
            'path': str(self.data_root.absolute()),
            'train': 'train',
            'val': 'val',
            'names': {i: str(i) for i in range(10)}
        }

        data_yaml_path = self.stage_config_dir / 'data.yaml'
        data_yaml_path.parent.mkdir(parents=True, exist_ok=True)
        with open(data_yaml_path, 'w') as f:
            yaml.dump(data_config, f, default_flow_style=False, sort_keys=False)

        print(f"✓ Data configuration: {data_yaml_path}")
        print(f"  Train: {train_dir}")
        print(f"  Val: {val_dir}")
        print(f"  Classes: 0-9")
        print()

        return str(data_yaml_path)

    def train(self, data_yaml: str, dry_run: bool = False):
        """Execute classification training."""
        print("=" * 80)
        print("TRAINING: Digit Classification (0-9)")
        print("=" * 80)
        print("Model: yolo26n-cls, imgsz=80")
        print()

        if not ULTRALYTICS_AVAILABLE:
            print("✗ Cannot train: Ultralytics not installed")
            return None

        config_path = self.stage_config_dir / 'train.yaml'
        with open(config_path, 'r') as f:
            train_config = yaml.safe_load(f)

        # Classification expects a directory path (train/val with class subdirs), not a YAML file
        train_config['data'] = str(self.data_root)
        train_config['project'] = str(self.workspace_root / 'runs' / 'classify')

        print(f"Configuration: {config_path}")
        print(f"Model: {train_config['model']}")
        print(f"Epochs: {train_config['epochs']}")
        print(f"Batch Size: {train_config['batch']}")
        print(f"Image Size: {train_config['imgsz']}")
        print()

        if dry_run:
            print("✓ Dry run: Configuration validated")
            return None

        print("Starting classification training...")
        print()

        model = YOLO(train_config['model'])
        results = model.train(
            data=train_config['data'],
            epochs=train_config['epochs'],
            batch=train_config['batch'],
            imgsz=train_config['imgsz'],
            device=train_config['device'],
            optimizer=train_config['optimizer'],
            lr0=train_config['lr0'],
            lrf=train_config['lrf'],
            momentum=train_config['momentum'],
            weight_decay=train_config['weight_decay'],
            warmup_epochs=train_config['warmup_epochs'],
            warmup_momentum=train_config['warmup_momentum'],
            warmup_bias_lr=train_config['warmup_bias_lr'],
            val=train_config['val'],
            save=train_config['save'],
            save_period=train_config['save_period'],
            exist_ok=train_config['exist_ok'],
            project=train_config['project'],
            name=train_config['name'],
            verbose=train_config['verbose'],
            plots=train_config['plots'],
            patience=train_config['patience'],
            amp=train_config['amp']
        )

        best_model_path = Path(results.save_dir) / 'weights' / 'best.pt'
        final_model = self.checkpoints_dir / 'final.pt'
        shutil.copy(best_model_path, final_model)

        print()
        print("=" * 80)
        print(f"✓ Training Complete!")
        print(f"  Best model: {best_model_path}")
        print(f"  Final model: {final_model}")
        print(f"  Training logs: {results.save_dir}")
        print("=" * 80)
        print()

        return str(final_model)

    def save_training_report(self, final_model: str):
        """Save a summary report of the training process."""
        report = {
            'training_date': datetime.now().isoformat(),
            'data_root': str(self.data_root),
            'workspace': str(self.workspace_root),
            'stage': 'pipeline_stage3_cls',
            'config': 'configs/pipeline/stage3_cls/train.yaml',
            'final_model': final_model
        }

        report_path = self.results_dir / 'training_report.yaml'
        with open(report_path, 'w') as f:
            yaml.dump(report, f, default_flow_style=False)

        print(f"✓ Training report saved: {report_path}")

    def run(self, dry_run: bool = False):
        """Execute complete Stage 3 training pipeline."""
        try:
            data_yaml = self.prepare_data()
            final_model = self.train(data_yaml, dry_run)

            if dry_run:
                print("\n✓ Dry run complete! Ready to train.")
                return

            self.save_training_report(final_model)

            print()
            print("=" * 80)
            print("🎉 STAGE 3 TRAINING COMPLETE!")
            print("=" * 80)
            print(f"Final model: {self.checkpoints_dir / 'final.pt'}")
            print(f"Checkpoints: {self.checkpoints_dir}")
            print(f"Results: {self.results_dir}")
            print()
            print("Next steps:")
            print("  1. Evaluate model on test set")
            print("  2. Update configs/model_topology.yaml with final model path")
            print("  3. Assemble 3-stage pipeline for inference (5-8 digit reading)")
            print("=" * 80)

        except FileNotFoundError as e:
            print(f"\n✗ Error: {e}")
            print("\nPlease run scripts/prepare_stage3_data.py first. See README_TRAINING.md for setup.")
            sys.exit(1)
        except Exception as e:
            print(f"\n✗ Training failed: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Train Pipeline Stage 3 (digit classification)'
    )
    parser.add_argument(
        '--data-root',
        type=str,
        default='data/digit_crops',
        help='Path to digit crops directory (train/{0-9}/, val/{0-9}/)'
    )
    parser.add_argument(
        '--workspace',
        type=str,
        default=None,
        help='Workspace root directory (defaults to current directory)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Validate configuration without training'
    )

    args = parser.parse_args()

    trainer = Stage3ClsTrainer(
        data_root=args.data_root,
        workspace_root=args.workspace
    )

    trainer.run(dry_run=args.dry_run)


if __name__ == '__main__':
    main()
