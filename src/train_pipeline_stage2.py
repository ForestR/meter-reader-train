#!/usr/bin/env python3
"""
Pipeline Stage 2 (Digit Position Detection) Training Script
Trains a YOLO detector to localize digit bounding boxes within the dial ROI.
Uses a 2-phase strategy: Phase 1 (Frozen Backbone) and Phase 2 (Unfrozen Fine-tuning).
nc=1 (single class "digit") — position detection only.
"""

import sys
import yaml
import argparse
import shutil
from pathlib import Path
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.manifest_loader import ManifestLoader
from src.data.data_yaml_generator import DataYAMLGenerator

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False
    print("⚠ Warning: Ultralytics not installed. Install with: pip install ultralytics")


class Stage2DigitTrainer:
    """
    Orchestrates Pipeline Stage 2 (digit position detection) training with phased strategy.
    Produces a single-class detector for digit bounding boxes within dial ROI.
    """

    def __init__(self,
                 manifest_path: str = 'datasets/mix_v1_stage2.yaml',
                 workspace_root: str = None):
        """
        Initialize the trainer.

        Args:
            manifest_path: Path to dataset manifest YAML
            workspace_root: Root directory for the project
        """
        self.manifest_path = Path(manifest_path)
        if workspace_root is None:
            self.workspace_root = Path.cwd()
        else:
            self.workspace_root = Path(workspace_root)

        # Paths
        self.configs_dir = self.workspace_root / 'configs'
        self.stage_config_dir = self.configs_dir / 'pipeline' / 'stage2_digit'
        self.checkpoints_dir = self.workspace_root / 'checkpoints' / 'pipeline' / 'stage2_digit'
        self.results_dir = self.workspace_root / 'results'

        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        print("=" * 80)
        print("PIPELINE STAGE 2: DIGIT POSITION DETECTION TRAINER")
        print("=" * 80)
        print(f"Workspace: {self.workspace_root}")
        print(f"Manifest: {self.manifest_path}")
        print()

    def prepare_data(self):
        """Prepare data configuration from manifest."""
        print("STEP 1: Preparing Data Configuration")
        print("-" * 80)

        loader = ManifestLoader(str(self.manifest_path), str(self.workspace_root))
        loader.print_statistics()

        print("\nGenerating Ultralytics data configuration...")
        generator = DataYAMLGenerator(loader)
        data_yaml_path = self.stage_config_dir / 'data.yaml'
        data_yaml_path.parent.mkdir(parents=True, exist_ok=True)
        config = generator.generate_for_pipeline_stage2(str(data_yaml_path))

        print()
        return str(data_yaml_path)

    def train_phase1(self, data_yaml: str, dry_run: bool = False):
        """Execute Phase 1 training: Head Adaptation with frozen backbone."""
        print("=" * 80)
        print("PHASE 1: HEAD ADAPTATION (Frozen Backbone)")
        print("=" * 80)
        print("Strategy: Freeze backbone, train only Neck + Head")
        print("Goal: Rapidly adapt to digit position detection within dial ROI")
        print()

        if not ULTRALYTICS_AVAILABLE:
            print("✗ Cannot train: Ultralytics not installed")
            return None

        config_path = self.stage_config_dir / 'train_phase1.yaml'
        with open(config_path, 'r') as f:
            train_config = yaml.safe_load(f)

        train_config['data'] = data_yaml
        train_config['project'] = str(self.workspace_root / 'runs' / 'detect')

        print(f"Configuration: {config_path}")
        print(f"Model: {train_config['model']}")
        print(f"Epochs: {train_config['epochs']}")
        print(f"Batch Size: {train_config['batch']}")
        print(f"Frozen Layers: {train_config['freeze']}")
        print(f"Learning Rate: {train_config['lr0']}")
        print(f"Rotation Augmentation: ±{train_config['degrees']}°")
        print()

        if dry_run:
            print("✓ Dry run: Configuration validated")
            return None

        print("Starting Phase 1 training...")
        print()

        model = YOLO(train_config['model'])
        results = model.train(
            data=train_config['data'],
            epochs=train_config['epochs'],
            batch=train_config['batch'],
            imgsz=train_config['imgsz'],
            device=train_config['device'],
            freeze=train_config['freeze'],
            optimizer=train_config['optimizer'],
            lr0=train_config['lr0'],
            lrf=train_config['lrf'],
            momentum=train_config['momentum'],
            weight_decay=train_config['weight_decay'],
            warmup_epochs=train_config['warmup_epochs'],
            warmup_momentum=train_config['warmup_momentum'],
            warmup_bias_lr=train_config['warmup_bias_lr'],
            hsv_h=train_config['hsv_h'],
            hsv_s=train_config['hsv_s'],
            hsv_v=train_config['hsv_v'],
            degrees=train_config['degrees'],
            translate=train_config['translate'],
            scale=train_config['scale'],
            shear=train_config['shear'],
            perspective=train_config['perspective'],
            flipud=train_config['flipud'],
            fliplr=train_config['fliplr'],
            mosaic=train_config['mosaic'],
            mixup=train_config['mixup'],
            val=train_config['val'],
            save=train_config['save'],
            save_period=train_config['save_period'],
            exist_ok=train_config['exist_ok'],
            project=train_config['project'],
            name=train_config['name'],
            verbose=train_config['verbose'],
            plots=train_config['plots'],
            patience=train_config['patience'],
            close_mosaic=train_config['close_mosaic'],
            amp=train_config['amp']
        )

        best_model_path = Path(results.save_dir) / 'weights' / 'best.pt'
        phase1_checkpoint = self.checkpoints_dir / 'phase1_best.pt'
        shutil.copy(best_model_path, phase1_checkpoint)

        print()
        print("=" * 80)
        print(f"✓ Phase 1 Complete!")
        print(f"  Best model: {phase1_checkpoint}")
        print(f"  Training logs: {results.save_dir}")
        print("=" * 80)
        print()

        return str(phase1_checkpoint)

    def train_phase2(self, phase1_model: str, data_yaml: str, dry_run: bool = False):
        """Execute Phase 2 training: Deep Fine-tuning with unfrozen layers."""
        print("=" * 80)
        print("PHASE 2: DEEP FINE-TUNING (Unfrozen All Layers)")
        print("=" * 80)
        print("Strategy: Unfreeze all layers, lower learning rate")
        print("Goal: Learn domain-specific digit features")
        print()

        if not ULTRALYTICS_AVAILABLE:
            print("✗ Cannot train: Ultralytics not installed")
            return None

        config_path = self.stage_config_dir / 'train_phase2.yaml'
        with open(config_path, 'r') as f:
            train_config = yaml.safe_load(f)

        train_config['model'] = phase1_model
        train_config['data'] = data_yaml
        train_config['project'] = str(self.workspace_root / 'runs' / 'detect')

        print(f"Configuration: {config_path}")
        print(f"Model: {phase1_model}")
        print(f"Epochs: {train_config['epochs']}")
        print(f"Batch Size: {train_config['batch']}")
        print(f"Frozen Layers: {train_config['freeze']} (all unfrozen)")
        print(f"Learning Rate: {train_config['lr0']} (10% of Phase 1)")
        print(f"Rotation Augmentation: ±{train_config['degrees']}°")
        print()

        if dry_run:
            print("✓ Dry run: Configuration validated")
            return None

        print("Starting Phase 2 training...")
        print()

        model = YOLO(phase1_model)
        results = model.train(
            data=train_config['data'],
            epochs=train_config['epochs'],
            batch=train_config['batch'],
            imgsz=train_config['imgsz'],
            device=train_config['device'],
            freeze=train_config['freeze'],
            optimizer=train_config['optimizer'],
            lr0=train_config['lr0'],
            lrf=train_config['lrf'],
            momentum=train_config['momentum'],
            weight_decay=train_config['weight_decay'],
            warmup_epochs=train_config['warmup_epochs'],
            warmup_momentum=train_config['warmup_momentum'],
            warmup_bias_lr=train_config['warmup_bias_lr'],
            hsv_h=train_config['hsv_h'],
            hsv_s=train_config['hsv_s'],
            hsv_v=train_config['hsv_v'],
            degrees=train_config['degrees'],
            translate=train_config['translate'],
            scale=train_config['scale'],
            shear=train_config['shear'],
            perspective=train_config['perspective'],
            flipud=train_config['flipud'],
            fliplr=train_config['fliplr'],
            mosaic=train_config['mosaic'],
            mixup=train_config['mixup'],
            val=train_config['val'],
            save=train_config['save'],
            save_period=train_config['save_period'],
            exist_ok=train_config['exist_ok'],
            project=train_config['project'],
            name=train_config['name'],
            verbose=train_config['verbose'],
            plots=train_config['plots'],
            patience=train_config['patience'],
            close_mosaic=train_config['close_mosaic'],
            amp=train_config['amp'],
            resume=train_config.get('resume', False)
        )

        best_model_path = Path(results.save_dir) / 'weights' / 'best.pt'
        final_checkpoint = self.checkpoints_dir / 'phase2_best.pt'
        shutil.copy(best_model_path, final_checkpoint)
        final_model = self.checkpoints_dir / 'final.pt'
        shutil.copy(best_model_path, final_model)

        print()
        print("=" * 80)
        print(f"✓ Phase 2 Complete!")
        print(f"  Best model: {final_checkpoint}")
        print(f"  Final model: {final_model}")
        print(f"  Training logs: {results.save_dir}")
        print("=" * 80)
        print()

        return str(final_checkpoint)

    def save_training_report(self, phase1_model: str, phase2_model: str):
        """Save a summary report of the training process."""
        report = {
            'training_date': datetime.now().isoformat(),
            'manifest': str(self.manifest_path),
            'workspace': str(self.workspace_root),
            'stage': 'pipeline_stage2_digit',
            'phase1': {
                'model': phase1_model,
                'config': 'configs/pipeline/stage2_digit/train_phase1.yaml'
            },
            'phase2': {
                'model': phase2_model,
                'config': 'configs/pipeline/stage2_digit/train_phase2.yaml'
            },
            'final_model': str(self.checkpoints_dir / 'final.pt')
        }

        report_path = self.results_dir / 'training_report.yaml'
        with open(report_path, 'w') as f:
            yaml.dump(report, f, default_flow_style=False)

        print(f"✓ Training report saved: {report_path}")

    def run(self, dry_run: bool = False, phase1_only: bool = False):
        """Execute complete Stage 2 training pipeline."""
        try:
            data_yaml = self.prepare_data()
            phase1_model = self.train_phase1(data_yaml, dry_run)

            if dry_run:
                print("\n✓ Dry run complete! Ready to train.")
                return

            if phase1_only:
                print("\n✓ Phase 1 training complete!")
                return

            phase2_model = self.train_phase2(phase1_model, data_yaml, dry_run)
            self.save_training_report(phase1_model, phase2_model)

            print()
            print("=" * 80)
            print("🎉 STAGE 2 TRAINING COMPLETE!")
            print("=" * 80)
            print(f"Final model: {self.checkpoints_dir / 'final.pt'}")
            print(f"Checkpoints: {self.checkpoints_dir}")
            print(f"Results: {self.results_dir}")
            print()
            print("Next steps:")
            print("  1. Evaluate model on test set")
            print("  2. Update configs/model_topology.yaml with final model path")
            print("  3. Train Stage 3 (digit classification) or deploy to edge devices")
            print("=" * 80)

        except FileNotFoundError as e:
            print(f"\n✗ Error: {e}")
            print("\nPlease run scripts/prepare_stage2_data.py first. See README_TRAINING.md for setup.")
            sys.exit(1)
        except Exception as e:
            print(f"\n✗ Training failed: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Train Pipeline Stage 2 (digit position detection) with phased strategy'
    )
    parser.add_argument(
        '--manifest',
        type=str,
        default='datasets/mix_v1_stage2.yaml',
        help='Path to dataset manifest YAML'
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
    parser.add_argument(
        '--phase1-only',
        action='store_true',
        help='Only run Phase 1 training'
    )

    args = parser.parse_args()

    trainer = Stage2DigitTrainer(
        manifest_path=args.manifest,
        workspace_root=args.workspace
    )

    trainer.run(
        dry_run=args.dry_run,
        phase1_only=args.phase1_only
    )


if __name__ == '__main__':
    main()
