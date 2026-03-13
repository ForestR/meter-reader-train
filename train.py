#!/usr/bin/env python3
"""
Unified training entrypoint.
Dispatches to the appropriate training script based on --model.
"""

import sys
import subprocess
from pathlib import Path


MODEL_SCRIPTS = {
    'pipeline-stage1': 'src.train_pipeline_stage1',
    'pipeline-stage2': 'src.train_pipeline_stage2',
    'pipeline-stage3': 'src.train_pipeline_stage3',
    'end2end': 'src.train_end2end_model',
}


def main():
    """Dispatch to the appropriate training module."""
    parser = __import__('argparse').ArgumentParser(
        description='Train meter reading models'
    )
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        choices=list(MODEL_SCRIPTS.keys()),
        help='Model to train: pipeline-stage1, pipeline-stage2, pipeline-stage3, or end2end'
    )
    args, rest = parser.parse_known_args()

    module_name = MODEL_SCRIPTS[args.model]
    cmd = [sys.executable, '-m', module_name] + rest
    sys.exit(subprocess.run(cmd, cwd=Path(__file__).parent).returncode)


if __name__ == '__main__':
    main()
