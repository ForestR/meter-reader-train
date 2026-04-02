"""CLI: train Stage 1 YOLO-seg from NDJSON or symlink-mix YAML."""

from __future__ import annotations

import argparse
import os
import random
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None  # type: ignore[misc, assignment]

try:
    from ultralytics.utils import RANK
except ImportError:
    RANK = -1  # type: ignore[misc, assignment]

from mega_meter_reader.stage1.dataset import (
    build_symlink_mix,
    materialize_ndjson,
    refresh_materialization,
)


def _load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _merge_train_kwargs(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for k, v in overrides.items():
        if v is not None:
            out[k] = v
    return out


def _scalar_float(x: Any) -> float:
    if hasattr(x, "item"):
        try:
            return float(x.item())  # torch.Tensor, numpy scalar
        except Exception:
            pass
    return float(x)


def _merge_metrics_dict(trainer: Any) -> Dict[str, Any]:
    """Combine labeled train losses and validation metrics for logging."""
    out: Dict[str, Any] = {}
    if hasattr(trainer, "label_loss_items") and getattr(trainer, "tloss", None) is not None:
        try:
            li = trainer.label_loss_items(trainer.tloss)
            if isinstance(li, dict):
                out.update(li)
        except Exception:
            pass
    m = getattr(trainer, "metrics", None)
    if isinstance(m, dict):
        out.update(m)
    return out


def _register_clean_train_logger(model: Any, log_path: Optional[Path]) -> None:
    """
    One-line-per-epoch summary (stdout + optional file) alongside Ultralytics tqdm output.
    """
    header_written = False

    def _write_header(keys: list[str]) -> None:
        nonlocal header_written
        if log_path is None or header_written:
            return
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "w", encoding="utf-8") as f:
            f.write("epoch," + ",".join(keys) + "\n")
        header_written = True

    def on_fit_epoch_end(trainer: Any) -> None:
        if RANK not in {-1, 0}:
            return
        ep = int(getattr(trainer, "epoch", 0)) + 1
        tot = int(getattr(trainer, "epochs", 0))
        merged = _merge_metrics_dict(trainer)
        box = _scalar_float(merged.get("train/box_loss", 0.0))
        seg = _scalar_float(merged.get("train/seg_loss", 0.0))
        cls_ = _scalar_float(merged.get("train/cls_loss", 0.0))
        map50_m = merged.get("metrics/mAP50(M)", merged.get("metrics/mAP50(B)", 0.0))
        map5095_m = merged.get("metrics/mAP50-95(M)", merged.get("metrics/mAP50-95(B)", 0.0))
        line = (
            f"[{ep:>3}/{tot}] "
            f"train box={box:.4f} seg={seg:.4f} cls={cls_:.4f} | "
            f"val mAP50(M)={_scalar_float(map50_m):.4f} mAP50-95(M)={_scalar_float(map5095_m):.4f}"
        )
        print(line, flush=True)

        if log_path is not None:
            keys = sorted(merged.keys())
            _write_header(keys)

            def _cell(v: Any) -> str:
                if isinstance(v, (int, float)):
                    return str(v)
                if hasattr(v, "item"):
                    try:
                        return str(_scalar_float(v))
                    except Exception:
                        pass
                return str(v)

            vals = [_cell(merged[k]) for k in keys]
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(f"{ep}," + ",".join(str(v) for v in vals) + "\n")

    def on_train_end(trainer: Any) -> None:
        if RANK not in {-1, 0}:
            return
        merged = _merge_metrics_dict(trainer)
        print(f"[done] final metrics keys: {len(merged)} (see results.csv and clean_train log)", flush=True)

    model.add_callback("on_fit_epoch_end", on_fit_epoch_end)
    model.add_callback("on_train_end", on_train_end)


def _resolve_phase_cfg_path(base_config_path: Optional[Path], phase_cfg: Optional[str]) -> Optional[Path]:
    if not phase_cfg:
        return None
    p = Path(phase_cfg).expanduser()
    if p.is_absolute():
        return p
    cwd_candidate = p.resolve()
    if cwd_candidate.exists():
        return cwd_candidate
    if base_config_path is not None:
        base_candidate = (base_config_path.parent / p).resolve()
        if base_candidate.exists():
            return base_candidate
    return cwd_candidate


def _collect_phase_train_kwargs(
    phase_cfg: Dict[str, Any],
    data_yaml: Path,
    project_override: Optional[str],
    name_override: Optional[str],
    device_override: Optional[str],
) -> Dict[str, Any]:
    train_kw: Dict[str, Any] = {"data": str(data_yaml)}
    reserved = {"model", "log_file", "seed"}
    for k, v in phase_cfg.items():
        if k not in reserved and v is not None:
            train_kw[k] = v

    if project_override is not None:
        train_kw["project"] = project_override
    if name_override is not None:
        train_kw["name"] = name_override
    if device_override is not None:
        train_kw["device"] = device_override

    # Force proper tqdm rendering to avoid duplicate progress lines in terminal logs.
    train_kw["verbose"] = True if train_kw.get("verbose") is None else bool(train_kw["verbose"])
    return train_kw


def _best_weight_from_results(results: Any) -> Path:
    save_dir = Path(getattr(results, "save_dir"))
    best = save_dir / "weights" / "best.pt"
    if not best.exists():
        raise FileNotFoundError(f"Expected checkpoint not found: {best}")
    return best


def _copy_checkpoint(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(src, dst)


def _run_phase(
    model_name: str,
    phase_name: str,
    phase_cfg: Dict[str, Any],
    data_yaml: Path,
    project_override: Optional[str],
    run_name_override: Optional[str],
    device_override: Optional[str],
    log_path: Optional[Path],
) -> Path:
    train_kw = _collect_phase_train_kwargs(
        phase_cfg=phase_cfg,
        data_yaml=data_yaml,
        project_override=project_override,
        name_override=run_name_override,
        device_override=device_override,
    )
    print(f"{phase_name} model: {model_name}")
    print(f"{phase_name} train kwargs: {train_kw}")

    model = YOLO(model_name)
    _register_clean_train_logger(model, log_path)
    results = model.train(**train_kw)
    return _best_weight_from_results(results)


def main(argv: Optional[list] = None) -> None:
    parser = argparse.ArgumentParser(description="Train Stage 1 dial segmentation (YOLO-seg)")
    g = parser.add_mutually_exclusive_group(required=False)
    g.add_argument("--data", type=str, help="Path to a single NDJSON dataset file")
    g.add_argument("--mix", type=str, help="Path to mix policy YAML (multiple NDJSON sources)")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="YAML with keys: data or mix, model, epochs, batch, imgsz, project, name, ...",
    )
    parser.add_argument("--model", type=str, default=None, help="Base weights, e.g. yolo11n-seg.pt")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--project", type=str, default=None)
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument(
        "--phase1-only",
        action="store_true",
        help="Run phase 1 only and skip phase 2",
    )
    parser.add_argument(
        "--checkpoints-dir",
        type=str,
        default="checkpoints/stage1",
        help="Directory to store phase checkpoints",
    )
    parser.add_argument(
        "--refresh-data",
        action="store_true",
        help="Delete cached materialized NDJSON dirs and mix dir before build",
    )
    parser.add_argument(
        "--fraction",
        type=float,
        default=None,
        help="Subsample fraction of images per source (0-1), for smoke tests",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed when using --fraction",
    )
    args = parser.parse_args(argv)

    if YOLO is None:
        print("ultralytics is required: pip install ultralytics", file=sys.stderr)
        sys.exit(1)

    os.environ.setdefault("YOLO_AUTOINSTALL", "0")  # suppress update-availability ping

    cfg: Dict[str, Any] = {}
    config_path: Optional[Path] = None
    if args.config:
        config_path = Path(args.config).resolve()
        cfg = _load_yaml(config_path)

    data_yaml: Optional[Path] = None
    mix_path: Optional[Path] = None
    nd_path: Optional[Path] = None

    if args.data:
        nd_path = Path(args.data).resolve()
    elif args.mix:
        mix_path = Path(args.mix).resolve()
    elif cfg.get("data"):
        nd_path = Path(str(cfg["data"])).resolve()
    elif cfg.get("mix"):
        mix_path = Path(str(cfg["mix"])).resolve()
    else:
        parser.error("Provide --data, --mix, or --config with data: or mix:")

    if args.fraction is not None and (args.fraction <= 0 or args.fraction > 1):
        parser.error("--fraction must be in (0, 1]")

    random.seed(args.seed)

    # Resolve data.yaml path
    if mix_path is not None:
        sources = _load_yaml(mix_path).get("train_policy") or []
        nd_list = [Path(str(s["source"])).resolve() for s in sources if s.get("source")]
        if args.refresh_data:
            refresh_materialization(mix_path, nd_list)
        data_yaml = build_symlink_mix(
            mix_path,
            fraction=args.fraction,
            seed=int(cfg.get("seed", args.seed)),
        )
    else:
        assert nd_path is not None
        if args.refresh_data:
            refresh_materialization(None, [nd_path])
        data_yaml = materialize_ndjson(
            nd_path,
            output_parent=nd_path.parent,
            refresh=args.refresh_data,
            fraction=args.fraction,
        )

    model_name = args.model or cfg.get("model") or "yolo11n-seg.pt"
    project = args.project or cfg.get("project")
    checkpoints_dir = Path(args.checkpoints_dir).expanduser()
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    phase1_cfg_path = _resolve_phase_cfg_path(config_path, cfg.get("phase1_config"))
    phase2_cfg_path = _resolve_phase_cfg_path(config_path, cfg.get("phase2_config"))

    if phase1_cfg_path is not None:
        phase1_cfg = _load_yaml(phase1_cfg_path)
    else:
        phase1_cfg = dict(cfg)
        if args.epochs is not None:
            phase1_cfg["epochs"] = args.epochs
        if args.batch is not None:
            phase1_cfg["batch"] = args.batch
        if args.imgsz is not None:
            phase1_cfg["imgsz"] = args.imgsz
        phase1_cfg["task"] = phase1_cfg.get("task", "segment")
        phase1_cfg["verbose"] = True

    if phase2_cfg_path is not None and not args.phase1_only:
        phase2_cfg = _load_yaml(phase2_cfg_path)
    else:
        phase2_cfg = {}

    if "workers" not in phase1_cfg:
        phase1_cfg["workers"] = int(cfg.get("workers", 8))
    if phase2_cfg and "workers" not in phase2_cfg:
        phase2_cfg["workers"] = int(cfg.get("workers", 8))

    default_name = args.name or cfg.get("name") or "train"
    phase1_name = f"{default_name}_phase1"
    phase2_name = f"{default_name}_phase2"

    if phase1_cfg_path is not None and "name" in phase1_cfg and args.name is None:
        phase1_name = str(phase1_cfg["name"])
    if phase2_cfg_path is not None and phase2_cfg and "name" in phase2_cfg and args.name is None:
        phase2_name = str(phase2_cfg["name"])

    print(f"data.yaml: {data_yaml}")
    print(f"phase1 config: {phase1_cfg_path if phase1_cfg_path else 'inline/main config'}")
    if phase2_cfg:
        print(f"phase2 config: {phase2_cfg_path if phase2_cfg_path else 'inline/main config'}")
    else:
        print("phase2 config: disabled")

    phase1_log = (checkpoints_dir / "phase1_clean_train.log").resolve()
    print(f"phase1 clean epoch log: {phase1_log}")
    phase1_best = _run_phase(
        model_name=model_name,
        phase_name="phase1",
        phase_cfg=phase1_cfg,
        data_yaml=data_yaml,
        project_override=project,
        run_name_override=phase1_name if args.name is not None else None,
        device_override=args.device or cfg.get("device"),
        log_path=phase1_log,
    )
    phase1_ckpt = checkpoints_dir / "phase1_best.pt"
    _copy_checkpoint(phase1_best, phase1_ckpt)
    print(f"phase1 best checkpoint: {phase1_ckpt}")

    if args.phase1_only or not phase2_cfg:
        print("[done] phase1-only training complete")
        return

    phase2_model_name = str(phase1_ckpt)
    phase2_log = (checkpoints_dir / "phase2_clean_train.log").resolve()
    print(f"phase2 clean epoch log: {phase2_log}")
    phase2_best = _run_phase(
        model_name=phase2_model_name,
        phase_name="phase2",
        phase_cfg=phase2_cfg,
        data_yaml=data_yaml,
        project_override=project,
        run_name_override=phase2_name if args.name is not None else None,
        device_override=args.device or cfg.get("device"),
        log_path=phase2_log,
    )
    phase2_ckpt = checkpoints_dir / "phase2_best.pt"
    final_ckpt = checkpoints_dir / "final.pt"
    _copy_checkpoint(phase2_best, phase2_ckpt)
    _copy_checkpoint(phase2_best, final_ckpt)
    print(f"phase2 best checkpoint: {phase2_ckpt}")
    print(f"final checkpoint: {final_ckpt}")


if __name__ == "__main__":
    main()
