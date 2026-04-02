"""
NDJSON → YOLO materialization and multi-source symlink mix (cache-friendly).

If Ultralytics exposes ``convert_ndjson_to_yolo``, it is preferred; otherwise a
fallback parser/materializer is used (segment NDJSON from Ultralytics Platform).
"""

from __future__ import annotations

import asyncio
import json
import os
import random
import shutil
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import yaml

# Optional Ultralytics helper (may be absent in some versions)
try:
    from ultralytics.data.converter import convert_ndjson_to_yolo as _ul_convert_ndjson_to_yolo
except Exception:  # pragma: no cover
    _ul_convert_ndjson_to_yolo = None

DEFAULT_NAMES = ["dial", "decimal_section"]


def _try_ultralytics_convert(ndjson_path: Path, output_path: Path) -> Optional[Path]:
    """Return path to materialized data.yaml if Ultralytics conversion succeeded."""
    if _ul_convert_ndjson_to_yolo is None:
        return None
    try:
        asyncio.run(_ul_convert_ndjson_to_yolo(str(ndjson_path), output_path=str(output_path)))
        stem = ndjson_path.stem
        candidate = output_path / stem / "data.yaml"
        if candidate.is_file():
            return candidate
    except Exception:
        return None
    return None


def _safe_name(fname: str) -> str:
    """Keep basename; avoid path traversal."""
    return Path(fname).name


def _download_file(url: str, dest: Path, timeout: int = 120) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "mega-meter-reader/0.1"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = resp.read()
    dest.write_bytes(data)


def _parse_ndjson_lines(path: Path) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines:
        raise ValueError(f"Empty NDJSON: {path}")
    header = json.loads(lines[0])
    images: List[Dict[str, Any]] = []
    for line in lines[1:]:
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        if rec.get("type") == "image":
            images.append(rec)
    return header, images


def _header_class_names(header: Dict[str, Any]) -> List[str]:
    cn = header.get("class_names") or {}
    if isinstance(cn, dict):
        # keys may be "0", "1"
        out = [""] * len(cn)
        for k, v in cn.items():
            try:
                idx = int(k)
            except (TypeError, ValueError):
                continue
            while len(out) <= idx:
                out.append("")
            out[idx] = str(v)
        return [x for x in out if x]
    return list(DEFAULT_NAMES)


def _materialize_ndjson_fallback(
    ndjson_path: Path,
    output_parent: Path,
    *,
    fraction: Optional[float] = None,
    seed: int = 42,
) -> Path:
    """
    Materialize NDJSON to ``output_parent /<stem>/`` with images/, labels/, data.yaml.

    Returns absolute path to data.yaml.
    """
    import random

    header, images = _parse_ndjson_lines(ndjson_path)
    stem = ndjson_path.stem
    root = output_parent / stem
    names = _header_class_names(header) or DEFAULT_NAMES
    nc = len(names)

    if fraction is not None and 0 < fraction < 1.0:
        random.seed(seed)
        images = random.sample(images, max(1, int(len(images) * fraction)))

    for split in ("train", "val", "test"):
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)

    normalized: List[Tuple[Dict[str, Any], str, Path]] = []
    for rec in images:
        split = rec.get("split") or "train"
        if split not in ("train", "val", "test"):
            split = "train"
        fname = _safe_name(rec["file"])
        url = rec.get("url")
        if not url:
            raise ValueError(f"Image record missing url: {fname}")
        img_path = root / "images" / split / fname
        normalized.append((rec, split, img_path))

    to_download = [(rec["url"], p) for rec, split, p in normalized if not p.is_file()]
    with ThreadPoolExecutor(max_workers=16) as exe:
        futures = {exe.submit(_download_file, url, p): p for url, p in to_download}
        for fut in as_completed(futures):
            fut.result()

    for rec, split, _img_path in normalized:
        fname = _safe_name(rec["file"])
        label_path = root / "labels" / split / (Path(fname).stem + ".txt")
        ann = rec.get("annotations") or {}
        segments = ann.get("segments") or []
        lines_out: List[str] = []
        for seg in segments:
            if not seg:
                continue
            cls_id = int(seg[0])
            rest = seg[1:]
            if len(rest) < 6 or len(rest) % 2 != 1:  # class + pairs
                continue
            coords = [float(x) for x in rest]
            line = f"{cls_id} " + " ".join(f"{c:.6f}" for c in coords)
            lines_out.append(line)
        label_path.write_text("\n".join(lines_out) + ("\n" if lines_out else ""), encoding="utf-8")

    data_yaml = root / "data.yaml"
    cfg = {
        "path": str(root.resolve()),
        "train": "images/train",
        "val": "images/val",
        "nc": nc,
        "names": names,
        "task": "segment",
    }

    with open(data_yaml, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)

    return data_yaml.resolve()


def materialize_ndjson(
    ndjson_path: Path,
    output_parent: Optional[Path] = None,
    *,
    refresh: bool = False,
    fraction: Optional[float] = None,
) -> Path:
    """
    Ensure ``output_parent/<stem>/`` exists with YOLO segment layout + data.yaml.

    Tries Ultralytics ``convert_ndjson_to_yolo`` first; falls back to built-in
    downloader/parser.

    Returns absolute path to ``data.yaml``.
    """
    ndjson_path = ndjson_path.resolve()
    if output_parent is None:
        output_parent = ndjson_path.parent
    else:
        output_parent = Path(output_parent).resolve()

    stem_dir = output_parent / ndjson_path.stem
    if refresh and stem_dir.exists():
        shutil.rmtree(stem_dir)

    if not refresh and (stem_dir / "data.yaml").is_file():
        return (stem_dir / "data.yaml").resolve()

    ul = _try_ultralytics_convert(ndjson_path, output_parent)
    if ul is not None:
        return ul.resolve()

    return _materialize_ndjson_fallback(ndjson_path, output_parent, fraction=fraction)


def _load_mix_policy(mix_yaml: Path) -> List[Tuple[Path, float]]:
    data = yaml.safe_load(mix_yaml.read_text(encoding="utf-8")) or {}
    policy = data.get("train_policy") or []
    out: List[Tuple[Path, float]] = []
    for item in policy:
        src = item.get("source")
        w = float(item.get("weight", 1.0))
        if not src:
            continue
        out.append((Path(src).resolve(), w))
    if not out:
        raise ValueError(f"No sources in mix policy: {mix_yaml}")
    return out


def _symlink_relative(target: Path, link_path: Path) -> None:
    link_path.parent.mkdir(parents=True, exist_ok=True)
    if link_path.is_symlink() or link_path.exists():
        link_path.unlink()
    rel = os.path.relpath(target, start=link_path.parent)
    os.symlink(rel, link_path)


def _list_train_images(stem_root: Path) -> List[Path]:
    """Image paths under ``images/train`` with supported extensions."""
    img_dir = stem_root / "images" / "train"
    if not img_dir.is_dir():
        return []
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return sorted(p for p in img_dir.glob("*") if p.suffix.lower() in exts)


def _allocate_weighted_counts(counts: List[int], weights: List[float]) -> List[int]:
    """
    Allocate integer targets per bucket so sum(targets) == sum(counts), targets[i] <= counts[i],
    with proportions ~ weights / sum(weights).
    """
    if not counts:
        return []
    n_total = sum(counts)
    if n_total == 0:
        return [0] * len(counts)
    w = [max(0.0, float(x)) for x in weights]
    w_sum = sum(w)
    if w_sum <= 0:
        w = [1.0] * len(counts)
        w_sum = float(len(counts))
    raw = [n_total * w[i] / w_sum for i in range(len(counts))]
    t = [min(counts[i], int(r)) for i, r in enumerate(raw)]
    deficit = n_total - sum(t)
    while deficit > 0:
        best_i = max(
            range(len(counts)),
            key=lambda i: (counts[i] - t[i], raw[i] - int(raw[i])),
        )
        if counts[best_i] - t[best_i] <= 0:
            break
        t[best_i] += 1
        deficit -= 1
    surplus = sum(t) - n_total
    while surplus > 0:
        worst_i = max(range(len(t)), key=lambda i: t[i])
        if t[worst_i] <= 0:
            break
        t[worst_i] -= 1
        surplus -= 1
    return t


def build_symlink_mix(
    mix_yaml: Path,
    *,
    workspace_root: Optional[Path] = None,
    fraction: Optional[float] = None,
    seed: int = 42,
) -> Path:
    """
    Prefetch each NDJSON in the mix policy, then build
    ``<mix_yaml_dir>/<mix_stem>/`` with symlinked images/labels and data.yaml.

    Train-split images are subsampled per source using ``train_policy`` weights; val/test
    splits include all images from each materialized source.

    Returns absolute path to mix ``data.yaml``.
    """
    mix_yaml = mix_yaml.resolve()
    mix_dir = mix_yaml.parent / mix_yaml.stem
    # Always rebuild the combined view (symlinks only); per-source trees stay cached.
    if mix_dir.exists():
        shutil.rmtree(mix_dir)

    if workspace_root is None:
        workspace_root = mix_yaml.parent.parent.parent  # datasets/ -> repo root heuristic
        if not (workspace_root / "data").is_dir():
            workspace_root = Path.cwd()
    _ = workspace_root  # reserved for future dataset_output_dir override

    sources = _load_mix_policy(mix_yaml)
    names_ref: Optional[List[str]] = None
    nc = 2
    materialized: List[Tuple[Path, Path, float]] = []  # (ndjson_path, stem_root, weight)

    for ndjson_path, weight in sources:
        dy = materialize_ndjson(
            ndjson_path,
            output_parent=ndjson_path.parent,
            refresh=False,
            fraction=fraction,
        )
        with open(dy, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        if names_ref is None:
            names_ref = list(cfg.get("names") or DEFAULT_NAMES)
            nc = int(cfg.get("nc") or len(names_ref))
        else:
            other = list(cfg.get("names") or [])
            if other != names_ref:
                raise ValueError(
                    f"Class names mismatch: {names_ref} vs {other} for {ndjson_path}"
                )
        # Ultralytics NDJSON converter copies the header into data.yaml and often omits ``path``.
        raw_root = cfg.get("path")
        if raw_root:
            stem_root = Path(raw_root)
            stem_root = (
                stem_root.resolve()
                if stem_root.is_absolute()
                else (dy.parent / stem_root).resolve()
            )
        else:
            stem_root = dy.parent.resolve()
        materialized.append((ndjson_path, stem_root, weight))

    rng = random.Random(seed)

    def _symlink_one(
        src_tag: str,
        stem_root: Path,
        img_file: Path,
        split: str,
    ) -> None:
        mix_img = mix_dir / "images" / split
        mix_lbl = mix_dir / "labels" / split
        lbl_dir = stem_root / "labels" / split
        unique = f"{src_tag}__{img_file.name}"
        _symlink_relative(img_file, mix_img / unique)
        lbl = lbl_dir / (img_file.stem + ".txt")
        if lbl.is_file():
            _symlink_relative(lbl, mix_lbl / (Path(unique).stem + ".txt"))
        else:
            (mix_lbl / (Path(unique).stem + ".txt")).write_text("", encoding="utf-8")

    # Weighted subsampling for train split only; val/test keep all images.
    train_pools: List[List[Path]] = []
    weights_list: List[float] = []
    meta: List[Tuple[Path, Path]] = []
    for ndjson_path, stem_root, w in materialized:
        train_pools.append(_list_train_images(stem_root))
        weights_list.append(w)
        meta.append((ndjson_path, stem_root))

    counts = [len(p) for p in train_pools]
    targets = _allocate_weighted_counts(counts, weights_list)

    for i, (ndjson_path, stem_root) in enumerate(meta):
        src_tag = ndjson_path.stem
        pool = list(train_pools[i])
        k = targets[i] if i < len(targets) else len(pool)
        k = min(k, len(pool))
        if k < len(pool):
            chosen = set(rng.sample(pool, k))
        else:
            chosen = set(pool)
        for img_file in sorted(pool):
            if img_file in chosen:
                _symlink_one(src_tag, stem_root, img_file, "train")

    for ndjson_path, stem_root in meta:
        src_tag = ndjson_path.stem
        for split in ("val", "test"):
            img_dir = stem_root / "images" / split
            lbl_dir = stem_root / "labels" / split
            if not img_dir.is_dir():
                continue
            for img_file in sorted(img_dir.glob("*")):
                if img_file.suffix.lower() not in (
                    ".jpg",
                    ".jpeg",
                    ".png",
                    ".bmp",
                    ".webp",
                ):
                    continue
                _symlink_one(src_tag, stem_root, img_file, split)

    out_yaml = mix_dir / "data.yaml"
    cfg = {
        "path": str(mix_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "nc": nc,
        "names": names_ref or DEFAULT_NAMES,
        "task": "segment",
    }
    with open(out_yaml, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)

    return out_yaml.resolve()


def refresh_materialization(
    mix_yaml: Optional[Path],
    ndjson_paths: Sequence[Path],
) -> None:
    """Remove per-source materialization dirs and optional mix dir (per skill doc)."""
    if mix_yaml is not None:
        mix_yaml = mix_yaml.resolve()
        mix_dir = mix_yaml.parent / mix_yaml.stem
        if mix_dir.exists():
            shutil.rmtree(mix_dir)
    for p in ndjson_paths:
        p = p.resolve()
        stem_dir = p.parent / p.stem
        if stem_dir.exists():
            shutil.rmtree(stem_dir)
