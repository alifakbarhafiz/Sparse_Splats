"""Create narrow-view subsets of the Lego dataset for sparse-view experiments."""

from __future__ import annotations

import json
import random
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set


def _load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, data: Dict) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=4)


def _normalize_frame_name(file_path: str) -> str:
    return Path(file_path.replace("./", "")).stem


def _choose_views(
    frames: List[Dict],
    view_count: Optional[int],
    *,
    indices: Optional[List[int]] = None,
    names: Optional[List[str]] = None,
    strategy: str = "uniform",
    seed: Optional[int] = None,
) -> List[str]:
    frame_names = [_normalize_frame_name(frame["file_path"]) for frame in frames]

    if names:
        normalized = []
        for raw in names:
            normalized.append(Path(raw).stem)
        return list(dict.fromkeys(normalized))

    if indices:
        selected = []
        for idx in indices:
            if 0 <= idx < len(frame_names):
                selected.append(frame_names[idx])
        return list(dict.fromkeys(selected))

    if view_count is None or view_count >= len(frame_names):
        return frame_names

    if strategy == "random":
        rnd = random.Random(seed)
        return rnd.sample(frame_names, min(view_count, len(frame_names)))

    # uniform spacing
    step = len(frame_names) / view_count
    selected_indices = [
        min(int(i * step), len(frame_names) - 1) for i in range(view_count)
    ]
    selected = []
    seen = set()
    for idx in selected_indices:
        if idx in seen:
            idx = min(idx + 1, len(frame_names) - 1)
        seen.add(idx)
        selected.append(frame_names[idx])
    return selected


def _copy_images(
    frames: Iterable[Dict],
    raw_dir: Path,
    subset_dir: Path,
    extension: str,
) -> None:
    copied: Set[Path] = set()
    for frame in frames:
        rel_path = Path(frame["file_path"].replace("./", ""))
        src = raw_dir / rel_path.with_suffix(extension)
        dest = subset_dir / rel_path.with_suffix(extension)
        if dest in copied:
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
        copied.add(dest)


def create_view_subset(
    raw_data_dir: Path,
    subset_dir: Path,
    view_count: Optional[int] = None,
    *,
    selection: Optional[Dict] = None,
    extension: str = ".png",
    full_test_set: bool = True,
) -> Dict[str, Optional[str]]:
    """Create a sparse-view subset. When full_test_set is True (default), train
    uses only selected views but test/val keep the full lists so all experiments
    are evaluated on the same held-out test set (PSNR comparable across view counts).
    """
    raw_data_dir = Path(raw_data_dir)
    subset_dir = Path(subset_dir)

    if subset_dir.exists():
        shutil.rmtree(subset_dir)
    subset_dir.mkdir(parents=True, exist_ok=True)

    selection = selection or {}
    indices = selection.get("indices")
    names = selection.get("names")
    strategy = selection.get("strategy", "uniform")
    seed = selection.get("seed")
    view_count = view_count or selection.get("view_count")

    train_path = raw_data_dir / "transforms_train.json"
    if not train_path.exists():
        raise FileNotFoundError(f"Transform file not found: {train_path}")

    base_frames = _load_json(train_path)["frames"]
    selected_views = _choose_views(
        base_frames,
        view_count,
        indices=indices,
        names=names,
        strategy=strategy,
        seed=seed,
    )
    selected_set = set(selected_views)

    transform_files = [train_path]
    for suffix in ("val", "test"):
        candidate = raw_data_dir / f"transforms_{suffix}.json"
        if candidate.exists():
            transform_files.append(candidate)

    summary: Dict[str, int] = {}
    frames_to_copy: List[Dict] = []

    for tf_path in transform_files:
        data = _load_json(tf_path)
        is_train = tf_path.name == "transforms_train.json"
        use_full = full_test_set and not is_train
        if use_full:
            frames = list(data["frames"])
        else:
            frames = [
                frame
                for frame in data["frames"]
                if _normalize_frame_name(frame["file_path"]) in selected_set
            ]
        data["frames"] = frames
        target = subset_dir / tf_path.name
        _write_json(target, data)
        summary[target.name] = len(frames)
        frames_to_copy.extend(frames)

    _copy_images(frames_to_copy, raw_data_dir, subset_dir, extension)

    return {
        "subset_dir": str(subset_dir),
        "view_count": len(selected_views),
        "selected_views": selected_views,
        "strategy": strategy,
        "selection_indices": indices,
        "selection_names": names,
        "files": summary,
    }


def create_subset_views(
    raw_data_dir: str,
    output_dir: str,
    view_count: Optional[int] = None,
    *,
    selection: Optional[Dict] = None,
    extension: str = ".png",
    full_test_set: bool = True,
) -> Dict[str, Optional[str]]:
    return create_view_subset(
        Path(raw_data_dir),
        Path(output_dir),
        view_count,
        selection=selection,
        extension=extension,
        full_test_set=full_test_set,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Subset a dataset to a handful of views."
    )
    parser.add_argument("--raw-data-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--view-count", type=int, default=3)
    parser.add_argument("--indices", nargs="+", type=int)
    parser.add_argument("--names", nargs="+", type=str)
    parser.add_argument("--strategy", choices=["uniform", "random"], default="uniform")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--extension", type=str, default=".png")
    parser.add_argument("--no-full-test-set", action="store_true", help="Filter test/val to selected views (metrics not comparable across view counts)")
    args = parser.parse_args()

    selection = {
        "indices": args.indices,
        "names": args.names,
        "strategy": args.strategy,
        "seed": args.seed,
    }
    metadata = create_subset_views(
        args.raw_data_dir,
        args.output_dir,
        view_count=args.view_count,
        selection=selection,
        extension=args.extension,
        full_test_set=not args.no_full_test_set,
    )
    print(
        f"Subset ready at {metadata['subset_dir']}: {metadata['view_count']} views, "
        f"copied frames per transform: {metadata['files']}"
    )
