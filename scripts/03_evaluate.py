"""Render saved models and compute metrics via the Gaussian Splatting helpers."""
from __future__ import annotations

import csv
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


RENDER_SCRIPT = Path(__file__).resolve().parents[1] / "external" / "gaussian-splatting" / "render.py"
METRICS_SCRIPT = Path(__file__).resolve().parents[1] / "external" / "gaussian-splatting" / "metrics.py"
METRICS_COLUMNS = [
    "timestamp",
    "subset_label",
    "view_count",
    "model_dir",
    "method",
    "iteration",
    "psnr",
    "ssim",
    "lpips",
    "selected_views",
]


def _flatten_cli_args(arg_map: Dict[str, Any]) -> List[str]:
    flattened: List[str] = []
    for key in sorted(arg_map.keys()):
        value = arg_map[key]
        if value is None:
            continue
        flag = f"--{key}"
        if isinstance(value, bool):
            if value:
                flattened.append(flag)
            continue
        if isinstance(value, (list, tuple)):
            flattened.append(flag)
            flattened.extend(str(v) for v in value)
            continue
        flattened.extend([flag, str(value)])
    return flattened


def _extract_iteration(method_name: str) -> Optional[int]:
    match = re.search(r"(\d+)", method_name)
    return int(match.group(1)) if match else None


def _normalize_iterations(render_conf: Dict[str, Any]) -> List[int]:
    iterations_field = render_conf.get("iterations")
    if iterations_field is None:
        iterations_field = render_conf.get("iteration")
    if iterations_field is None:
        return []

    if isinstance(iterations_field, (list, tuple)):
        return sorted({int(value) for value in iterations_field})

    return [int(iterations_field)]


def _run_render(model_dir: Path, source_dir: Path, render_conf: Dict[str, Any]) -> None:
    iterations = _normalize_iterations(render_conf)
    if not iterations:
        raise ValueError("render configuration must specify `iterations` or `iteration`.")

    base_command = [
        sys.executable,
        str(RENDER_SCRIPT),
        "-s",
        str(source_dir),
        "-m",
        str(model_dir),
    ]
    extra_args = render_conf.get("args", {})

    for iteration in iterations:
        command = base_command + ["--iteration", str(iteration)]
        command.extend(_flatten_cli_args(extra_args))
        subprocess.run(command, check=True)


def _run_metrics(model_dir: Path, metrics_conf: Dict[str, Any]) -> None:
    command = [sys.executable, str(METRICS_SCRIPT), "-m", str(model_dir)]
    command.extend(_flatten_cli_args(metrics_conf.get("args", {})))
    subprocess.run(command, check=True)


def _load_results(model_dir: Path) -> Dict[str, Dict[str, float]]:
    result_path = model_dir / "results.json"
    if not result_path.exists():
        return {}
    with result_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _rows_from_results(
    results: Dict[str, Dict[str, float]],
    model_dir: Path,
    metadata: Dict[str, Any],
) -> List[Dict[str, Any]]:
    timestamp = metadata.get("timestamp", time.time())
    selected_views = metadata.get("selected_views", [])
    subset_label = metadata.get("subset_label", "")
    view_count = metadata.get("view_count")

    rows: List[Dict[str, Any]] = []
    for method, metrics in results.items():
        rows.append(
            {
                "timestamp": timestamp,
                "subset_label": subset_label,
                "view_count": view_count,
                "model_dir": str(model_dir),
                "method": method,
                "iteration": _extract_iteration(method),
                "psnr": metrics.get("PSNR"),
                "ssim": metrics.get("SSIM"),
                "lpips": metrics.get("LPIPS"),
                "selected_views": ";".join(selected_views),
            }
        )
    return rows


def _append_metrics_csv(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=METRICS_COLUMNS)
        if not file_exists:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)


def evaluate_model(configuration: Dict[str, Any]) -> None:
    source_dir = Path(configuration["source_dir"])
    model_dir = Path(configuration["model_dir"])
    render_conf = configuration["render"]
    metrics_conf = configuration["metrics"]
    metadata = configuration.get("metadata", {})

    _run_render(model_dir, source_dir, render_conf)
    _run_metrics(model_dir, metrics_conf)
    results = _load_results(model_dir)
    if results:
        rows = _rows_from_results(results, model_dir, metadata)
        csv_path = Path(metrics_conf.get("csv", "results/metrics.csv"))
        _append_metrics_csv(csv_path, rows)


if __name__ == "__main__":
    import argparse
    import yaml

    parser = argparse.ArgumentParser(description="Evaluate a Gaussian Splatting run via YAML.")
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    evaluate_model(config)
