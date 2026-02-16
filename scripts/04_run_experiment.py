"""Run the full sparse-view experiment pipeline end-to-end."""
from __future__ import annotations

import argparse
import importlib.util
import subprocess
from pathlib import Path
from typing import Dict, List

import yaml


def _load_symbol(script_name: str, symbol: str):
    script_path = Path(__file__).resolve().parent / script_name
    spec = importlib.util.spec_from_file_location(script_name, script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {script_name}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, symbol)


create_view_subset = _load_symbol("01_subset_views.py", "create_view_subset")
train_model = _load_symbol("02_train.py", "train_model")
evaluate_model = _load_symbol("03_evaluate.py", "evaluate_model")


def _load_yaml(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _normalize_list(value):
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [int(v) for v in value]
    return [int(value)]


def _run_single_config(
    config_path: Path,
    *,
    raw_data_dir: Path,
    splits_dir: Path,
    experiments_dir: Path,
    results_csv: Path,
) -> None:
    config = _load_yaml(config_path)
    subset_conf = config.get("subset", {})
    subset_label = config.get("name") or subset_conf.get("name") or f"{subset_conf.get('view_count', 'auto')}_views"
    view_count = subset_conf.get("view_count")
    target_dir = Path(subset_conf.get("output_dir") or splits_dir / f"{view_count}_views")
    raw_dir = Path(subset_conf.get("raw_dir") or raw_data_dir)
    selection = dict(subset_conf.get("selection", {}))
    selection.setdefault("strategy", "uniform")
    selection.setdefault("view_count", view_count)

    subset_meta = create_view_subset(
        raw_dir,
        target_dir,
        view_count,
        selection=selection,
        extension=subset_conf.get("extension", ".png"),
        full_test_set=subset_conf.get("full_test_set", True),
    )

    training_conf = dict(config.get("training", {}))
    args_map = dict(training_conf.get("args", {}))
    training_conf["args"] = args_map
    training_conf["source_dir"] = subset_meta["subset_dir"]
    training_conf.setdefault("model_dir", str(experiments_dir / subset_label))

    train_model(training_conf)

    render_conf = config.get("render")
    if render_conf is None:
        render_conf = {"iterations": _normalize_list(training_conf["args"].get("iterations"))}

    metrics_conf = dict(config.get("metrics", {}))
    metrics_conf.setdefault("csv", str(results_csv))

    evaluate_conf = {
        "source_dir": training_conf["source_dir"],
        "model_dir": training_conf["model_dir"],
        "render": render_conf,
        "metrics": metrics_conf,
        "metadata": {
            "subset_label": subset_label,
            "view_count": subset_meta["view_count"],
            "selected_views": subset_meta["selected_views"],
            "timestamp": None,
        },
    }
    evaluate_model(evaluate_conf)


def run_experiment(args: argparse.Namespace) -> None:
    config_paths: List[Path] = []
    if args.configs:
        config_paths = [Path(path) for path in args.configs]
    else:
        config_paths = sorted(Path(args.config_dir).glob("views_*.yaml"))

    if not config_paths:
        raise FileNotFoundError("No configuration files found for experiments.")

    for config_path in config_paths:
        print(f"Running experiment defined in {config_path}")
        try:
            _run_single_config(
                config_path,
                raw_data_dir=Path(args.raw_data_dir),
                splits_dir=Path(args.splits_dir),
                experiments_dir=Path(args.experiments_dir),
                results_csv=Path(args.results_csv),
            )
        except subprocess.CalledProcessError as exc:
            print(f"Command failed for {config_path}: {exc}")
            continue


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Lego sparse-view experiments.")
    parser.add_argument("--config-dir", type=str, default="configs")
    parser.add_argument("--configs", nargs="+", help="Explicit list of config files to run.")
    parser.add_argument("--raw-data-dir", type=str, default="data/raw/lego")
    parser.add_argument("--splits-dir", type=str, default="data/splits")
    parser.add_argument("--experiments-dir", type=str, default="experiments/lego")
    parser.add_argument("--results-csv", type=str, default="results/metrics.csv")
    args = parser.parse_args()

    run_experiment(args)
