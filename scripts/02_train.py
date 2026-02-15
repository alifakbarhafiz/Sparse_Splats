"""Train the Gaussian Splatting model through the bundled script."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List


TRAIN_SCRIPT = Path(__file__).resolve().parents[1] / "external" / "gaussian-splatting" / "train.py"


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


def train_model(config: Dict[str, Any]) -> None:
    """Invoke the Gaussian Splatting training script with the provided configuration."""
    parameters = dict(config)
    source_dir = Path(parameters.pop("source_dir"))
    model_dir = Path(parameters.pop("model_dir"))
    args_map = parameters.pop("args", {})

    command: List[str] = [
        sys.executable,
        str(TRAIN_SCRIPT),
        "-s",
        str(source_dir),
        "-m",
        str(model_dir),
    ]
    command.extend(_flatten_cli_args(args_map))

    model_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(command, check=True)


if __name__ == "__main__":
    import argparse
    import yaml

    parser = argparse.ArgumentParser(description="Train a Gaussian Splatting model from a YAML snippet.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML describing source_dir, model_dir, and args.")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as handle:
        configuration = yaml.safe_load(handle)
    train_model(configuration)
