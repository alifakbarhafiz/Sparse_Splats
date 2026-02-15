"""Aggregate experiment metrics and produce summary plots."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import pandas as pd


def _ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _save_summary_csv(summary: pd.DataFrame, dest: Path) -> None:
    _ensure_directory(dest.parent)
    summary.to_csv(dest, index=False)


def _plot_latest_metrics(latest: pd.DataFrame, dest: Path) -> None:
    fig, ax = plt.subplots()
    for metric in ("psnr", "ssim", "lpips"):
        if metric not in latest:
            continue
        ax.plot(latest["view_count"], latest[metric], marker="o", label=metric.upper())
    ax.set_title("Highest iteration metrics per view-count")
    ax.set_xlabel("View count")
    ax.set_ylabel("Metric")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    _ensure_directory(dest.parent)
    fig.savefig(dest)
    plt.close(fig)


def _plot_iteration_trends(summary: pd.DataFrame, dest_prefix: Path) -> None:
    for metric in ("psnr", "ssim", "lpips"):
        if metric not in summary:
            continue
        fig, ax = plt.subplots()
        for view_count, group in summary.groupby("view_count"):
            group = group.sort_values("iteration")
            ax.plot(group["iteration"], group[metric], marker="o", label=str(view_count))
        ax.set_title(f"{metric.upper()} vs iteration")
        ax.set_xlabel("Iteration")
        ax.set_ylabel(metric.upper())
        ax.legend(title="views")
        ax.grid(True)
        fig.tight_layout()
        dest_path = dest_prefix.with_name(f"{dest_prefix.stem}_{metric}_trend.png")
        _ensure_directory(dest_path.parent)
        fig.savefig(dest_path)
        plt.close(fig)


def aggregate_results(metrics_csv: Path, plots_dir: Path) -> None:
    if not metrics_csv.exists():
        print(f"No metrics file found at {metrics_csv}; nothing to aggregate.")
        return

    df = pd.read_csv(metrics_csv)
    if df.empty:
        print(f"{metrics_csv} is empty; skipping aggregation.")
        return

    df["view_count"] = pd.to_numeric(df["view_count"], errors="coerce").fillna(0).astype(int)
    df["iteration"] = pd.to_numeric(df["iteration"], errors="coerce").fillna(0).astype(int)

    summary = (
        df.groupby(["view_count", "iteration"], as_index=False)
        .agg({"psnr": "mean", "ssim": "mean", "lpips": "mean"})
        .sort_values(["view_count", "iteration"])
    )
    plots_dir = Path(plots_dir)
    _save_summary_csv(summary, plots_dir / "metrics_summary.csv")

    latest = summary.sort_values("iteration").groupby("view_count", as_index=False).last()
    _plot_latest_metrics(latest, plots_dir / "latest_metrics.png")
    _plot_iteration_trends(summary, plots_dir / "iteration_trends")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aggregate experiment metrics into plots.")
    parser.add_argument("--metrics-csv", type=Path, default=Path("results/metrics.csv"))
    parser.add_argument("--plots-dir", type=Path, default=Path("results/plots"))
    args = parser.parse_args()
    aggregate_results(args.metrics_csv, args.plots_dir)
