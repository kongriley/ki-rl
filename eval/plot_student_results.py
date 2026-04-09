"""Line plot of closed-book accuracy from eval_questions JSON outputs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt


# Matches eval/run_test_student_models.sh; Base also accepts base_results.json if present.
BASE_SERIES: tuple[str, tuple[str, ...]] = (
    "Base",
    ("student_base_results.json", "base_results.json"),
)
STUDENT_SERIES: list[tuple[str, tuple[str, ...]]] = [
    ("Student 0", ("student_0_results.json",)),
    ("Student 1", ("student_1_results.json",)),
    ("Student 2", ("student_2_results.json",)),
    ("Student 3", ("student_3_results.json",)),
    ("Student 4", ("student_4_results.json",)),
]
DEFAULT_SERIES: list[tuple[str, tuple[str, ...]]] = [BASE_SERIES] + STUDENT_SERIES


def _load_accuracy(results_dir: Path, candidates: tuple[str, ...]) -> tuple[Path, float] | None:
    for fname in candidates:
        path = results_dir / fname
        if path.is_file():
            with path.open(encoding="utf-8") as f:
                data = json.load(f)
            return path, float(data["accuracy"]) * 100.0
    return None


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--results-dir",
        type=Path,
        default=Path("eval/out"),
        help="Directory containing student_*_results.json files.",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="PNG path (default: <results-dir>/student_results_accuracy.png).",
    )
    p.add_argument(
        "--no-base",
        action="store_true",
        help="Exclude the base (pretrained) model; plot only student checkpoints.",
    )
    args = p.parse_args()
    results_dir = args.results_dir
    out_path = args.output or (results_dir / "student_results_accuracy.png")

    plot_series = STUDENT_SERIES if args.no_base else DEFAULT_SERIES

    labels: list[str] = []
    pct: list[float] = []
    missing: list[tuple[str, tuple[str, ...]]] = []

    for label, candidates in plot_series:
        loaded = _load_accuracy(results_dir, candidates)
        if loaded is None:
            missing.append((label, candidates))
            continue
        labels.append(label)
        pct.append(loaded[1])

    if not labels:
        raise SystemExit(
            "No result files found. Expected one of:\n  "
            + "\n  ".join(
                f"{label}: {' | '.join(str(results_dir / c) for c in cands)}"
                for label, cands in plot_series
            )
        )
    if missing:
        print("Warning: skipping missing checkpoints:", file=sys.stderr)
        for label, cands in missing:
            print(
                f"  {label}: tried "
                + ", ".join(str(results_dir / c) for c in cands),
                file=sys.stderr,
            )

    fig, ax = plt.subplots(figsize=(6, 4.5))
    x = list(range(len(labels)))
    ax.plot(
        x,
        pct,
        color="#276749",
        linewidth=2.2,
        marker="o",
        markersize=9,
        markerfacecolor="#2f855a",
        markeredgecolor="#1c4532",
        markeredgewidth=1.0,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Closed-book accuracy of student model")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)
    for xi, v in zip(x, pct):
        ax.annotate(
            f"{v:.1f}%",
            xy=(xi, v),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            fontsize=9,
        )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved plot to {out_path}")


if __name__ == "__main__":
    main()
