"""Line plot of closed-book accuracy from eval_questions JSON outputs."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


BASE_SERIES: tuple[str, tuple[str, ...]] = (
    "Base",
    ("student_base_results.json", "base_results.json"),
)

_STUDENT_PATTERNS = [
    re.compile(r"^student_(\d+)_results\.json$"),
    re.compile(r"^results_student_model_(\d+)\.json$"),
]


def _discover_student_series(results_dir: Path) -> list[tuple[str, tuple[str, ...]]]:
    """Auto-discover student result files and return sorted series entries."""
    found: dict[int, str] = {}
    for p in results_dir.iterdir():
        for pat in _STUDENT_PATTERNS:
            m = pat.match(p.name)
            if m:
                found.setdefault(int(m.group(1)), p.name)
                break
    return [(f"{i}", (found[i],)) for i in sorted(found)]


def _load_accuracy(results_dir: Path, candidates: tuple[str, ...]) -> tuple[Path, float] | None:
    for fname in candidates:
        path = results_dir / fname
        if path.is_file():
            with path.open(encoding="utf-8") as f:
                data = json.load(f)
            return path, float(data["accuracy"]) * 100.0
    return None


def _compute_ylim(values: list[float], pad_frac: float = 0.08) -> tuple[float, float]:
    """Choose a tight y-range with a little padding for paper-style figures."""
    vmin = min(values)
    vmax = max(values)

    if vmin == vmax:
        pad = max(1.0, 0.05 * max(abs(vmin), 1.0))
        return max(0.0, vmin - pad), min(100.0, vmax + pad)

    span = vmax - vmin
    pad = max(1.0, span * pad_frac)
    lo = max(0.0, vmin - pad)
    hi = min(100.0, vmax + pad)
    return lo, hi


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
        help="Output path (default: <results-dir>/student_results_accuracy.pdf).",
    )
    p.add_argument(
        "--no-base",
        action="store_true",
        help="Exclude the base (pretrained) model; plot only student checkpoints.",
    )
    p.add_argument(
        "--annotate",
        action="store_true",
        help="Annotate only key points (base, best, final).",
    )
    args = p.parse_args()

    results_dir = args.results_dir
    out_path = args.output or (results_dir / "student_results_accuracy.png")

    student_series = _discover_student_series(results_dir)
    plot_series = student_series if args.no_base else [BASE_SERIES] + student_series

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

    # Paper-friendly matplotlib defaults.
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    # Single-column friendly size for papers.
    fig, ax = plt.subplots(figsize=(3.5, 2.5), constrained_layout=True)

    x = list(range(len(labels)))

    # Main line: neutral dark color for paper readability.
    ax.plot(
        x,
        pct,
        color="black",
        linewidth=1.8,
        marker="o",
        markersize=4.8,
        markerfacecolor="white",
        markeredgecolor="black",
        markeredgewidth=1.0,
        zorder=3,
    )

    # Optional subtle emphasis on the base point.
    if labels and labels[0] == "Base":
        ax.scatter(
            [x[0]],
            [pct[0]],
            s=34,
            facecolor="black",
            edgecolor="black",
            zorder=4,
            label="Base",
        )

    k = max(1, len(labels) // 8)  # ~8 labels total

    ax.set_xticks(x)
    ax.set_xticklabels([
        lbl if i % k == 0 else ""
        for i, lbl in enumerate(labels)
    ])

    # Only rotate if needed.
    if max(len(lbl) for lbl in labels) > 3:
        plt.setp(ax.get_xticklabels(), rotation=25, ha="right")

    ax.set_xlabel("Checkpoint")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Closed-book accuracy")

    # Tight y-range is usually much better for papers than 0-100.
    ylo, yhi = _compute_ylim(pct)
    ax.set_ylim(ylo, yhi)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))

    # Light horizontal grid only.
    ax.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.3)
    ax.set_axisbelow(True)

    # Optional minimal annotation: base, best, final only.
    if args.annotate:
        idxs_to_annotate = {0, len(pct) - 1, max(range(len(pct)), key=lambda i: pct[i])}
        for i in sorted(idxs_to_annotate):
            ax.annotate(
                f"{pct[i]:.1f}",
                xy=(x[i], pct[i]),
                xytext=(0, 6),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot to {out_path}")


if __name__ == "__main__":
    main()