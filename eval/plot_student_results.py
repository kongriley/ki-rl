"""Line plot of closed-book accuracy from eval_questions JSON outputs.

Discovers any files of the form ``results_{x}.json`` or ``results_{x}_{i}.json``
(where ``i`` is a non-negative integer checkpoint index) in ``--results-dir``.

Each distinct ``x`` is treated as a separate method. Methods with indexed files
are plotted as a trajectory over checkpoints; methods with a single
non-indexed file are drawn as a horizontal reference line.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


# Map the ``x`` portion of ``results_{x}(_{i}).json`` to a display label.
METHOD_LABELS: dict[str, str] = {
    "base": "Closed-book base",
    "icl": "Oracle RAG (ICL)",
    "rag": "Normal RAG",
    "student": "GRPO+Distill (Ours)",
    "student_model": "GRPO+Distill (Ours)",
}


_FILE_PATTERN = re.compile(r"^results_(?P<name>.+?)(?:_(?P<idx>\d+))?\.json$")


def _discover(
    results_dir: Path,
) -> dict[str, list[tuple[int | None, Path]]]:
    """Return ``{method_name: [(index, path), ...]}`` for result files found."""
    methods: dict[str, list[tuple[int | None, Path]]] = defaultdict(list)
    for p in sorted(results_dir.iterdir()):
        if not p.is_file():
            continue
        m = _FILE_PATTERN.fullmatch(p.name)
        if m is None:
            continue
        idx = int(m["idx"]) if m["idx"] is not None else None
        methods[m["name"]].append((idx, p))
    return methods


def _load_accuracy(path: Path) -> float:
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    return float(data["accuracy"]) * 100.0


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


def _parse_label_overrides(specs: list[str]) -> dict[str, str]:
    overrides: dict[str, str] = {}
    for spec in specs:
        if "=" not in spec:
            raise SystemExit(f"Invalid --label spec: {spec!r} (expected NAME=LABEL)")
        key, value = spec.split("=", 1)
        overrides[key.strip()] = value.strip()
    return overrides


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Directory containing results_*.json files.",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path (default: <results-dir>/accuracy.png).",
    )
    p.add_argument(
        "--label",
        action="append",
        default=[],
        metavar="NAME=LABEL",
        help="Override the display label for a method name. Repeatable.",
    )
    p.add_argument(
        "--include",
        action="append",
        default=None,
        metavar="NAME",
        help="Only plot these method names. Repeatable. Defaults to all found.",
    )
    p.add_argument(
        "--exclude",
        action="append",
        default=[],
        metavar="NAME",
        help="Skip these method names. Repeatable.",
    )
    p.add_argument(
        "--annotate",
        action="store_true",
        help="Annotate key points (first, best, last) per trajectory.",
    )
    args = p.parse_args()

    results_dir: Path = args.results_dir
    label_map = {**METHOD_LABELS, **_parse_label_overrides(args.label)}

    discovered = _discover(results_dir)
    if not discovered:
        raise SystemExit(f"No results_*.json files found in {results_dir}")

    include = set(args.include) if args.include else None
    exclude = set(args.exclude)

    # Merge method names that share a display label into a single series.
    series: dict[str, list[tuple[int | None, Path]]] = defaultdict(list)
    for name, items in discovered.items():
        if include is not None and name not in include:
            continue
        if name in exclude:
            continue
        label = label_map.get(name, name.replace("_", " ").title())
        series[label].extend(items)

    if not series:
        raise SystemExit("No methods left to plot after applying --include/--exclude.")

    trajectories: dict[str, list[tuple[int, float]]] = {}
    references: dict[str, float] = {}
    for label, items in series.items():
        indexed = sorted((i, pth) for i, pth in items if i is not None)
        non_indexed = [pth for i, pth in items if i is None]
        if indexed:
            trajectories[label] = [(i, _load_accuracy(pth)) for i, pth in indexed]
            for pth in non_indexed:
                print(
                    f"Warning: ignoring non-indexed file {pth} for trajectory '{label}'",
                    file=sys.stderr,
                )
        elif non_indexed:
            if len(non_indexed) > 1:
                print(
                    f"Warning: multiple files map to '{label}' without indices; "
                    f"using {non_indexed[0]}",
                    file=sys.stderr,
                )
            references[label] = _load_accuracy(non_indexed[0])

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 7,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    fig, ax = plt.subplots(figsize=(5.6, 2.8), constrained_layout=True)

    color_cycle = plt.get_cmap("tab10").colors
    ci = 0

    for label, points in trajectories.items():
        color = color_cycle[ci % len(color_cycle)]
        ci += 1
        xs = [i for i, _ in points]
        ys = [v for _, v in points]
        ax.plot(
            xs,
            ys,
            color=color,
            linewidth=1.8,
            marker="o",
            markersize=4.6,
            markerfacecolor="white",
            markeredgecolor=color,
            markeredgewidth=1.2,
            label=label,
            zorder=3,
        )
        if args.annotate and ys:
            best = max(range(len(ys)), key=lambda k: ys[k])
            for k in sorted({0, len(ys) - 1, best}):
                ax.annotate(
                    f"{ys[k]:.1f}",
                    xy=(xs[k], ys[k]),
                    xytext=(0, 6),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    color=color,
                )

    for label, value in references.items():
        color = color_cycle[ci % len(color_cycle)]
        ci += 1
        ax.axhline(
            value,
            color=color,
            linestyle="--",
            linewidth=1.4,
            label=f"{label}",
            zorder=2,
        )

    all_values: list[float] = [v for pts in trajectories.values() for _, v in pts]
    all_values.extend(references.values())
    ylo, yhi = _compute_ylim(all_values)
    ax.set_ylim(ylo, yhi)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))

    all_indices = [i for pts in trajectories.values() for i, _ in pts]
    if all_indices:
        ax.set_xlim(min(all_indices), max(all_indices))
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=8))
    else:
        ax.set_xticks([])

    ax.set_xlabel("Checkpoint")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Closed-book accuracy")
    ax.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.3)
    ax.set_axisbelow(True)

    if trajectories or references:
        ax.legend(loc="upper left", bbox_to_anchor=(1.05, 1))

    out_path: Path = args.output or (results_dir / "accuracy.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot to {out_path}")


if __name__ == "__main__":
    main()
