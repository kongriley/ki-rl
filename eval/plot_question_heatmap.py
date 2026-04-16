"""Heatmap of per-question correctness across student model checkpoints."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from matplotlib.patches import Patch


BASE_SERIES: tuple[str, tuple[str, ...]] = (
    "Base",
    ("student_base_results.json", "base_results.json"),
)

_STUDENT_PATTERNS = [
    re.compile(r"^student_(\d+)_results\.json$"),
    re.compile(r"^results_student_model_(\d+)\.json$"),
]


def _discover_student_series(results_dir: Path) -> list[tuple[str, tuple[str, ...]]]:
    found: dict[int, str] = {}
    for p in results_dir.iterdir():
        for pat in _STUDENT_PATTERNS:
            m = pat.match(p.name)
            if m:
                found.setdefault(int(m.group(1)), p.name)
                break
    return [(f"{i}", (found[i],)) for i in sorted(found)]


def _load_results(results_dir: Path, candidates: tuple[str, ...]) -> list[dict] | None:
    for fname in candidates:
        path = results_dir / fname
        if path.is_file():
            with path.open(encoding="utf-8") as f:
                data = json.load(f)
            return data["results"]
    return None


def _question_key(r: dict) -> str:
    """Stable identifier for a question across result files."""
    return f"{r['id']}|||{r['question']}"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--results-dir", type=Path, default=Path("eval/out"),
        help="Directory containing student result JSON files.",
    )
    p.add_argument(
        "--output", type=Path, default=None,
        help="Output path (default: <results-dir>/question_heatmap.png).",
    )
    p.add_argument(
        "--no-base", action="store_true",
        help="Exclude the base (pretrained) model.",
    )
    p.add_argument(
        "--sort", choices=["difficulty", "passage", "flip"], default="flip",
        help=(
            "Sort order for questions. "
            "'difficulty': by total correct count. "
            "'passage': group by passage ID. "
            "'flip': by first checkpoint where the question becomes correct (default)."
        ),
    )
    p.add_argument(
        "--truncate-labels", type=int, default=60,
        help="Max characters for question labels on the y-axis.",
    )
    args = p.parse_args()

    results_dir = args.results_dir
    out_path = args.output or (results_dir / "question_heatmap.png")

    student_series = _discover_student_series(results_dir)
    plot_series = student_series if args.no_base else [BASE_SERIES] + student_series

    model_labels: list[str] = []
    model_results: list[list[dict]] = []
    missing: list[tuple[str, tuple[str, ...]]] = []

    for label, candidates in plot_series:
        loaded = _load_results(results_dir, candidates)
        if loaded is None:
            missing.append((label, candidates))
            continue
        model_labels.append(label)
        model_results.append(loaded)

    if not model_labels:
        raise SystemExit(
            "No result files found. Check --results-dir points to the right directory."
        )

    if missing:
        print("Warning: skipping missing checkpoints:", file=sys.stderr)
        for label, cands in missing:
            print(f"  {label}: tried " + ", ".join(str(results_dir / c) for c in cands),
                  file=sys.stderr)

    all_keys: list[str] = []
    key_set: set[str] = set()
    for r in model_results[0]:
        k = _question_key(r)
        if k not in key_set:
            all_keys.append(k)
            key_set.add(k)

    n_questions = len(all_keys)
    n_models = len(model_labels)

    correctness = np.full((n_questions, n_models), np.nan)
    key_to_idx = {k: i for i, k in enumerate(all_keys)}

    for mi, results in enumerate(model_results):
        for r in results:
            k = _question_key(r)
            qi = key_to_idx.get(k)
            if qi is not None:
                correctness[qi, mi] = 1.0 if r["is_correct"] else 0.0

    question_meta: dict[str, dict] = {}
    for r in model_results[0]:
        k = _question_key(r)
        if k not in question_meta:
            question_meta[k] = {"id": r["id"], "question": r["question"]}

    if args.sort == "difficulty":
        total_correct = np.nansum(correctness, axis=1)
        order = np.argsort(total_correct)
    elif args.sort == "passage":
        order = np.array(sorted(
            range(n_questions),
            key=lambda i: (question_meta[all_keys[i]]["id"], all_keys[i]),
        ))
    else:  # flip
        def _flip_key(i: int) -> tuple:
            row = correctness[i]
            first_correct = n_models
            for j in range(n_models):
                if row[j] == 1.0:
                    first_correct = j
                    break
            total = np.nansum(row)
            return (first_correct, -total)
        order = np.array(sorted(range(n_questions), key=_flip_key))

    correctness = correctness[order]
    ordered_keys = [all_keys[i] for i in order]

    trunc = args.truncate_labels
    y_labels = []
    for k in ordered_keys:
        meta = question_meta[k]
        q = meta["question"]
        if len(q) > trunc:
            q = q[:trunc - 1] + "…"
        y_labels.append(f"[{meta['id']}] {q}")

    plt.rcParams.update({
        "font.size": 9,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 7,
        "figure.dpi": 150,
        "savefig.dpi": 300,
    })

    cell_h = 0.22
    fig_h = max(3.0, n_questions * cell_h + 1.5)
    fig_w = max(4.0, n_models * 0.7 + 3.5)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), constrained_layout=True)

    cmap = mcolors.ListedColormap(["#d94f4f", "#4a90d9"])
    bounds = [-0.5, 0.5, 1.5]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    display = np.where(np.isnan(correctness), -1, correctness)
    cmap_with_missing = mcolors.ListedColormap(["#e0e0e0", "#d94f4f", "#4a90d9"])
    bounds_with_missing = [-1.5, -0.5, 0.5, 1.5]
    norm_with_missing = mcolors.BoundaryNorm(bounds_with_missing, cmap_with_missing.N)

    ax.imshow(
        display, aspect="auto", cmap=cmap_with_missing, norm=norm_with_missing,
        interpolation="nearest",
    )

    for i in range(n_questions + 1):
        ax.axhline(i - 0.5, color="white", linewidth=0.5)
    for j in range(n_models + 1):
        ax.axvline(j - 0.5, color="white", linewidth=0.5)

    ax.set_xticks(range(n_models))
    ax.set_xticklabels(model_labels)
    ax.xaxis.set_ticks_position("top")
    ax.xaxis.set_label_position("top")

    ax.set_yticks(range(n_questions))
    ax.set_yticklabels(y_labels)

    ax.set_xlabel("Checkpoint")
    ax.set_title("Per-question correctness across student models", pad=14)

    legend_elements = [
        Patch(facecolor="#4a90d9", edgecolor="white", label="Correct"),
        Patch(facecolor="#d94f4f", edgecolor="white", label="Incorrect"),
    ]
    if np.any(np.isnan(correctness)):
        legend_elements.append(
            Patch(facecolor="#e0e0e0", edgecolor="white", label="Missing")
        )
    ax.legend(handles=legend_elements, loc="lower right", fontsize=8, framealpha=0.9)

    per_model_acc = np.nanmean(correctness, axis=0) * 100
    for j, acc in enumerate(per_model_acc):
        ax.text(j, n_questions - 0.1, f"{acc:.0f}%", ha="center", va="top",
                fontsize=7, color="white", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.15", fc="black", alpha=0.6))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved heatmap to {out_path}")


if __name__ == "__main__":
    main()
