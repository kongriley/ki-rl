#!/usr/bin/env python3
"""Format rows from distill _questions_*.jsonl files (id, question, answer) to a text file."""

from __future__ import annotations

import argparse
import json
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, TextIO


def iter_question_files(directory: Path) -> list[Path]:
    files = sorted(directory.glob("_questions_*.jsonl"), key=lambda p: int(p.stem.split("_")[-1]))
    return files


def resolve_paths(path: Path, iterations: list[int] | None) -> list[Path]:
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(str(path))
    files = iter_question_files(path)
    if not files:
        raise FileNotFoundError(f"No _questions_*.jsonl under {path}")
    if iterations is None:
        return files
    out: list[Path] = []
    for i in iterations:
        match = [p for p in files if int(p.stem.split("_")[-1]) == i]
        if not match:
            avail = [int(p.stem.split("_")[-1]) for p in files]
            raise FileNotFoundError(f"No _questions_{i}.jsonl; available iterations: {avail}")
        out.append(match[0])
    return out


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {e}") from e
    return rows


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "path",
        nargs="?",
        default="distill/out/grpo_distill",
        type=Path,
        help="JSONL file, or directory containing _questions_*.jsonl (default: %(default)s)",
    )
    p.add_argument(
        "-i",
        "--iteration",
        type=int,
        action="append",
        dest="iterations",
        metavar="N",
        help="Only show this iteration (repeatable). Default: all files in directory.",
    )
    p.add_argument(
        "-n",
        "--limit",
        type=int,
        default=None,
        help="Max rows per file",
    )
    p.add_argument(
        "--no-answer",
        action="store_true",
        help="Omit answers",
    )
    p.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        metavar="FILE",
        help="Write formatted questions here. Use '-' for stdout. "
        "Default: inside the question directory, _questions_display.txt (single dir) "
        "or next to a single JSONL file as <stem>_display.txt.",
    )
    return p.parse_args()


def default_output_path(question_path: Path) -> Path:
    if question_path.is_file():
        return question_path.parent / f"{question_path.stem}_display.txt"
    return question_path / "_questions_display.txt"


@contextmanager
def open_output(path_or_dash: str | Path) -> Iterator[TextIO]:
    if str(path_or_dash) == "-":
        yield sys.stdout
        return
    p = Path(path_or_dash)
    if not p.is_absolute():
        p = (Path.cwd() / p).resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        yield f


def main() -> int:
    args = parse_args()
    path = args.path
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()

    try:
        files = resolve_paths(path, args.iterations)
    except FileNotFoundError as e:
        print(e, file=sys.stderr)
        return 1

    if args.output is None:
        out_target: str | Path = default_output_path(path)
    else:
        out_target = args.output

    try:
        with open_output(out_target) as out:
            for fp in files:
                rows = load_jsonl(fp)
                if args.limit is not None:
                    rows = rows[: args.limit]
                out.write(f"\n{'=' * 72}\n{fp.name}  ({len(rows)} rows)\n{'=' * 72}\n")
                for idx, row in enumerate(rows, 1):
                    qid = row.get("id", "")
                    question = row.get("question", "")
                    answer = row.get("answer", "")
                    out.write(f"\n[{idx}] id={qid}\n")
                    out.write(question.rstrip() + "\n")
                    if not args.no_answer and answer != "":
                        out.write(f"answer: {answer}\n")
    except OSError as e:
        print(f"Cannot write --output {out_target!r}: {e}", file=sys.stderr)
        return 1

    if str(out_target) != "-":
        written = Path(out_target)
        if not written.is_absolute():
            written = (Path.cwd() / written).resolve()
        else:
            written = written.resolve()
        print(f"Wrote {written}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
