# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Knowledge Injection via Question Generation (ki-rl). Iteratively trains LLMs to generate passage-grounded questions, then distills passage knowledge into student models through a multi-phase GRPO + distillation loop. Built on PyTorch, Hugging Face Transformers, TRL, and vLLM.

## Running the Pipeline

Always activate the venv first: `source .venv/bin/activate`.

```bash
# Full pipeline (GRPO + distill loop)
./main.sh

# Skip GRPO; use the unmodified base model as the question generator
MODE=no_gen ./main.sh

# Debug mode (max_steps=1 for quick smoke tests)
DEBUG=1 ./main.sh
```

`main.sh` calls `distill/main.py`. It wires a common set of args (default dataset, iteration count, eval judge, etc.) and dispatches by `MODE`. To change the dataset, edit `DATASET=` at the top of `main.sh` or invoke `distill/main.py` directly with `--dataset_path`.

### Ablations / Experiment Sweeps

```bash
# Run all configured ablations locally (sequential)
./run_ablations.sh

# Submit each ablation as a separate SLURM job
./run_ablations.sh --slurm

# Dry run: print what would execute
./run_ablations.sh --list

# Override dataset/iteration count
DATASET=thetech NUM_ITERATIONS=30 ./run_ablations.sh
```

Each experiment writes to `distill/out/ablation_<name>/` and copies its per-iteration evaluation JSON to `results/<dataset>/results_ablation_<name>_<iter>.json`.

### Standalone Evaluation

```bash
# Evaluate any checkpoint (or the base model) against pre-generated questions
python3 eval/eval_questions.py \
  --model allenai/OLMo-2-1124-7B-Instruct \
  --data_path data/wiki_20/data.json \
  --questions_path data/wiki_20/gpt-5-mini_questions.jsonl \
  --judge_backend openai --judge_model gpt-5-mini --batch_judge

# Evaluate the standard baselines on a corpus (base | icl | rag | cpt)
bash eval/eval_question_baselines.sh icl
```

### Baselines

```bash
# Continued pretraining on raw passages
python3 baselines/cpt.py --model_name allenai/OLMo-2-1124-7B-Instruct --data_path data/wiki_20/data.json
bash baselines/cpt.sh            # SLURM-friendly multi-GPU wrapper (set DATASET inside)

# Active reading (LM-generated study notes → CPT)
python3 baselines/active_reading.py --data_path data/wiki_20/data.json --run_cpt

# Sanity check that CPT actually reduced perplexity on the passages
python3 baselines/sanity_check_cpt.py
```

### Reference Question Generation

Reference question sets are stored as `data/<corpus>/gpt-5-mini_questions.jsonl` and are generated once with `data/generate_questions.py` (OpenAI backend). They are held out from training and reused across all evaluations of a given corpus. To regenerate:

```bash
python3 data/generate_questions.py --backend openai --model gpt-5-mini \
  --dataset_path data/<corpus>/data.json
```

For arbitrary web-scraped corpora, `data/fetch_articles.py` downloads URLs and runs `trafilatura` main-content extraction into the canonical dataset format.

### HPC Submission

```bash
# Submit any python script to SLURM (default partition: vision-pulkitag-h100, 8 GPUs)
./submit_job.sh distill/main.py --dataset_path ./data/wiki_20/data.json

# Override partition
./submit_job.sh -p some-other-partition distill/main.py [...]
```

`submit_job.sh` is hard-coded to user/account `rileyis` paths under `/data/scratch/`. `submit_job_ccc.sh` is the analogous launcher for the CCC cluster.

## Architecture

### 3-Phase Iterative Loop (`distill/main.py`)

Each of `num_generation_iterations` runs:

1. **GRPO Phase** (`distill/grpo_phase_worker.py`): Trains the question generator via Group Relative Policy Optimization. The reward function evaluates each generated question both closed-book and open-book against the current student:
   - `+1.0` — good question (student fails closed-book, succeeds open-book).
   - `-0.5` — too easy (student succeeds closed-book).
   - `-1.0` — garbage (student fails both).
   - `-1.5` — `PRESIDIO_ANONYMIZED` token leaked into the question/answer.
   - `-2.0` — malformed (fails the `Q:`/`A:` parser or violates length filters).

   Questions that score `+1.0` are streamed to `_good_questions_<iter>.jsonl` for direct reuse in the distillation phase. An optional Jaccard diversity penalty (`--diversity_penalty_weight`) is subtracted from each reward, computed against both batch peers and a running per-passage history.

2. **Deficit Generation** (`data/generate_questions.py` → `build_question_dataset_for_deficits`): For each passage, compute `deficit = num_question_generations − len(good_questions_for_passage)`. Run the trained generator under vLLM to fill exactly the deficit. With `--accumulate_questions`, the previous iteration's generated questions are appended to the prompt with an instruction to produce *different* questions; cap with `--num_accumulated_questions`.

3. **Distillation Phase** (`distill/distill_phase_worker.py` → `distill/distil_trainer.py`): On-policy self-distillation. The student samples completions via colocated vLLM; the same tokens are scored under both the student prompt (question only) and teacher prompt (passage + question), and the loss is full-vocab forward KL between the two distributions.

### Subprocess Isolation and Manifest IPC

Both the GRPO and distillation phases are launched as fresh Python subprocesses, not function calls. The orchestrator writes a JSON manifest (`_grpo_manifest_<iter>.json` / `_distill_manifest_<iter>.json`) listing input paths and trainer kwargs, then `subprocess.run([sys.executable, worker_py, manifest_path], check=True)`. The worker loads its own models, saves to the path named in the manifest, and `os._exit(0)` after tearing down vLLM and CUDA.

This is required because vLLM's colocated sleep-mode memory pool is a per-process singleton, and the pluggable CUDA allocator does not return memory to the driver between vLLM lifecycles — so memory leaks would accumulate across iterations without a fresh process per phase. **Do not refactor either worker into an in-process function call.**

### Checkpoint Layout

Within `--output_dir`, iteration `t` writes `student_model_<t>/` and `question_model_<t>/`. Iteration `t+1` reads these as its inputs. Old directories are deleted at the start of iteration `t+2` unless `--save_student_model` / `--save_question_model` is set. The final student is always persisted as `final_student_model/`.

### Key Modules

- `distill/main.py` — orchestrator (parses args, runs the outer loop, writes manifests, spawns subprocesses, runs optional per-iter evaluation).
- `distill/distil_config.py` — `DistilConfig` dataclass extending `TrainingArguments` (TRL-flavoured KL distillation, vLLM, importance-sampling correction, full-vs-sampled-logit distillation toggle).
- `distill/distil_trainer.py` — custom TRL trainer that runs forward KL over the full vocabulary between teacher (passage-conditioned) and student (closed-book) on student-sampled rollouts.
- `data/generate_questions.py` — prompt template, format validation, vLLM and OpenAI question-generation paths, deficit-aware generation with previous-question conditioning.
- `eval/inference.py` — shared prompt builders (`build_student_prompt`, `build_teacher_prompt`, `build_judge_prompt`), batched generation, single- and multi-prompt judging.
- `eval/eval_questions.py` — end-to-end evaluation script with `--icl`, `--rag` (Qwen3-Embedding-8B), and `--model` modes; OpenAI- or HF-based judging.
- `baselines/{cpt,active_reading}.py` — text-only baselines (raw CPT, LM-generated study notes + CPT).

### Data Formats

- **Dataset input** (`data/<corpus>/data.json`): JSON array of `{"id", "text"[, "url", "title"]}` passage objects.
- **Reference questions** (`data/<corpus>/gpt-5-mini_questions.jsonl`): one `{"id", "instruction", "input", "output"}` record per line (`input` = question, `output` = reference answer).
- **Internal questions** (in-memory `Dataset` rows during distillation): `{"id", "question", "answer", "prompt": [{role, content}, ...], "teacher_prompt": [{role, content}, ...]}` — `prompt`/`teacher_prompt` are conversation-format lists, not plain strings.
- **Per-iteration evaluation** (`results/<corpus>/results_<run>_<iter>.json`): `{"iteration", "accuracy", "correct", "total", "results": [{"question", "reference_answer", "student_answer", "verdict", "is_correct", "id"}, ...]}`.

### Datasets

Three corpora ship under `data/`: `wiki_20` (20 Wikipedia articles, default development corpus), `thetech` (10 MIT *The Tech* articles scraped via `data/fetch_articles.py`), `2025_disasters` (9 articles on post-cutoff natural disasters).

## Environment

- Python 3.12, venv at `.venv/`. `requirements.txt` pins the deps.
- Key deps: `torch`, `transformers`, `trl` (GRPO), `vllm` (colocated rollouts), `deepspeed`, `bitsandbytes` (4-bit reward models), `wandb`, `trafilatura` (article scraping).
- `.env` contains `OPENAI_API_KEY` (used for OpenAI question generation and OpenAI-backed judging).
- Experiment tracking via Weights & Biases (`report_to="wandb"` in some baselines; the main loop uses `"none"` by default).
- GPU expectations: a single 80 GB GPU is enough when `--quantize-reward-models` is set (default in `main.sh`); without quantisation the student/judge are swapped between CPU and GPU around each reward call and runs are much slower.

## Conventions

- Long-running training writes intermediate logs/checkpoints under `out/` and `distill/out/`. Old training runs are moved to `archive/` rather than deleted.
- Per-experiment evaluation JSONs land in `results/<corpus>/` and are aggregated by the plotting scripts (`eval/plot_student_results.py`, `eval/plot_question_heatmap.py`).
- When changing the reward function or its penalty constants, update both `distill/grpo_phase_worker.py` (the constants at the top) and the corresponding description in `methodology.tex` / `experiments.tex`.
