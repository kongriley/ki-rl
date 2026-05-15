#!/bin/bash
#
# SLURM wrapper: submit a Python entrypoint (e.g. distill/main.py) with args.
# Usage:
#   ./submit_job_kevin.sh [-p PARTITION] path/to/script.py [script_args...]
#
# Edit the "Site defaults" block below for account, QoS, and paths.

set -euo pipefail

# --- Site defaults (edit as needed) ------------------------------------------
PARTITION_NAME="vision-pulkitag-h100"
SLURM_QOS="vision-pulkitag-debug"
SLURM_ACCOUNT="vision-pulkitag-urops"
NUM_GPUS=1 # pipeline uses vllm_tensor_parallel_size=1; request 1 fat GPU (e.g. H100 80GB)

REPO_ROOT="/data/scratch/kbzhu/ki-rl"
RUN_LOG_DIR="/data/scratch/kbzhu/runs/ki-rl"
# Optional: point HOME at scratch so HF/torch caches land on fast shared storage
export HOME_SCRATCH="/data/scratch/kbzhu"
# -----------------------------------------------------------------------------

# Parse optional args; everything after the script path is passed through to python
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        -p|--partition)
            if [[ -z "${2:-}" ]]; then
                echo "Error: --partition requires a value" >&2
                echo "Usage: ./submit_job_kevin.sh [-p partition_name] path/to/script.py [script_args...]" >&2
                exit 1
            fi
            PARTITION_NAME="$2"
            shift 2
            ;;
        -*)
            if [[ -n "${PY_SCRIPT:-}" ]]; then
                EXTRA_ARGS+=("$1")
                shift
            else
                echo "Unknown option: $1" >&2
                echo "Usage: ./submit_job_kevin.sh [-p partition_name] path/to/script.py [script_args...]" >&2
                exit 1
            fi
            ;;
        *)
            if [[ -z "${PY_SCRIPT:-}" ]]; then
                PY_SCRIPT="$1"
            else
                EXTRA_ARGS+=("$1")
            fi
            shift
            ;;
    esac
done

if [[ -z "${PY_SCRIPT:-}" ]]; then
    echo "Usage: ./submit_job_kevin.sh [-p partition_name] path/to/script.py [script_args...]" >&2
    exit 1
fi

mkdir -p "$RUN_LOG_DIR"

JOB_NAME=$(basename "$PY_SCRIPT" .py)

echo "Submitting job: $JOB_NAME"
echo "Python file: $PY_SCRIPT"
if ((${#EXTRA_ARGS[@]} > 0)); then
    echo "Script args: ${EXTRA_ARGS[*]}"
fi
echo "Partition: $PARTITION_NAME  |  GPUs: $NUM_GPUS"

# Build one shell-escaped command string so every flag stays its own argv entry after the
# batch script runs (heredoc + "$*" / [*] mistakes are a common cause of "unrecognized
# arguments" with all flags lumped together).
CMD_LINE=""
for token in python -u "$PY_SCRIPT" "${EXTRA_ARGS[@]}"; do
  printf -v q '%q' "$token"
  CMD_LINE+="${CMD_LINE:+ }$q"
done

sbatch <<EOF
#!/bin/bash
#SBATCH -J $JOB_NAME
#SBATCH -p $PARTITION_NAME
#SBATCH -q $SLURM_QOS
#SBATCH -A $SLURM_ACCOUNT
#SBATCH --gres=gpu:$NUM_GPUS
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=6:00:00
#SBATCH --export=ALL
#SBATCH --output=$RUN_LOG_DIR/%x.%j.out

export HOME=$HOME_SCRATCH
set -euo pipefail
source "\$HOME/.bashrc" 2>/dev/null || true
cd "$REPO_ROOT"
source .venv/bin/activate

# Optional secrets (WANDB_*, HF_TOKEN, …). File is gitignored; edit .env in repo root.
if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

# vLLM + FlashInfer may JIT-compile sampling CUDA for sm_90a; distro /usr/bin/nvcc often
# does not support compute_90a ("nvcc fatal: Unsupported gpu architecture 'compute_90a'").
# This uses PyTorch sampling instead (slightly slower, avoids ninja/nvcc for those kernels).
export VLLM_USE_FLASHINFER_SAMPLER=0
# If your site provides a newer toolkit, prefer: module load cuda/12.x && export CUDA_HOME=...
# and set VLLM_USE_FLASHINFER_SAMPLER=1 (or unset) after verifying: nvcc --help | grep sm_90

echo "+ job start: \$(date -Is)" >&2
echo "+ eval $CMD_LINE" >&2
eval "$CMD_LINE"
EOF
