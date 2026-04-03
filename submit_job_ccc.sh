#!/bin/bash

# Default queue and GPU model (can be overridden with flags)
QUEUE_NAME="normal"
GPU_MODEL="NVIDIAH100_80GBHBM3"
NUM_GPUS=1
NUM_CPUS=16
MEM="200GB"
WALL_TIME="120"

# Parse optional args
while [[ $# -gt 0 ]]; do
    case "$1" in
        -q|--queue)
            if [ -z "$2" ]; then
                echo "Error: --queue requires a value (normal, interactive)"
                echo "Usage: ./submit_job_ccc.sh [-q queue] [-g gpu_model] path/to/script.py"
                exit 1
            fi
            QUEUE_NAME="$2"
            shift 2
            ;;
        -g|--gpu-model)
            if [ -z "$2" ]; then
                echo "Error: --gpu-model requires a value"
                echo "Available: TeslaV100_SXM2_32GB, NVIDIAA100_SXM4_40GB, NVIDIAA100_SXM4_80GB, NVIDIAH100_80GBHBM3"
                exit 1
            fi
            GPU_MODEL="$2"
            shift 2
            ;;
        -*)
            echo "Unknown option: $1"
            echo "Usage: ./submit_job_ccc.sh [-q queue] [-g gpu_model] path/to/script.py"
            exit 1
            ;;
        *)
            PY_SCRIPT="$1"
            shift
            ;;
    esac
done

if [ -z "$PY_SCRIPT" ]; then
    echo "Usage: ./submit_job_ccc.sh [-q queue] [-g gpu_model] path/to/script.py"
    exit 1
fi

JOB_NAME=$(basename "$PY_SCRIPT" .py)

if [ "$QUEUE_NAME" = "interactive" ]; then
    WALL_TIME="360"
fi

echo "Submitting job: $JOB_NAME"
echo "Python file: $PY_SCRIPT"
echo "Queue: $QUEUE_NAME"
echo "GPU: $GPU_MODEL"

bsub <<EOF
#!/bin/bash
#BSUB -J $JOB_NAME
#BSUB -q $QUEUE_NAME
#BSUB -gpu "num=${NUM_GPUS}:mode=exclusive_process:gmodel=${GPU_MODEL}"
#BSUB -R "rusage[ngpus=${NUM_GPUS}, cpu=${NUM_CPUS}, mem=${MEM}]"
#BSUB -W $WALL_TIME
#BSUB -o /proj/rlmit/runs/ki-rl/%J.out

source /data/scratch/rileyis/.bashrc
cd /data/scratch/rileyis/ki-rl/
source .venv/bin/activate

python $PY_SCRIPT
EOF
