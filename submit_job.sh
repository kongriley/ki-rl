#!/bin/bash

# Default partition (can be overridden with -p/--partition)
PARTITION_NAME="vision-pulkitag-h200"

# Parse optional args
while [[ $# -gt 0 ]]; do
    case "$1" in
        -p|--partition)
            if [ -z "$2" ]; then
                echo "Error: --partition requires a value"
                echo "Usage: ./submit_job.sh [-p partition_name] path/to/script.py"
                exit 1
            fi
            PARTITION_NAME="$2"
            shift 2
            ;;
        -*)
            echo "Unknown option: $1"
            echo "Usage: ./submit_job.sh [-p partition_name] path/to/script.py"
            exit 1
            ;;
        *)
            PY_SCRIPT="$1"
            shift
            ;;
    esac
done

# Check that a python file was provided
if [ -z "$PY_SCRIPT" ]; then
    echo "Usage: ./submit_job.sh [-p partition_name] path/to/script.py"
    exit 1
fi

# Extract filename without path and .py extension
JOB_NAME=$(basename "$PY_SCRIPT" .py)

echo "Submitting job: $JOB_NAME"
echo "Python file: $PY_SCRIPT"
echo "Partition: $PARTITION_NAME"

sbatch <<EOF
#!/bin/bash
#SBATCH -J $JOB_NAME
#SBATCH -p $PARTITION_NAME
#SBATCH -q vision-pulkitag-debug
#SBATCH -A vision-pulkitag-urops
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=2:00:00
#SBATCH --export=ALL
#SBATCH --output=/data/scratch/rileyis/runs/ki-rl/%x.%j.out

export HOME=/data/scratch/rileyis
source /data/scratch/rileyis/.bashrc
cd /data/scratch/rileyis/ki-rl/
source .venv/bin/activate

python $PY_SCRIPT
EOF