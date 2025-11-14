#!/bin/bash
#SBATCH -p vision-pulkitag-h100,vision-pulkitag-a100        # specify the partition
#SBATCH -q vision-pulkitag-debug                            # specify the QoS (free cycles)
#SBATCH -A vision-pulkitag-urops
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=1:00:00
#SBATCH --export=ALL
#SBATCH --output=out/%x.%j.out

export HOME=/data/scratch/rileyis

# Source your bashrc
source /data/scratch/rileyis/.bashrc

# Navigate to your project directory
cd /data/scratch/rileyis/ki-rl/

source .venv/bin/activate

python eval.py