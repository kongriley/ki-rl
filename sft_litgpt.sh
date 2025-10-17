#!/bin/bash
#SBATCH -p vision-pulkitag-h100,vision-pulkitag-a100        # specify the partition
#SBATCH -q vision-pulkitag-debug                            # specify the QoS (free cycles)
#SBATCH -A vision-pulkitag-urops
#SBATCH -t 2:00:00                                          # job time
#SBATCH -n 1                                                # number of tasks
#SBATCH --gres=gpu:4                                        # request GPU resource
#SBATCH --mem=200G
#SBATCH --output=out/%x.%j.out

export HOME=/data/scratch/rileyis

# Source your bashrc
source /data/scratch/rileyis/.bashrc

# Navigate to your project directory
cd /data/scratch/rileyis/ki-rl/

source .venv/bin/activate

litgpt finetune_lora allenai/OLMo-2-1124-7B-Instruct \
    --data JSON \
    --data.json_path data/wiki_1k_questions.jsonl \
    --data.val_split_fraction 0.1 \
    --seed 42