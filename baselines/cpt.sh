NUM_GPUS=$(python -c "import torch; print(torch.cuda.device_count())")
DATASET="thetech"
MODE="active_reading" # cpt or active_reading

torchrun --standalone --nproc_per_node=${NUM_GPUS} baselines/cpt.py \
    --model_name allenai/OLMo-2-1124-7B-Instruct \
    --data_path data/${DATASET}/$(if [ "${MODE}" = "cpt" ]; then echo "data"; else echo "${MODE}"; fi).json \
    --output_dir out/${MODE}_${DATASET} \
    --block_size 2048 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --num_train_epochs 10 \
    --learning_rate 5e-5 \
    --seed 42
