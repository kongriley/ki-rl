NUM_GPUS=$(python -c "import torch; print(torch.cuda.device_count())")

torchrun --standalone --nproc_per_node=${NUM_GPUS} baselines/cpt.py \
    --model_name allenai/OLMo-2-1124-7B-Instruct \
    --data_path data/wiki_20/data.json \
    --output_dir out/cpt_wiki_20 \
    --block_size 2048 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --num_train_epochs 10 \
    --learning_rate 5e-5 \
    --seed 42
