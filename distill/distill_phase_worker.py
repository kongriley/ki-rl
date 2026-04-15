"""Distillation phase worker — runs in a subprocess for a clean CUDA context.

vLLM colocated mode leaks CUDA memory (pluggable allocator, CUDA graphs) that
torch.cuda.empty_cache() cannot reclaim.  Running each distillation phase in a
fresh process prevents those leaks from accumulating across iterations.
"""

import gc
import json
import os
import socket
import sys


def _find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


# Force a free port before any torch/accelerate import can call init_process_group.
# The parent process may still hold port 29500 from its own GRPO trainer.
os.environ["MASTER_ADDR"] = os.environ.get("MASTER_ADDR", "127.0.0.1")
os.environ["MASTER_PORT"] = str(_find_free_port())

import torch  # noqa: E402
from datasets import Dataset  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from distil_config import DistilConfig  # noqa: E402
from distil_trainer import DistilTrainer  # noqa: E402


def main():
    manifest_path = sys.argv[1]
    with open(manifest_path) as f:
        manifest = json.load(f)

    student = AutoModelForCausalLM.from_pretrained(
        manifest["student_model_path"], torch_dtype=torch.bfloat16,
    ).to("cuda")

    teacher = AutoModelForCausalLM.from_pretrained(
        manifest["teacher_model_path"], torch_dtype=torch.bfloat16,
    )

    tokenizer = AutoTokenizer.from_pretrained(manifest["tokenizer_name"])
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = Dataset.from_json(manifest["dataset_path"])

    config = DistilConfig(**manifest["distil_config"])

    trainer = DistilTrainer(
        model=student,
        ref_model=teacher,
        processing_class=tokenizer,
        args=config,
        train_dataset=dataset,
    )
    trainer.train()

    output_dir = manifest["output_dir"]
    student.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"[distill_phase_worker] Saved student to {output_dir}")

    # Tear down explicitly to avoid CUDAPluggableAllocator crash during
    # Python finalization (vLLM colocate allocates through a custom path
    # that the default teardown order can't handle).
    del trainer, student, teacher
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
