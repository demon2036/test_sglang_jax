#!/usr/bin/env python3
"""Smoke-run EasyDeL GRPO on GSM8K for 10 steps.

Example:
  python -u plugins/easydel/run_grpo_gsm8k_10steps.py
"""

from __future__ import annotations

import argparse
import os
import re

import jax
import jax.numpy as jnp
from datasets import load_dataset
from transformers import AutoTokenizer

import easydel as ed

DEFAULT_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_DATASET_ID = "openai/gsm8k"
DEFAULT_DATASET_CONFIG = "main"
DEFAULT_DATASET_SPLIT = "train[:128]"
DEFAULT_SYSTEM_PROMPT = (
    "Solve the math problem. Show work briefly and put the final answer after '#### '."
)

ANSWER_RE = re.compile(r"####\s*(-?\d+(?:\.\d+)?)")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default=os.environ.get("EASYDEL_MODEL_ID", DEFAULT_MODEL_ID))
    parser.add_argument("--dataset-id", default=os.environ.get("EASYDEL_DATASET_ID", DEFAULT_DATASET_ID))
    parser.add_argument("--dataset-config", default=os.environ.get("EASYDEL_DATASET_CONFIG", DEFAULT_DATASET_CONFIG))
    parser.add_argument("--dataset-split", default=os.environ.get("EASYDEL_DATASET_SPLIT", DEFAULT_DATASET_SPLIT))
    parser.add_argument("--max-samples", type=int, default=int(os.environ.get("EASYDEL_MAX_SAMPLES", "128")))
    parser.add_argument("--max-steps", type=int, default=int(os.environ.get("EASYDEL_MAX_STEPS", "10")))
    parser.add_argument("--total-batch-size", type=int, default=int(os.environ.get("EASYDEL_BATCH_SIZE", "1")))
    parser.add_argument(
        "--num-return-sequences",
        type=int,
        default=int(os.environ.get("EASYDEL_NUM_RETURN_SEQS", "2")),
    )
    parser.add_argument(
        "--max-prompt-length",
        type=int,
        default=int(os.environ.get("EASYDEL_MAX_PROMPT_LEN", "256")),
    )
    parser.add_argument(
        "--max-completion-length",
        type=int,
        default=int(os.environ.get("EASYDEL_MAX_COMPLETION_LEN", "128")),
    )
    parser.add_argument("--learning-rate", type=float, default=float(os.environ.get("EASYDEL_LR", "1e-6")))
    parser.add_argument("--temperature", type=float, default=float(os.environ.get("EASYDEL_TEMP", "0.7")))
    parser.add_argument("--top-p", type=float, default=float(os.environ.get("EASYDEL_TOP_P", "0.9")))
    parser.add_argument("--top-k", type=int, default=int(os.environ.get("EASYDEL_TOP_K", "50")))
    parser.add_argument("--beta", type=float, default=float(os.environ.get("EASYDEL_BETA", "0.04")))
    parser.add_argument(
        "--save-dir",
        default=os.environ.get("EASYDEL_SAVE_DIR", "grpo_gsm8k_10steps"),
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        default=os.environ.get("EASYDEL_TRUST_REMOTE_CODE", "0") == "1",
    )
    return parser.parse_args()


def _build_prompt(question: str) -> str:
    return f"{DEFAULT_SYSTEM_PROMPT}\n\nQuestion: {question}\nAnswer:"


def _format_reward(prompts, completions, **_kwargs) -> list[float]:
    rewards: list[float] = []
    for completion in completions:
        text = completion
        if isinstance(completion, list):
            if completion and isinstance(completion[0], dict):
                text = completion[0].get("content", "")
            else:
                text = str(completion)
        rewards.append(1.0 if ANSWER_RE.search(str(text)) else 0.0)
    return rewards


def main() -> None:
    args = _parse_args()

    os.environ.setdefault("JAX_PLATFORMS", "tpu")
    os.environ.setdefault("HF_HOME", "/root/.cache/huggingface")
    os.environ.setdefault("HF_DATASETS_CACHE", "/root/.cache/huggingface/datasets")

    print(f"JAX backend: {jax.default_backend()}")
    print(f"Model: {args.model_id}")
    print(f"Dataset: {args.dataset_id} ({args.dataset_config}) split={args.dataset_split}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=args.trust_remote_code)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    # GRPOTrainer expects padding_value for collators; set at class level to avoid upstream edits.
    ed.GRPOTrainer.padding_value = tokenizer.pad_token_id

    dataset = load_dataset(args.dataset_id, args.dataset_config, split=args.dataset_split)
    if "question" not in dataset.column_names:
        raise ValueError("Dataset missing 'question' column; GSM8K should include it.")

    dataset = dataset.map(
        lambda ex: {"prompt": _build_prompt(ex["question"])},
        remove_columns=dataset.column_names,
    )

    if args.max_samples:
        try:
            if len(dataset) > args.max_samples:
                dataset = dataset.select(range(args.max_samples))
        except TypeError:
            pass

    max_sequence_length = args.max_prompt_length + args.max_completion_length

    model = ed.AutoEasyDeLModelForCausalLM.from_pretrained(
        args.model_id,
        auto_shard_model=True,
        sharding_axis_dims=(1, -1, 1, 1, 1),
        config_kwargs=ed.EasyDeLBaseConfigDict(
            freq_max_position_embeddings=max_sequence_length,
            mask_max_position_embeddings=max_sequence_length,
            attn_dtype=jnp.bfloat16,
            attn_softmax_dtype=jnp.bfloat16,
            kvdtype=jnp.bfloat16,
            attn_mechanism=ed.AttentionMechanisms.AUTO,
            gradient_checkpointing=ed.EasyDeLGradientCheckPointers.NONE,
        ),
        param_dtype=jnp.bfloat16,
        dtype=jnp.bfloat16,
        precision=jax.lax.Precision.DEFAULT,
        trust_remote_code=args.trust_remote_code,
    )

    config = ed.GRPOConfig(
        model_name="grpo_gsm8k_10steps",
        save_directory=args.save_dir,
        total_batch_size=args.total_batch_size,
        num_train_epochs=1,
        max_training_steps=args.max_steps,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        num_return_sequences=args.num_return_sequences,
        learning_rate=args.learning_rate,
        beta=args.beta,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        optimizer=ed.EasyDeLOptimizers.ADAMW,
        scheduler=ed.EasyDeLSchedulers.NONE,
        log_steps=1,
        report_steps=1,
        progress_bar_type="json",
        do_last_save=False,
        save_steps=None,
        save_optimizer_state=False,
        use_wandb=False,
        track_memory=False,
    )

    trainer = ed.GRPOTrainer(
        arguments=config,
        model=model,
        reward_funcs=[_format_reward],
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    trainer.train()


if __name__ == "__main__":
    main()
