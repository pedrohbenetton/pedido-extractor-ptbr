"""QLoRA com TRL (perda só na resposta do assistente). Pensado para T4 de Colab/Kaggle."""

from __future__ import annotations

import argparse
from dataclasses import dataclass


@dataclass
class TrainConfig:
    model: str = "Qwen/Qwen2.5-1.5B-Instruct"
    data_dir: str = "data/synthetic"
    out_dir: str = "outputs/lora"
    load_in_4bit: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    epochs: int = 2
    lr: float = 2e-4
    per_device_batch: int = 8
    grad_accum: int = 2
    max_seq_len: int = 512
    seed: int = 0

    def __post_init__(self) -> None:
        if self.lora_r < 1:
            raise ValueError("lora_r deve ser >= 1")
        if self.epochs < 1:
            raise ValueError("epochs deve ser >= 1")


def parse_args(argv: list[str] | None = None) -> TrainConfig:
    d = TrainConfig()
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=d.model)
    ap.add_argument("--data-dir", default=d.data_dir)
    ap.add_argument("--out-dir", default=d.out_dir)
    ap.add_argument("--no-4bit", action="store_true")
    ap.add_argument("--lora-r", type=int, default=d.lora_r)
    ap.add_argument("--lora-alpha", type=int, default=d.lora_alpha)
    ap.add_argument("--epochs", type=int, default=d.epochs)
    ap.add_argument("--lr", type=float, default=d.lr)
    ap.add_argument("--per-device-batch", type=int, default=d.per_device_batch)
    ap.add_argument("--grad-accum", type=int, default=d.grad_accum)
    ap.add_argument("--max-seq-len", type=int, default=d.max_seq_len)
    ap.add_argument("--seed", type=int, default=d.seed)
    a = ap.parse_args(argv)
    return TrainConfig(
        model=a.model, data_dir=a.data_dir, out_dir=a.out_dir, load_in_4bit=not a.no_4bit,
        lora_r=a.lora_r, lora_alpha=a.lora_alpha, epochs=a.epochs, lr=a.lr,
        per_device_batch=a.per_device_batch, grad_accum=a.grad_accum,
        max_seq_len=a.max_seq_len, seed=a.seed,
    )


def precision_flags(precision: str) -> dict[str, bool]:
    return {"fp16": precision == "fp16", "bf16": precision == "bf16"}


def trainable_dtypes(model) -> dict[str, int]:
    out: dict[str, int] = {}
    for p in model.parameters():
        if p.requires_grad:
            k = str(p.data.dtype).removeprefix("torch.")
            out[k] = out.get(k, 0) + 1
    return out


def cast_trainable_to_fp32(model) -> int:
    """GradScaler (fp16) não faz unscale de gradientes bf16; adapters LoRA devem ser fp32."""
    n = 0
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()
            n += 1
    return n


def _to_prompt_completion(row: dict) -> dict:
    *prompt, answer = row["messages"]
    return {"prompt": prompt, "completion": [answer]}


def main(argv: list[str] | None = None) -> None:
    cfg = parse_args(argv)
    from datasets import load_dataset
    from peft import LoraConfig
    from trl import SFTConfig, SFTTrainer

    from .hf_runtime import current_precision, load

    model, tok = load(cfg.model, load_in_4bit=cfg.load_in_4bit)
    ds = load_dataset(
        "json", data_files={"train": f"{cfg.data_dir}/train.jsonl", "val": f"{cfg.data_dir}/val.jsonl"}
    ).map(_to_prompt_completion, remove_columns=["messages"])

    trainer = SFTTrainer(
        model=model,
        processing_class=tok,
        train_dataset=ds["train"],
        eval_dataset=ds["val"],
        peft_config=LoraConfig(
            r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
            target_modules="all-linear", task_type="CAUSAL_LM",
        ),
        args=SFTConfig(
            output_dir=cfg.out_dir, num_train_epochs=cfg.epochs, learning_rate=cfg.lr,
            per_device_train_batch_size=cfg.per_device_batch,
            gradient_accumulation_steps=cfg.grad_accum, max_length=cfg.max_seq_len,
            eval_strategy="epoch", save_strategy="epoch", logging_steps=20,
            seed=cfg.seed, report_to="none", **precision_flags(current_precision()),
        ),
    )
    precision = current_precision()
    print(f"precisão={precision} | dtypes treináveis antes: {trainable_dtypes(trainer.model)}")
    if precision == "fp16":
        cast_trainable_to_fp32(trainer.model)
        print(f"dtypes treináveis depois: {trainable_dtypes(trainer.model)}")
    trainer.train()
    trainer.save_model(cfg.out_dir)


if __name__ == "__main__":
    main()
