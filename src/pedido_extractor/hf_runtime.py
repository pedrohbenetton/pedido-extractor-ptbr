"""Carrega modelo (base ou base+LoRA) e expõe `generate_fn`. Imports pesados são preguiçosos."""

from __future__ import annotations

from .evaluation import GenerateFn


def load(model: str, adapter: str | None = None, load_in_4bit: bool = True):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tok = AutoTokenizer.from_pretrained(model)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    quant = (
        BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16
        )
        if load_in_4bit
        else None
    )
    m = AutoModelForCausalLM.from_pretrained(model, quantization_config=quant, device_map="auto")
    if adapter:
        from peft import PeftModel

        m = PeftModel.from_pretrained(m, adapter)
    m.eval()
    return m, tok


def make_generate_fn(model, tok, max_new_tokens: int = 256) -> GenerateFn:
    import torch

    def gen(batch):
        prompts = [
            tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True) for msgs in batch
        ]
        enc = tok(prompts, return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False)
        return tok.batch_decode(out[:, enc["input_ids"].shape[1] :], skip_special_tokens=True)

    return gen
