import pytest

from pedido_extractor.train import TrainConfig, parse_args


def test_defaults_razoaveis_para_gpu_gratuita():
    c = parse_args([])
    assert c.model == "Qwen/Qwen2.5-1.5B-Instruct"
    assert c.load_in_4bit and c.lora_r == 16 and c.max_seq_len <= 1024
    assert c.per_device_batch * c.grad_accum >= 16


def test_overrides():
    c = parse_args(["--model", "x/y", "--epochs", "1", "--no-4bit", "--lora-r", "8"])
    assert (c.model, c.epochs, c.load_in_4bit, c.lora_r) == ("x/y", 1, False, 8)


def test_rejeita_valores_invalidos():
    with pytest.raises(ValueError):
        TrainConfig(lora_r=0)
    with pytest.raises(ValueError):
        TrainConfig(epochs=0)
