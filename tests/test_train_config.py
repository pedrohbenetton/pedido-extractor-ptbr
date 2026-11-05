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


@pytest.mark.parametrize(
    ("cc_major", "esperado"),
    [(7, "fp16"), (7.5, "fp16"), (8, "bf16"), (9, "bf16"), (None, "fp16")],
)
def test_precisao_pela_capacidade_da_gpu(cc_major, esperado):
    from pedido_extractor.hf_runtime import precision_for

    assert precision_for(cc_major) == esperado


def test_flags_de_precisao_sao_exclusivas():
    from pedido_extractor.train import precision_flags

    assert precision_flags("fp16") == {"fp16": True, "bf16": False}
    assert precision_flags("bf16") == {"fp16": False, "bf16": True}
