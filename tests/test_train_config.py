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


class _Tensor:
    def __init__(self, dtype):
        self.dtype = dtype

    def float(self):
        return _Tensor("float32")


class _Param:
    def __init__(self, dtype, requires_grad):
        self.data = _Tensor(dtype)
        self.requires_grad = requires_grad


class _Model:
    def __init__(self, params):
        self._params = params

    def parameters(self):
        return iter(self._params)


def test_cast_trainable_to_fp32_so_mexe_nos_treinaveis():
    from pedido_extractor.train import cast_trainable_to_fp32

    lora, congelado = _Param("bfloat16", True), _Param("float16", False)
    n = cast_trainable_to_fp32(_Model([lora, congelado]))
    assert n == 1
    assert lora.data.dtype == "float32" and congelado.data.dtype == "float16"


def test_resumo_de_dtypes_dos_treinaveis():
    from pedido_extractor.train import trainable_dtypes

    params = [_Param("bfloat16", True), _Param("bfloat16", True), _Param("float32", True), _Param("x", False)]
    assert trainable_dtypes(_Model(params)) == {"bfloat16": 2, "float32": 1}
