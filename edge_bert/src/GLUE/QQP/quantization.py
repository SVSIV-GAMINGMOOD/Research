from __future__ import annotations

from copy import deepcopy

import torch
import torch.nn as nn

from GLUE.QQP.task_config import QUANTIZATION_SETTINGS


EMBEDDING_BITS = QUANTIZATION_SETTINGS["embedding_bits"]


def fake_quantize_tensor(tensor: torch.Tensor, bits: int) -> torch.Tensor:
    qmin = 0.0
    qmax = 2.0**bits - 1.0
    min_val, max_val = tensor.min(), tensor.max()
    scale = (max_val - min_val) / (qmax - qmin + 1e-8)
    zero_point = qmin - min_val / (scale + 1e-8)
    q_tensor = zero_point + tensor / (scale + 1e-8)
    q_tensor.clamp_(qmin, qmax).round_()
    return scale * (q_tensor - zero_point)


def quantize_tensor_symmetric(tensor: torch.Tensor, bits: int) -> torch.Tensor:
    qmin = -(2 ** (bits - 1))
    qmax = (2 ** (bits - 1)) - 1
    scale = tensor.abs().max() / max(qmax, 1)
    scale = scale + 1e-8
    q = torch.round(tensor / scale).clamp(qmin, qmax)
    return q * scale


def apply_fake_quantization_to_linear_weights(
    model: nn.Module,
    bit_config: dict[str, int],
    *,
    locked_layers: set[str] | None = None,
) -> nn.Module:
    locked = locked_layers or set()
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            bits = 8 if name in locked else bit_config.get(name, 8)
            module.weight.data = fake_quantize_tensor(module.weight.data, bits)
    return model


def build_quantized_checkpoint_state(
    model: nn.Module,
    bit_config: dict[str, int],
    *,
    locked_layers: set[str] | None = None,
) -> dict[str, torch.Tensor]:
    quantized_model = deepcopy(model)
    locked = locked_layers or set()
    for name, module in quantized_model.named_modules():
        if isinstance(module, nn.Linear):
            bits = 8 if name in locked else bit_config.get(name, 8)
            module.weight.data = quantize_tensor_symmetric(module.weight.data, bits)
    return quantized_model.state_dict()


class QuantizedLinear(nn.Module):
    def __init__(self, linear_layer: nn.Linear, bits: int):
        super().__init__()
        self.bits = bits
        weight = linear_layer.weight.data
        q_min = 0
        q_max = 2**bits - 1
        w_min = weight.min()
        w_max = weight.max()
        scale = (w_max - w_min) / (q_max - q_min + 1e-8)
        zero_point = torch.round(q_min - w_min / (scale + 1e-8))
        q_weight = torch.clamp(torch.round(weight / scale + zero_point), q_min, q_max).to(torch.int32)
        self.register_buffer("q_weight", q_weight)
        self.register_buffer("scale", scale)
        self.register_buffer("zero_point", zero_point)
        if linear_layer.bias is not None:
            self.register_buffer("bias", linear_layer.bias.data.clone())
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.scale * (self.q_weight.float() - self.zero_point)
        return nn.functional.linear(x, weight, self.bias)


class QuantizedEmbedding(nn.Module):
    def __init__(self, embedding_layer: nn.Embedding):
        super().__init__()
        weight = embedding_layer.weight.data
        q_min, q_max = -128, 127
        abs_max = weight.abs().max(dim=1, keepdim=True).values + 1e-8
        scale = abs_max / q_max
        q_weight = torch.clamp(torch.round(weight / scale), q_min, q_max).to(torch.int8)
        self.register_buffer("q_weight", q_weight)
        self.register_buffer("scale", scale)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        q_rows = self.q_weight[input_ids]
        scales = self.scale[input_ids]
        return q_rows.float() * scales


def replace_linear_layers(
    model: nn.Module,
    bit_config: dict[str, int],
    *,
    locked_layers: set[str] | None = None,
) -> nn.Module:
    locked = locked_layers or set()
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear):
            bits = 8 if name in locked else bit_config.get(name, 8)
            parent = model
            parts = name.split(".")
            for part in parts[:-1]:
                parent = getattr(parent, part)
            setattr(parent, parts[-1], QuantizedLinear(module, bits))
    return model


def replace_embedding_layers(model: nn.Module) -> nn.Module:
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Embedding):
            parent = model
            parts = name.split(".")
            for part in parts[:-1]:
                parent = getattr(parent, part)
            setattr(parent, parts[-1], QuantizedEmbedding(module))
    return model
