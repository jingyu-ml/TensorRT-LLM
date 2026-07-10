# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Unit tests for Wan2.2 NVFP4 SVDQuant (``NVFP4_SVD``) checkpoint loading:
- SVDQuant checkpoints build self-attention with SEPARATE_QKV (per-projection
  rank-r LoRA factors cannot be concatenated into the fused-QKV projection);
  plain NVFP4 keeps the fused QKV projection.
- load_weights swaps quantized Linears to NVFP4SVDLinearMethod, loads the LoRA
  factors, and activates the fused kernel for rank 32 (reference fallback for
  other ranks); NVFP4-excluded modules are untouched.
- The MLP GELU epilogue fusion must not bypass a SVDQuant projection.
"""

from types import SimpleNamespace

import pytest
import torch

import tensorrt_llm  # noqa: F401  (registers the trtllm torch ops)
from tensorrt_llm._torch.modules.linear import Linear, NVFP4SVDLinearMethod
from tensorrt_llm._torch.visual_gen.config import DiffusionModelConfig, DiffusionPipelineConfig
from tensorrt_llm._torch.visual_gen.models.wan.transformer_wan import WanTransformer3DModel
from tensorrt_llm._torch.visual_gen.modules.attention import QKVMode
from tensorrt_llm._torch.visual_gen.quantization.ops import quantize_nvfp4
from tensorrt_llm.visual_gen.args import AttentionConfig

_IS_SM100 = torch.cuda.is_available() and torch.cuda.get_device_capability() == (10, 0)
skip_sm100 = pytest.mark.skipif(
    not _IS_SM100, reason="NVFP4 SVDQuant kernels require SM100 (Blackwell)"
)

# Tiny Wan: hidden 256 = 2 heads x 128 head_dim, K/N multiples of 256 so both
# the fused SVDQuant GEMM and the split norm+RoPE kernel envelopes are honored.
_HEADS = 2
_HEAD_DIM = 128
_HIDDEN = _HEADS * _HEAD_DIM
_FFN = 512
_LAYERS = 2
_RANK = 32


def _quantization_config(quant_algo: str) -> dict:
    return {
        "config_groups": {
            "group_0": {
                "input_activations": {"dynamic": False, "num_bits": 4, "group_size": 16},
                "weights": {"dynamic": False, "num_bits": 4, "group_size": 16},
                "targets": ["Linear"],
                **({"lora_rank": _RANK} if quant_algo == "NVFP4_SVD" else {}),
            }
        },
        "ignore": ["blocks.0*", "condition_embedder*", "patch_embedding", "proj_out"],
        "quant_algo": quant_algo,
        "quant_method": "modelopt",
    }


def _make_model(quant_algo: str = "NVFP4_SVD") -> WanTransformer3DModel:
    pretrained_config = SimpleNamespace(
        _name_or_path="tiny-wan",
        num_attention_heads=_HEADS,
        attention_head_dim=_HEAD_DIM,
        num_layers=_LAYERS,
        ffn_dim=_FFN,
        text_dim=96,
        freq_dim=32,
        in_channels=4,
        out_channels=4,
        patch_size=[1, 2, 2],
        eps=1e-6,
        cross_attn_norm=True,
        qk_norm="rms_norm_across_heads",
        quantization_config=_quantization_config(quant_algo),
    )
    quant_config, quant_config_dict, dynamic_weight_quant, dynamic_activation_quant = (
        DiffusionPipelineConfig.load_diffusion_quant_config(pretrained_config.quantization_config)
    )
    model_config = DiffusionModelConfig(
        component_name="transformer",
        pretrained_config=pretrained_config,
        quant_config=quant_config,
        quant_config_dict=quant_config_dict,
        dynamic_weight_quant=dynamic_weight_quant,
        force_dynamic_quantization=dynamic_activation_quant,
        skip_create_weights_in_init=True,
        attention=AttentionConfig(backend="VANILLA"),
    )
    return WanTransformer3DModel(model_config).to("cuda").eval()


def _is_svdquant_target(module) -> bool:
    return (
        isinstance(module, Linear)
        and module.quant_config is not None
        and module.quant_config.quant_algo is not None
    )


def _ckpt_name(module_name: str) -> str:
    """TRT-LLM module name -> diffusers checkpoint name (inverse load remap)."""
    return module_name.replace(".ffn.up_proj", ".ffn.net.0.proj").replace(
        ".ffn.down_proj", ".ffn.net.2"
    )


def _build_svdquant_state_dict(model: WanTransformer3DModel, rank: int) -> dict:
    torch.manual_seed(0)
    weights = {}
    covered = set()
    # ModelOpt's self-attn calibration yields bit-identical pre_quant_scale /
    # input_scale across a self-attention's q/k/v (same input distribution).
    # Mirror that for attn1 triples so the shared-quantize path is exercised;
    # give every other projection its own scales (negative case for attn2).
    shared_pqs = {}
    for name, module in model.named_modules():
        if not _is_svdquant_target(module):
            continue
        out_f, in_f = module.out_features, module.in_features
        w_ref = (torch.randn(out_f, in_f, device="cuda") * 0.02).to(torch.bfloat16)
        qweight, weight_scale, weight_scale_2 = quantize_nvfp4(w_ref)
        ck = _ckpt_name(name)
        parent, _, leaf = name.rpartition(".")
        if parent.endswith(".attn1") and leaf in ("to_q", "to_k", "to_v"):
            if parent not in shared_pqs:
                shared_pqs[parent] = (torch.rand(in_f) * 0.5 + 0.75).to(torch.bfloat16)
            pre_quant_scale = shared_pqs[parent]
        else:
            pre_quant_scale = (torch.rand(in_f) * 0.5 + 0.75).to(torch.bfloat16)
        weights[f"{ck}.weight"] = qweight.cpu()
        weights[f"{ck}.weight_scale"] = weight_scale.cpu()
        weights[f"{ck}.weight_scale_2"] = weight_scale_2.cpu()
        weights[f"{ck}.input_scale"] = torch.tensor(0.005, dtype=torch.float32)
        weights[f"{ck}.pre_quant_scale"] = pre_quant_scale
        weights[f"{ck}.svdquant_lora_a"] = (torch.randn(rank, in_f) * 0.01).to(torch.bfloat16)
        weights[f"{ck}.svdquant_lora_b"] = (torch.randn(out_f, rank) * 0.01).to(torch.bfloat16)
        weights[f"{ck}.bias"] = torch.zeros(out_f, dtype=torch.bfloat16)
        covered.update(f"{name}.{param_name}" for param_name, _ in module.named_parameters())
    for name, param in model.named_parameters():
        if name in covered:
            continue
        ck = _ckpt_name(name)
        if "norm" in name and name.endswith(".weight"):
            weights[ck] = torch.ones(param.shape, dtype=torch.bfloat16)
        elif name.endswith(".bias"):
            weights[ck] = torch.zeros(param.shape, dtype=torch.bfloat16)
        else:
            weights[ck] = (torch.randn(param.shape) * 0.02).to(torch.bfloat16)
    return weights


def _block_inputs(seq_len: int = 64, text_len: int = 8):
    torch.manual_seed(1)
    device = torch.device("cuda")
    x = torch.randn(1, seq_len, _HIDDEN, dtype=torch.bfloat16, device=device)
    encoder = torch.randn(1, text_len, _HIDDEN, dtype=torch.bfloat16, device=device)
    temb = torch.randn(1, 6, _HIDDEN, dtype=torch.bfloat16, device=device)
    freqs_cos = torch.ones(seq_len, _HEAD_DIM, dtype=torch.float32, device=device)
    freqs_sin = torch.zeros(seq_len, _HEAD_DIM, dtype=torch.float32, device=device)
    return x, encoder, temb, freqs_cos, freqs_sin


def _sqnr_db(ref: torch.Tensor, got: torch.Tensor) -> float:
    err = (ref - got).float()
    noise = (err**2).mean()
    if noise == 0:
        return float("inf")
    return float(10 * torch.log10((ref.float() ** 2).mean() / noise))


@skip_sm100
def test_svdquant_checkpoint_builds_separate_qkv():
    model = _make_model("NVFP4_SVD")
    attn1 = model.blocks[0].attn1
    assert attn1.qkv_mode == QKVMode.SEPARATE_QKV
    assert hasattr(attn1, "to_q") and not hasattr(attn1, "qkv_proj")


@skip_sm100
def test_plain_nvfp4_checkpoint_keeps_fused_qkv():
    model = _make_model("NVFP4")
    attn1 = model.blocks[0].attn1
    assert attn1.qkv_mode == QKVMode.FUSE_QKV
    assert hasattr(attn1, "qkv_proj")


@skip_sm100
def test_load_weights_swaps_method_and_activates_fused_kernel():
    model = _make_model("NVFP4_SVD")
    model.load_weights(_build_svdquant_state_dict(model, rank=_RANK))
    model.post_load_weights()

    quantized = model.blocks[1].attn1.to_q
    assert isinstance(quantized.quant_method, NVFP4SVDLinearMethod)
    assert quantized.svdquant_lora_a is not None
    assert quantized.svdquant_lora_a.shape[0] == _RANK
    assert quantized._svdquant_use_fused

    ffn_up = model.blocks[1].ffn.up_proj
    assert isinstance(ffn_up.quant_method, NVFP4SVDLinearMethod)
    assert ffn_up._svdquant_use_fused

    # Self-attn q/k/v share bit-identical scales -> one shared smoothed
    # quantize; cross-attn to_q has its own scales -> no sharing.
    assert model.blocks[1].attn1._svdquant_share_qkv_quantize
    assert not model.blocks[1].attn2._svdquant_share_qkv_quantize

    # blocks.0* is in the quantization ignore list: stays unquantized/unswapped.
    excluded = model.blocks[0].attn1.to_q
    assert not isinstance(excluded.quant_method, NVFP4SVDLinearMethod)
    assert excluded.weight.dtype == torch.bfloat16


@skip_sm100
def test_rank64_loads_and_falls_back_to_reference_path():
    model = _make_model("NVFP4_SVD")
    model.load_weights(_build_svdquant_state_dict(model, rank=64))
    model.post_load_weights()

    quantized = model.blocks[1].attn1.to_q
    assert isinstance(quantized.quant_method, NVFP4SVDLinearMethod)
    assert quantized.svdquant_lora_a.shape[0] == 64
    assert not quantized._svdquant_use_fused
    # Sharing requires the fused path; the reference fallback keeps
    # per-projection quantization.
    assert not model.blocks[1].attn1._svdquant_share_qkv_quantize

    x, encoder, temb, freqs_cos, freqs_sin = _block_inputs()
    with torch.inference_mode():
        out = model.blocks[1](x, encoder, temb, freqs_cos, freqs_sin)
    assert torch.isfinite(out.float()).all()


@skip_sm100
def test_block_forward_fused_matches_reference():
    model = _make_model("NVFP4_SVD")
    model.load_weights(_build_svdquant_state_dict(model, rank=_RANK))
    model.post_load_weights()
    block = model.blocks[1]

    x, encoder, temb, freqs_cos, freqs_sin = _block_inputs()
    with torch.inference_mode():
        out_fused = block(x, encoder, temb, freqs_cos, freqs_sin)

    svd_linears = [
        module
        for module in block.modules()
        if isinstance(module, Linear)
        and isinstance(module.quant_method, NVFP4SVDLinearMethod)
        and module._svdquant_use_fused
    ]
    assert svd_linears, "expected fused SVDQuant Linears in the quantized block"
    for module in svd_linears:
        module._svdquant_use_fused = False
    # Keep flags consistent with the reference path (finalize ties sharing to
    # the fused path in production).
    block.attn1._svdquant_share_qkv_quantize = False
    with torch.inference_mode():
        out_reference = block(x, encoder, temb, freqs_cos, freqs_sin)

    assert torch.isfinite(out_fused.float()).all()
    assert _sqnr_db(out_reference, out_fused) > 25.0


@skip_sm100
def test_mlp_does_not_take_fused_gelu_epilogue_with_svdquant(monkeypatch):
    model = _make_model("NVFP4_SVD")
    model.load_weights(_build_svdquant_state_dict(model, rank=_RANK))
    model.post_load_weights()
    ffn = model.blocks[1].ffn
    assert ffn.up_proj.svdquant_lora_a is not None

    def _must_not_fuse(*args, **kwargs):
        raise AssertionError("MLP fused GELU epilogue must not run for SVDQuant projections")

    monkeypatch.setattr(type(ffn), "_fused_gelu", _must_not_fuse)
    x = torch.randn(64, _HIDDEN, dtype=torch.bfloat16, device="cuda")
    with torch.inference_mode():
        out = ffn(x)
    assert torch.isfinite(out.float()).all()
