from __future__ import annotations

import torch

from backbones.foundationstereo_adapter import (
    _extract_state_dict,
    _fallback_flash_attn_func,
    _fallback_flash_attn_qkvpacked_func,
    infer_foundation_stereo_vit_size,
)


def test_training_checkpoint_model_dict_and_vits_inference() -> None:
    patch = torch.empty(384, 3, 14, 14)
    state, fmt = _extract_state_dict(
        {"model": {"feature.dino.depth_anything.pretrained.patch_embed.proj.weight": patch}}
    )
    assert fmt == "dict[model]"
    assert infer_foundation_stereo_vit_size(state) == "vits"


def test_vitl_is_inferred_from_patch_embedding_width() -> None:
    patch = torch.empty(1024, 3, 14, 14)
    assert infer_foundation_stereo_vit_size(
        {"feature.dino.depth_anything.pretrained.patch_embed.proj.weight": patch}
    ) == "vitl"


def test_flash_attention_fallback_preserves_unpacked_and_packed_shapes() -> None:
    query = torch.randn(2, 7, 4, 8)
    output = _fallback_flash_attn_func(query, query, query)
    assert output.shape == query.shape

    packed = torch.stack([query, query, query], dim=2)
    packed_output = _fallback_flash_attn_qkvpacked_func(packed)
    assert packed_output.shape == query.shape
