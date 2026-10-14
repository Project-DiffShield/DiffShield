"""Offline random-init tiny models for CPU tests and `--backend tiny`. Never downloads anything."""
from __future__ import annotations

import torch


def tiny_vae():
    from diffusers import AutoencoderKL

    torch.manual_seed(0)
    return AutoencoderKL(
        in_channels=3, out_channels=3,
        down_block_types=("DownEncoderBlock2D",) * 4,
        up_block_types=("UpDecoderBlock2D",) * 4,
        block_out_channels=(8, 8, 8, 16),  # 8x downsample like SD (cheap mid-block attention at 512 px)
        layers_per_block=1, latent_channels=4, norm_num_groups=8,
    )


def tiny_clip_vision(projection_dim: int = 16):
    from transformers import CLIPVisionConfig, CLIPVisionModelWithProjection

    torch.manual_seed(0)
    cfg = CLIPVisionConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=2,
                           image_size=224, patch_size=32, projection_dim=projection_dim)
    return CLIPVisionModelWithProjection(cfg)


def tiny_clip_text(projection_dim: int = 16):
    from transformers import CLIPTextConfig, CLIPTextModelWithProjection

    torch.manual_seed(0)
    cfg = CLIPTextConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=2,
                         vocab_size=100, max_position_embeddings=16, projection_dim=projection_dim,
                         bos_token_id=1, eos_token_id=99, pad_token_id=0)
    return CLIPTextModelWithProjection(cfg)


class _StubTokenizer:
    """Callable returning fixed token ids (no vocabulary files needed)."""

    def __call__(self, texts, return_tensors="pt", padding=True, truncation=True):
        n = len(texts)
        ids = torch.ones(n, 8, dtype=torch.long) * 5
        ids[:, 0] = 1
        ids[:, -1] = 99
        return {"input_ids": ids, "attention_mask": torch.ones(n, 8, dtype=torch.long)}


def tiny_target_embedding(dim: int = 16, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    e = torch.randn(1, dim, generator=g)
    return e / e.norm(dim=-1, keepdim=True)


def tiny_v1_loss(device: str = "cpu"):
    from src.losses import DiffShieldLoss

    return DiffShieldLoss(device=device, vae=tiny_vae(), clip_vision=tiny_clip_vision(),
                          clip_text=tiny_clip_text(), clip_tokenizer=_StubTokenizer())
