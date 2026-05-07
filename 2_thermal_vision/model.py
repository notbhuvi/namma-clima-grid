"""
Module 2a — CNN-ViT Hybrid Thermal Vision Engine
=================================================

Predicts Land Surface Temperature (LST) and a heat-stress score from
multi-spectral ward patches, and produces a compact spatial embedding
for use as node features in the Module 3 Spatio-Temporal GNN.

Architecture
─────────────
  Input: (B, C=7, H=64, W=64)  — multi-band ward patch
        │
        ├─ CNN Branch ─────────────────────────────────────────────────
        │  ResNet50 backbone (ImageNet-pretrained, first conv re-init
        │  to handle C≠3 channels via weight averaging)
        │  GlobalAvgPool → (B, 2048)
        │
        ├─ ViT Branch ─────────────────────────────────────────────────
        │  PatchEmbed(patch_size=8) → 64 spatial tokens of dim 256
        │  Prepend learnable [CLS] token → 65 tokens
        │  Add 2D positional embedding
        │  6 × TransformerEncoderLayer(d_model=256, nhead=8, ffn=1024)
        │  Take [CLS] → (B, 256)
        │
        └─ Fusion ──────────────────────────────────────────────────────
           cat([CNN, ViT]) → (B, 2304)
           LayerNorm → FC(2304→512) → GELU → Dropout(0.1)
           FC(512→256) → GELU
           Three parallel output heads:
             * lst_pred       : FC(256→1)          — Celsius regression
             * heat_stress    : FC(256→1)+Sigmoid  — 0‒100 score
             * ward_embedding : FC(256→128)+L2-norm — GNN node feature

Input channels
──────────────
  0  NDVI             (−1..1,  normalised)
  1  NDBI             (−1..1,  normalised)
  2  impervious_pct   (0..100, normalised)
  3  rainfall_mm_24h  (mm,     log-normalised)
  4  pm25_ugm3        (μg/m³,  log-normalised)
  5  population_den   (people/km², log-normalised)
  6  heat_proxy       (synthetic UHI gradient, 0..1)

Target
──────
  lst_celsius — Land Surface Temperature in °C
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Number of input spectral / derived channels
N_CHANNELS: int = 7
PATCH_SIZE: int = 64   # spatial resolution of each ward patch (pixels)
VIT_DIM: int = 256     # ViT embedding dimension
VIT_DEPTH: int = 6     # ViT transformer depth
VIT_HEADS: int = 8     # ViT attention heads
VIT_PATCH: int = 8     # ViT spatial patch size (pixels)
CNN_DIM: int = 2048    # ResNet50 output dim (after GlobalAvgPool)
EMBED_DIM: int = 128   # Ward embedding dim (GNN node feature size)


# ---------------------------------------------------------------------------
# CNN Branch — ResNet50 with channel-adapted first conv
# ---------------------------------------------------------------------------
class ChannelAdaptedResNet50(nn.Module):
    """
    ResNet50 backbone whose first convolution is re-initialised to accept
    an arbitrary number of input channels.

    For pretrained=True the extra channels are initialised by averaging the
    RGB weights so that zero-initialised channels don't kill the gradient and
    the spatial filters retain their ImageNet-learned structure.
    """

    def __init__(self, in_channels: int = N_CHANNELS, pretrained: bool = True) -> None:
        super().__init__()
        try:
            import timm  # type: ignore
            self.backbone = timm.create_model(
                "resnet50", pretrained=pretrained,
                num_classes=0, global_pool="avg",
            )
        except ImportError:
            import torchvision.models as tvm  # type: ignore
            weights = tvm.ResNet50_Weights.DEFAULT if pretrained else None
            base = tvm.resnet50(weights=weights)
            # Strip the classifier; add a global avg-pool wrapper
            self.backbone = nn.Sequential(
                *list(base.children())[:-1],   # up to AdaptiveAvgPool
                nn.Flatten(),
            )

        # Patch first conv if in_channels ≠ 3
        if in_channels != 3:
            self._adapt_first_conv(in_channels, pretrained)

        self.out_dim: int = CNN_DIM

    # ------------------------------------------------------------------
    def _adapt_first_conv(self, in_channels: int, pretrained: bool) -> None:
        """Replace the first 3-channel conv with an in_channels-channel one."""
        # Locate the first Conv2d (timm names it 'conv1'; torchvision has it at [0][0])
        old: Optional[nn.Conv2d] = None
        if hasattr(self.backbone, "conv1"):
            old = self.backbone.conv1
            key = "conv1"
        elif isinstance(self.backbone, nn.Sequential) and isinstance(
            self.backbone[0], nn.Conv2d
        ):
            old = self.backbone[0]
            key = "0"
        else:
            raise RuntimeError("Cannot locate first Conv2d in backbone")

        new_conv = nn.Conv2d(
            in_channels, old.out_channels,
            kernel_size=old.kernel_size, stride=old.stride,
            padding=old.padding, bias=False,
        )

        if pretrained and old.weight.shape[1] == 3:
            with torch.no_grad():
                # Average over RGB → (out_ch, 1, kH, kW) then tile
                avg = old.weight.data.mean(dim=1, keepdim=True)
                new_conv.weight.data = avg.repeat(1, in_channels, 1, 1)
                # Preserve expected output scale
                new_conv.weight.data *= 3.0 / in_channels

        if key == "conv1":
            self.backbone.conv1 = new_conv
        else:
            self.backbone[0] = new_conv

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return (B, 2048) feature vector."""
        return self.backbone(x)


# ---------------------------------------------------------------------------
# ViT Branch — lightweight Vision Transformer
# ---------------------------------------------------------------------------

class _PatchEmbed(nn.Module):
    """Embed spatial patches via a single strided convolution."""

    def __init__(self, in_channels: int, img_size: int, patch_size: int, embed_dim: int) -> None:
        super().__init__()
        assert img_size % patch_size == 0, "img_size must be divisible by patch_size"
        self.n_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W) → (B, n_patches, embed_dim)
        x = self.proj(x)          # (B, embed_dim, H/P, W/P)
        return x.flatten(2).transpose(1, 2)  # (B, n_patches, embed_dim)


class _TransformerBlock(nn.Module):
    """Standard Pre-LN Transformer encoder block."""

    def __init__(self, dim: int, n_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        mlp_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim), nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm1(x)
        x, _ = self.attn(x, x, x)
        x = residual + x
        x = x + self.mlp(self.norm2(x))
        return x


class LSTViTEncoder(nn.Module):
    """
    Compact Vision Transformer for spatial context encoding of ward patches.

    Returns the [CLS] token embedding — a global summary of the spatial
    distribution of spectral properties within the ward.
    """

    def __init__(
        self,
        in_channels: int = N_CHANNELS,
        img_size: int = PATCH_SIZE,
        patch_size: int = VIT_PATCH,
        embed_dim: int = VIT_DIM,
        depth: int = VIT_DEPTH,
        n_heads: int = VIT_HEADS,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.patch_embed = _PatchEmbed(in_channels, img_size, patch_size, embed_dim)
        n_patches = self.patch_embed.n_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(dropout)

        self.blocks = nn.Sequential(
            *[_TransformerBlock(embed_dim, n_heads, dropout=dropout) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.out_dim = embed_dim

        self._init_weights()

    # ------------------------------------------------------------------
    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        # 2D sinusoidal positional embedding
        n = int(math.sqrt(self.patch_embed.n_patches))
        pos_embed = self._sincos2d(n, n, self.pos_embed.shape[-1])
        self.pos_embed.data[:, 1:, :] = pos_embed.unsqueeze(0)
        # CLS position at index 0 stays as zeros

    @staticmethod
    def _sincos2d(h: int, w: int, dim: int) -> torch.Tensor:
        assert dim % 4 == 0
        half = dim // 2
        ys = torch.arange(h, dtype=torch.float32)
        xs = torch.arange(w, dtype=torch.float32)
        freq = torch.arange(half // 2, dtype=torch.float32) / (half // 2)
        freq = 1.0 / (10000 ** freq)
        # (h, half//2)
        y_enc = torch.outer(ys, freq)
        x_enc = torch.outer(xs, freq)
        y_enc = torch.cat([y_enc.sin(), y_enc.cos()], dim=-1)  # (h, half)
        x_enc = torch.cat([x_enc.sin(), x_enc.cos()], dim=-1)  # (w, half)
        pos = torch.cat([
            y_enc.unsqueeze(1).expand(h, w, half),
            x_enc.unsqueeze(0).expand(h, w, half),
        ], dim=-1)  # (h, w, dim)
        return pos.view(h * w, dim)

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        tokens = self.patch_embed(x)                       # (B, n_patches, D)
        cls = self.cls_token.expand(B, -1, -1)             # (B, 1, D)
        tokens = torch.cat([cls, tokens], dim=1)           # (B, n_patches+1, D)
        tokens = self.pos_drop(tokens + self.pos_embed)
        tokens = self.blocks(tokens)
        tokens = self.norm(tokens)
        return tokens[:, 0]                                # (B, D) — [CLS] token


# ---------------------------------------------------------------------------
# Fusion Head
# ---------------------------------------------------------------------------

class _FusionHead(nn.Module):
    """
    MLP that fuses CNN and ViT features and produces three outputs:
      lst_pred:        (B, 1)   regression target (°C)
      heat_stress:     (B, 1)   0–100 score
      ward_embedding:  (B, 128) L2-normalised spatial descriptor for GNN
    """

    def __init__(self, cnn_dim: int, vit_dim: int, hidden: int = 512, embed_dim: int = EMBED_DIM) -> None:
        super().__init__()
        fused = cnn_dim + vit_dim
        self.norm  = nn.LayerNorm(fused)
        self.fc1   = nn.Linear(fused, hidden)
        self.act1  = nn.GELU()
        self.drop1 = nn.Dropout(0.3)
        self.fc2   = nn.Linear(hidden, 256)
        self.act2  = nn.GELU()
        self.drop2 = nn.Dropout(0.3)

        # Output heads
        self.lst_head   = nn.Linear(256, 1)
        self.hs_head    = nn.Sequential(nn.Linear(256, 1), nn.Sigmoid())
        self.embed_head = nn.Linear(256, embed_dim)

    def forward(
        self, cnn_feat: torch.Tensor, vit_feat: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        x = torch.cat([cnn_feat, vit_feat], dim=-1)
        x = self.norm(x)
        x = self.drop1(self.act1(self.fc1(x)))
        x = self.drop2(self.act2(self.fc2(x)))

        lst_pred      = self.lst_head(x).squeeze(-1)        # (B,)
        heat_stress   = self.hs_head(x).squeeze(-1) * 100   # (B,) → 0..100
        ward_emb      = F.normalize(self.embed_head(x), p=2, dim=-1)  # (B, 128)

        return {
            "lst_pred":       lst_pred,
            "heat_stress":    heat_stress,
            "ward_embedding": ward_emb,
        }


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class HybridLSTPredictor(nn.Module):
    """
    CNN-ViT hybrid model for per-ward LST prediction.

    Args:
        in_channels: Number of input spectral channels (default 7).
        img_size:    Spatial resolution of each ward patch (default 64).
        embed_dim:   Dimension of the ward embedding vector (default 128).
        pretrained:  Load ImageNet weights for the ResNet50 backbone.
    """

    def __init__(
        self,
        in_channels: int = N_CHANNELS,
        img_size: int = PATCH_SIZE,
        embed_dim: int = EMBED_DIM,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        self.cnn = ChannelAdaptedResNet50(in_channels=in_channels, pretrained=pretrained)
        self.vit = LSTViTEncoder(in_channels=in_channels, img_size=img_size)
        self.fusion = _FusionHead(
            cnn_dim=self.cnn.out_dim,
            vit_dim=self.vit.out_dim,
            embed_dim=embed_dim,
        )

        self.in_channels = in_channels
        self.img_size    = img_size
        self.embed_dim   = embed_dim

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: (B, C, H, W) normalised multi-band ward patch.

        Returns:
            dict with keys:
              'lst_pred'      : (B,) — predicted LST in Celsius
              'heat_stress'   : (B,) — 0–100 heat stress score
              'ward_embedding': (B, embed_dim) — L2-normalised spatial descriptor
        """
        cnn_feat = self.cnn(x)   # (B, 2048)
        vit_feat = self.vit(x)   # (B, 256)
        return self.fusion(cnn_feat, vit_feat)

    # ------------------------------------------------------------------
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return only the ward embedding (for GNN node feature extraction)."""
        return self.forward(x)["ward_embedding"]

    # ------------------------------------------------------------------
    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Convenience inference wrapper (disables grad)."""
        self.eval()
        return self.forward(x)

    # ------------------------------------------------------------------
    def parameter_count(self) -> Dict[str, int]:
        """Return trainable parameter counts per sub-module."""
        def _count(m: nn.Module) -> int:
            return sum(p.numel() for p in m.parameters() if p.requires_grad)
        return {
            "cnn":    _count(self.cnn),
            "vit":    _count(self.vit),
            "fusion": _count(self.fusion),
            "total":  _count(self),
        }


# ---------------------------------------------------------------------------
# __main__ — architecture smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    from loguru import logger

    logger.info("HybridLSTPredictor — architecture smoke test")

    model = HybridLSTPredictor(pretrained=False)
    params = model.parameter_count()
    logger.info(
        f"Parameters | CNN={params['cnn']:,}  ViT={params['vit']:,}  "
        f"Fusion={params['fusion']:,}  Total={params['total']:,}"
    )

    # Forward pass with a batch of 4 synthetic ward patches
    dummy = torch.randn(4, N_CHANNELS, PATCH_SIZE, PATCH_SIZE)
    out = model(dummy)

    assert out["lst_pred"].shape       == (4,),          "lst_pred shape error"
    assert out["heat_stress"].shape    == (4,),           "heat_stress shape error"
    assert out["ward_embedding"].shape == (4, EMBED_DIM), "ward_embedding shape error"
    assert out["heat_stress"].min() >= 0 and out["heat_stress"].max() <= 100, \
        "heat_stress out of [0,100]"
    # Ward embeddings should be L2-normalised
    norms = out["ward_embedding"].norm(dim=-1)
    assert torch.allclose(norms, torch.ones(4), atol=1e-5), "ward_embedding not L2-normalised"

    logger.success(
        f"Outputs | lst_pred={out['lst_pred'].tolist()} | "
        f"heat_stress={out['heat_stress'].tolist()[:2]} | "
        f"ward_embedding[0][:4]={out['ward_embedding'][0,:4].tolist()}"
    )
    logger.success("model.py smoke test passed.")
