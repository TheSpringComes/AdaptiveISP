"""Network components for the AdaptiveISP Controller.

Separates the pure-nn.Module building blocks (FeatureExtractor CNN + the
critic head that consumes hand-crafted image stats) from the RL policy
logic that lives in `agent.py`.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from pipeline.state import PipelineState


class FeatureExtractor(nn.Module):
    """Downsampling CNN backbone (mirrors the original agent.FeatureExtractor)."""
    def __init__(
        self,
        in_channels: int,
        input_hw: int = 64,
        mid_channels: int = 32,
        output_dim: int = 4096,
        dropout_prob: float = 0.0,
    ) -> None:
        super().__init__()
        min_map = 4
        assert output_dim % (min_map ** 2) == 0
        self.output_dim = output_dim

        layers: list[nn.Module] = []
        size = input_hw // 2
        c_prev, c_now = in_channels, mid_channels
        layers += [
            nn.Conv2d(c_prev, c_now, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(c_now),
            nn.LeakyReLU(negative_slope=0.2),
        ]
        while size > min_map:
            c_prev = c_now
            c_now = output_dim // (min_map ** 2) if size == min_map * 2 else c_now * 2
            assert size % 2 == 0
            size //= 2
            layers += [
                nn.Conv2d(c_prev, c_now, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(c_now),
                nn.LeakyReLU(negative_slope=0.2),
            ]
        self.layers = nn.Sequential(*layers)
        self.dropout = nn.Dropout(p=dropout_prob) if dropout_prob > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layers(x)
        x = x.reshape(x.shape[0], self.output_dim)
        return self.dropout(x)


def pdf_sample(pdf: torch.Tensor, uniform_noise: torch.Tensor) -> torch.Tensor:
    """Inverse-CDF sample from a batched pdf, given per-row uniform noise."""
    pdf = pdf / (pdf.sum(dim=1, keepdim=True) + 1e-36)
    cdf_lower = torch.cumsum(pdf, dim=1) - pdf
    return (cdf_lower < uniform_noise).to(torch.int64).sum(dim=1) - 1


class AdaptiveISPValueNet(nn.Module):
    """Critic: image + state + hand-crafted stats -> scalar value.

    Mirrors the original value.Value: downsample image to obs_hw, compute
    luminance / contrast / saturation stats, broadcast state ⊕ stats as
    extra channels, run through a FeatureExtractor + 2 fc layers.
    """
    def __init__(
        self,
        n_ops: int,
        obs_hw: int = 64,
        mid_channels: int = 32,
        fc1_size: int = 128,
        feature_dim: int = 4096,
    ) -> None:
        super().__init__()
        n_state = 3 + n_ops       # step, stopped, has_reward + op_usage
        n_stats = 3               # luminance, contrast, saturation
        in_channels = 3 + n_state + n_stats

        self.obs_hw = obs_hw
        self.down_sample = nn.AdaptiveAvgPool2d((obs_hw, obs_hw))
        self.feature_extractor = FeatureExtractor(
            in_channels=in_channels, input_hw=obs_hw,
            mid_channels=mid_channels, output_dim=feature_dim,
        )
        self.fc1 = nn.Linear(feature_dim, fc1_size)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2)
        self.fc2 = nn.Linear(fc1_size, 1)

    def _state_channels(self, state: PipelineState, hw: int) -> torch.Tensor:
        B = state.batch_size
        pieces = [
            state.step.float().view(B, 1, 1, 1).expand(B, 1, hw, hw),
            state.stopped.float().view(B, 1, 1, 1).expand(B, 1, hw, hw),
            state.has_reward.float().view(B, 1, 1, 1).expand(B, 1, hw, hw),
            state.op_usage.float().view(B, -1, 1, 1).expand(B, state.op_usage.shape[1], hw, hw),
        ]
        return torch.cat(pieces, dim=1)

    def _image_stats(self, images: torch.Tensor) -> torch.Tensor:
        """Return per-sample luminance / contrast / saturation, broadcast to spatial."""
        B, _, H, W = images.shape
        lum = (0.27 * images[:, 0] + 0.67 * images[:, 1] + 0.06 * images[:, 2] + 1e-5)[:, None]
        luminance = lum.mean(dim=(1, 2, 3))
        contrast = lum.var(dim=(1, 2, 3))
        clipped = torch.clip(images, 0.0, 1.0)
        i_max = clipped.max(dim=1)[0]
        i_min = clipped.min(dim=1)[0]
        sat = (i_max - i_min) / (torch.minimum(i_max + i_min, 2.0 - i_max - i_min) + 1e-2)
        saturation = sat.mean(dim=(1, 2))
        stats = torch.stack([luminance, contrast, saturation], dim=1)
        return stats[:, :, None, None].expand(B, 3, H, W)

    def forward(self, state: PipelineState) -> torch.Tensor:
        img_ds = self.down_sample(state.image)
        state_ch = self._state_channels(state, hw=img_ds.shape[-1])
        stats_ch = self._image_stats(img_ds)
        obs = torch.cat([img_ds, state_ch, stats_ch], dim=1)
        feat = self.feature_extractor(obs)
        return self.fc2(self.lrelu(self.fc1(feat)))


__all__ = ["FeatureExtractor", "pdf_sample", "AdaptiveISPValueNet"]
