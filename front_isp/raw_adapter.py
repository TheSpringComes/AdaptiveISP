"""Input adapter: raw / mosaic inputs -> canonical linear RGB.

数据入网前的统一适配层（Dataset / Input Adapter 层），完整链路：

    Dataset/Input Adapter → 数据格式解析 → Bayer reconstruction（如需要）
        → demosaic → canonical linear RGB → ISP

ISP 对外接口不变：ISP 始终只接收标准 3 通道 RGB (B, 3, H, W)，
不感知 Bayer、packed RAW 或 demosaic 过程。

支持的输入格式（`to_canonical_rgb`）：
    (3, H, W)   — 已是 3 通道 RGB：直接 passthrough，不执行 demosaic。
    (1, H, W)   — Bayer mosaic：按 pattern demosaic。
    (4, h, w)   — Bayer packed 4-plane（R, G(code1), G(code3), B，
                  每平面为 mosaic 的 (2h, 2w) 半分辨率子采样）：先按
                  pattern 重建 full-res mosaic，再 demosaic，输出
                  (3, 2h, 2w)。

CFA pattern 从数据 / 配置获取（per-file metadata 或 dataset config），
支持常见 Bayer 排列 RGGB / BGGR / GRBG / GBRG，不硬编码单一排列。
pattern 同时接受名称字符串或 2x2 颜色码网格（libraw 码：
0=R, 1=G, 2=B, 3=第二绿）。

demosaic 默认方法：
    RGB = 0.5 * Malvar(2004) + 0.5 * Bilinear

Malvar 5x5 核与 colour-demosaicing（BSD-3-Clause,
Colour Developers）的参考实现逐系数一致；在 tools/verify_raw_adapter.py
中与其输出做数值对拍。实现基于 torch conv2d（CPU 即可，可批处理）。

Demosaic 输出统一为 (3, H, W) float32、值域 [0, 1] 的 linear RGB；
target 不再因 Bayer packing 被下采样 2x，与输入保持全分辨率对齐
（显式尺寸对齐由调用方/Dataset 做 center-crop）。
"""
from __future__ import annotations

from typing import Optional, Union

import numpy as np
import torch
import torch.nn.functional as F

__all__ = [
    "BAYER_PATTERNS",
    "parse_pattern",
    "reconstruct_mosaic",
    "demosaic_bayer",
    "to_canonical_rgb",
]

# 名称 -> libraw 颜色码网格 (2, 2)。码：0=R, 1=G, 2=B, 3=第二绿。
# 与 tools/fivek_cfa_scan.py 的实测扫描映射一致（原始 DNG raw_pattern）。
BAYER_PATTERNS: dict[str, np.ndarray] = {
    "RGGB": np.array([[0, 1], [3, 2]], dtype=np.int8),
    "BGGR": np.array([[2, 3], [1, 0]], dtype=np.int8),
    "GRBG": np.array([[1, 0], [2, 3]], dtype=np.int8),
    "GBRG": np.array([[3, 2], [0, 1]], dtype=np.int8),
}

# packed 4-plane 的通道约定（tools/fivek_build_cache.py 的打包顺序）：
# ch0=R, ch1=G(code1), ch2=G(code3), ch3=B。颜色码 -> plane 下标。
_CODE_TO_PLANE = {0: 0, 1: 1, 3: 2, 2: 3}

# ---------------------------------------------------------------------------
# Malvar-He-Cutler (2004) 5x5 核（与 colour-demosaicing 参考实现一致）。
# 引用: Malvar, He, Cutler, "High-Quality Linear Interpolation for
# Demosaicing of Bayer-Patterned Color Images", ICASSP 2004.
# ---------------------------------------------------------------------------
_K_G = torch.tensor([  # G at R/B sites
    [0, 0, -1.0, 0, 0],
    [0, 0, 2.0, 0, 0],
    [-1.0, 2.0, 4.0, 2.0, -1.0],
    [0, 0, 2.0, 0, 0],
    [0, 0, -1.0, 0, 0],
]) / 8.0
_K_RB = torch.tensor([  # R at G-in-R-row (R 水平相邻) / B at G-in-B-row；转置用于另一方向
    [0, 0, 0.5, 0, 0],
    [0, -1.0, 0, -1.0, 0],
    [-1.0, 4.0, 5.0, 4.0, -1.0],
    [0, -1.0, 0, -1.0, 0],
    [0, 0, 0.5, 0, 0],
]) / 8.0
_K_RB_T = _K_RB.t().contiguous()
_K_X = torch.tensor([  # R at B sites / B at R sites（对角重建）
    [0, 0, -1.5, 0, 0],
    [0, 2.0, 0, 2.0, 0],
    [-1.5, 0, 6.0, 0, -1.5],
    [0, 2.0, 0, 2.0, 0],
    [0, 0, -1.5, 0, 0],
]) / 8.0

# Bilinear 3x3 核
_K_BIL_G = torch.tensor([  # G at R/B: 4 邻域均值
    [0, 1.0, 0],
    [1.0, 0, 1.0],
    [0, 1.0, 0],
]) / 4.0
_K_BIL_H = torch.tensor([  # 水平相邻对 (R 或 B at G)：左右均值
    [0, 0, 0],
    [0.5, 0, 0.5],
    [0, 0, 0],
])
_K_BIL_V = _K_BIL_H.t().contiguous()
_K_BIL_D = torch.tensor([  # 对角 (R at B / B at R)：4 对角均值
    [0.25, 0, 0.25],
    [0, 0, 0],
    [0.25, 0, 0.25],
])


def parse_pattern(pattern: Union[str, np.ndarray, list]) -> np.ndarray:
    """解析 CFA pattern 为 (2, 2) libraw 颜色码网格。

    接受 "RGGB"/"BGGR"/"GRBG"/"GBRG"（大小写不敏感）或直接的 2x2
    颜色码数组（0=R, 1=G, 2=B, 3=第二绿）。不硬编码默认排列——
    pattern 必须由调用方从数据 metadata 或 dataset config 提供。
    """
    if isinstance(pattern, str):
        key = pattern.strip().upper()
        if key not in BAYER_PATTERNS:
            raise ValueError(
                f"unknown Bayer pattern {pattern!r}; expected one of "
                f"{sorted(BAYER_PATTERNS)} or a 2x2 code grid")
        return BAYER_PATTERNS[key]
    grid = np.asarray(pattern)
    if grid.shape != (2, 2):
        raise ValueError(f"pattern grid must be (2, 2), got {grid.shape}")
    # 合法 Bayer 排列：0/1/2/3（R/G/B/第二绿）各恰好出现一次
    if sorted(np.unique(grid).tolist()) != [0, 1, 2, 3]:
        raise ValueError(f"pattern grid must contain each of 0/1/2/3 once: {grid}")
    return grid.astype(np.int8)


def reconstruct_mosaic(raw4: np.ndarray, pattern: Union[str, np.ndarray]) -> np.ndarray:
    """Bayer packed 4-plane -> full-resolution Bayer mosaic。

    输入 (4, h, w)：ch0=R, ch1=G(code1), ch2=G(code3), ch3=B（每平面为
    mosaic 在对应 CFA 位置的 (h, w) 子采样，见 tools/fivek_build_cache.py
    的打包逻辑）。按 pattern 的颜色码网格把各平面放回 (2h, 2w) 的
    对应 (i::2, j::2) 位置，恢复原始 mosaic。输出 float32。
    """
    grid = parse_pattern(pattern)
    raw4 = np.asarray(raw4, dtype=np.float32)
    if raw4.ndim != 3 or raw4.shape[0] != 4:
        raise ValueError(f"expected packed raw (4, h, w), got {raw4.shape}")
    h, w = raw4.shape[1], raw4.shape[2]
    mosaic = np.empty((2 * h, 2 * w), dtype=np.float32)
    for (i, j), code in np.ndenumerate(grid):
        mosaic[i::2, j::2] = raw4[_CODE_TO_PLANE[int(code)]]
    return mosaic


def _conv(img: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """5x5/3x3 二维卷积，symmetric 边界（与 scipy.ndimage.convolve 默认
    'reflect' 模式逐像素一致；torch 的 F.pad 没有等价的 symmetric 模式，
    故先 numpy pad 再零填充卷积）。(H, W) -> (H, W)。"""
    k = kernel.to(img.dtype)
    r = k.shape[0] // 2
    x = torch.from_numpy(np.pad(img.numpy(), r, mode="symmetric"))[None, None]
    return F.conv2d(x, k[None, None])[0, 0]


def _conv_multi(items: list[tuple[torch.Tensor, torch.Tensor]]) -> list[torch.Tensor]:
    """批量的 (image, kernel) 二维卷积（symmetric 边界，同 _conv）。

    同尺寸核合成尽量少的 F.conv2d：所有输入相同（Malvar：完整 CFA）
    时用单输入多输出核；输入不同（Bilinear：掩码后单通道）时用
    grouped conv（每组一个输入通道一个核），消除逐次 pad/conv 开销。
    """
    by_r: dict[int, list] = {}
    for img, k in items:
        by_r.setdefault(k.shape[0] // 2, []).append((img, k))
    out: list[torch.Tensor] = []
    for r, group in by_r.items():
        if all(img is group[0][0] for img, _ in group):
            # 同输入多核：weight (K, 1, k, k)，输出 (K, H, W)
            x = torch.from_numpy(
                np.pad(group[0][0].numpy(), r, mode="symmetric"))[None, None]
            w = torch.stack([k.to(x.dtype) for _, k in group])[:, None]
            ys = F.conv2d(x, w)[0]
        else:
            # 异输入：input (1, K, H, W) + weight (K, 1, k, k) + groups=K
            xs = torch.stack(
                [torch.from_numpy(np.pad(img.numpy(), r, mode="symmetric"))
                 for img, _ in group])[None]
            ws = torch.stack([k.to(xs.dtype) for _, k in group])[:, None]
            ys = F.conv2d(xs, ws, groups=len(group))[0]
        out.extend(ys.unbind(0))
    return out


def demosaic_bayer(
    mosaic: Union[np.ndarray, torch.Tensor],
    pattern: Union[str, np.ndarray],
    method: str = "malvar_bilinear",
) -> np.ndarray:
    """Bayer mosaic (H, W) -> linear RGB (3, H, W) float32。

    method:
        "malvar"          — 纯 Malvar-He-Cutler (2004)
        "bilinear"        — 纯双线性
        "malvar_bilinear" — 默认，RGB = 0.5*Malvar + 0.5*Bilinear

    返回未 clamp 的 linear RGB（由 `to_canonical_rgb` 统一 clamp 到 [0,1]）。
    """
    if method not in ("malvar", "bilinear", "malvar_bilinear"):
        raise ValueError(f"unknown demosaic method {method!r}")
    grid = parse_pattern(pattern)
    m = torch.as_tensor(np.asarray(mosaic), dtype=torch.float32)
    if m.ndim == 3 and m.shape[0] == 1:
        m = m[0]
    if m.ndim != 2:
        raise ValueError(f"expected mosaic (H, W), got {tuple(m.shape)}")
    if m.shape[0] % 2 or m.shape[1] % 2:
        raise ValueError(f"mosaic dims must be even, got {tuple(m.shape)}")

    H, W = m.shape
    # 颜色码掩码 (H, W)：code_map[i, j] = grid[i%2, j%2]
    codes = torch.from_numpy(np.tile(grid, (H // 2, W // 2)).astype(np.int64))
    r_m, b_m = codes == 0, codes == 2
    g_m = (codes == 1) | (codes == 3)   # 两个绿位都视为 G
    # R 行/列、B 行/列（colour-demosaicing 的应用逻辑）
    r_row = r_m.any(dim=1)[:, None].expand(H, W)
    b_col = b_m.any(dim=0)[None, :].expand(H, W)
    b_row = b_m.any(dim=1)[:, None].expand(H, W)
    r_col = r_m.any(dim=0)[None, :].expand(H, W)

    r_c, g_c, b_c = m * r_m, m * g_m, m * b_m

    if method == "malvar":
        # Malvar 核直接作用于完整 CFA（核内含 R/B/G 位置权重）。
        # 4 个核一次批量卷积：[K_G, K_RB, K_RB_T, K_X]。
        (c_g, c_rb, c_rbt, c_x) = _conv_multi(
            [(m, _K_G), (m, _K_RB), (m, _K_RB_T), (m, _K_X)])
        g = torch.where(r_m | b_m, c_g, g_c)
        r = torch.where(r_row & b_col, c_rb,               # G in R row
                        torch.where(b_row & r_col, c_rbt,  # G in B row
                                    torch.where(b_row & b_col, c_x, r_c)))
        b = torch.where(b_row & r_col, c_rb,               # G in B row（B 水平相邻）
                        torch.where(r_row & b_col, c_rbt,   # G in R row（B 垂直相邻）
                                    torch.where(r_row & r_col, c_x, b_c)))
        rgb = torch.stack([r, g, b], dim=0)
    else:
        # Bilinear 核作用于掩码后的单通道（核即邻域平均）。
        # grouped conv 一次完成 7 个 (输入, 核) 对。
        (g_bil, r_h, r_v, r_d, b_h, b_v, b_d) = _conv_multi(
            [(g_c, _K_BIL_G),
             (r_c, _K_BIL_H), (r_c, _K_BIL_V), (r_c, _K_BIL_D),
             (b_c, _K_BIL_H), (b_c, _K_BIL_V), (b_c, _K_BIL_D)])
        g = torch.where(r_m | b_m, g_bil, g_c)
        r = torch.where(r_row & b_col, r_h,
                        torch.where(b_row & r_col, r_v,
                                    torch.where(b_row & b_col, r_d, r_c)))
        b = torch.where(b_row & r_col, b_h,
                        torch.where(r_row & b_col, b_v,
                                    torch.where(r_row & r_col, b_d, b_c)))
        rgb = torch.stack([r, g, b], dim=0)

    if method == "malvar_bilinear":
        # RGB = 0.5*Malvar + 0.5*Bilinear（默认混合）。
        # 注意上面的 if/else 在 malvar_bilinear 时走的是 bilinear 分支，
        # 结果直接复用，避免重复计算。
        rgb_bl = rgb
        rgb_mv = demosaic_bayer(m.numpy(), pattern, method="malvar")
        rgb = 0.5 * torch.from_numpy(rgb_mv) + 0.5 * rgb_bl
    return rgb.numpy()


def to_canonical_rgb(
    x: Union[np.ndarray, torch.Tensor],
    pattern: Optional[Union[str, np.ndarray]] = None,
    method: str = "malvar_bilinear",
) -> np.ndarray:
    """任意支持格式 -> canonical linear RGB (3, H, W) float32。

    (3, H, W)  -> 直接 passthrough（float32 化，不 demosaic）。
    (1, H, W)  -> Bayer mosaic，需要 pattern，demosaic。
    (4, h, w)  -> Bayer packed 4-plane，需要 pattern，先 reconstruct_mosaic
                  再 demosaic，输出 (3, 2h, 2w)。

    返回值统一 float32、[0, 1]（clamp），linear RGB（无 WB/CCM/gamma）。
    """
    arr = np.asarray(x)
    if arr.ndim != 3:
        raise ValueError(f"expected (C, H, W), got {arr.shape}")
    c = arr.shape[0]
    if c == 3:
        out = arr.astype(np.float32)
    elif c in (1, 4):
        if pattern is None:
            raise ValueError(
                "Bayer/mosaic input requires a CFA pattern from data "
                "metadata or dataset config (not hardcoded)")
        if c == 4:
            mosaic = reconstruct_mosaic(arr, pattern)
        else:
            mosaic = arr[0].astype(np.float32)
        out = demosaic_bayer(mosaic, pattern, method=method)
    else:
        raise ValueError(f"unsupported channel count {c} for raw adapter")
    return np.clip(out, 0.0, 1.0)
