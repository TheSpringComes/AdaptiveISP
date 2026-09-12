import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image

PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_PARAMS_PATH = PROJECT_DIR / 'fitted' / 'params.json'
DEFAULT_VIGNETTE_PATH = PROJECT_DIR / 'fitted' / 'vignette.json'


def read_raw(path, defaults=None):
    path = Path(path)
    if path.is_dir():
        path = path / 'raw.raw'
    sidecar = path.with_name('metadata.json')
    meta = json.loads(sidecar.read_text(encoding='utf-8-sig')) if sidecar.exists() else {}
    frame, fmt = meta.get('raw', {}), dict(defaults or {})
    recorded = dict(width=frame.get('width', meta.get('width')),
                    height=frame.get('height', meta.get('height')),
                    bit_depth=meta.get('raw_bit_depth', meta.get('bit_depth')),
                    bayer_pattern=meta.get('bayer_pattern'))
    fmt.update({k: v for k, v in recorded.items() if v is not None})
    if any(fmt.get(k) is None for k in recorded):
        raise ValueError(f'{path}: 缺少尺寸/位深/Bayer 排列')
    w, h, bits = (int(fmt[k]) for k in ('width', 'height', 'bit_depth'))
    pattern = fmt['bayer_pattern'].lower()
    if min(w, h) < 16 or w % 2 or h % 2 or bits not in (8, 10, 12):
        raise ValueError('只支持偶数宽高（至少16）及 unpacked 8/10/12 位 RAW')
    if pattern not in ('rggb', 'bggr', 'grbg', 'gbrg'):
        raise ValueError('不支持的 Bayer 排列')
    # PFNC 格式已记录时，拒绝 Packed 或与声明排列冲突的数据。
    formats = {(0x01080008 if b == 8 else 0x0110000c if b == 10 else 0x01100010)+i: (b, p)
               for b in (8, 10, 12) for i, p in enumerate(('grbg', 'rggb', 'gbrg', 'bggr'))}
    if frame.get('pixel_format') is not None and formats.get(frame['pixel_format']) != (bits, pattern):
        raise ValueError('PFNC 格式不支持，或与位深/Bayer 信息冲突')
    dtype = np.dtype('u1' if bits == 8 else '<u2')
    if path.stat().st_size != w*h*dtype.itemsize:
        raise ValueError(f'{path}: RAW 字节数不匹配（不支持 padding/Packed）')
    raw = np.fromfile(path, dtype=dtype).reshape(h, w)
    if raw.max() >= 2**bits:
        raise ValueError('码值超出声明位深')
    return raw, dict(width=w, height=h, bit_depth=bits, bayer_pattern=pattern), meta


def convolve(src, kernel):
    """小核对称边界卷积；本文件使用的核均中心对称。"""
    h, w = src.shape
    padded = np.pad(src, len(kernel)//2, mode='symmetric')
    out = np.zeros_like(src)
    for i, j in np.ndindex(kernel.shape):
        if kernel[i, j]:
            out += padded[i:i+h, j:j+w] * kernel[i, j]
    return out


def demosaic(src, pattern='gbrg', mix=1.0, clip=True):
    """输入归一化 Bayer；mix=1 为 Malvar，mix=0 为双线性。保留浮点精度。"""
    gk = np.float32([[0,0,-1,0,0],[0,0,2,0,0],[-1,2,4,2,-1],[0,0,2,0,0],[0,0,-1,0,0]])/8
    hk = np.float32([[0,0,.5,0,0],[0,-1,0,-1,0],[-1,4,5,4,-1],[0,-1,0,-1,0],[0,0,.5,0,0]])/8
    dk = np.float32([[0,0,-1.5,0,0],[0,2,0,2,0],[-1.5,0,6,0,-1.5],[0,2,0,2,0],[0,0,-1.5,0,0]])/8
    g, h, v, d = [convolve(src, k) for k in (gk, hk, hk.T, dk)]
    r, b = src.copy(), src.copy()
    for c, ch in [('r', r), ('b', b)]:
        y, x = divmod(pattern.index(c), 2)
        ch[y::2, 1-x::2] = h[y::2, 1-x::2]
        ch[1-y::2, x::2] = v[1-y::2, x::2]
        ch[1-y::2, 1-x::2] = d[1-y::2, 1-x::2]
    for y, x in np.ndindex((2, 2)):
        if pattern[2*y+x] == 'g':
            g[y::2, x::2] = src[y::2, x::2]
    malvar = np.stack([r, g, b], -1)
    if mix == 1:
        return np.clip(malvar, 0, 1) if clip else malvar
    linear = []
    for color in 'rgb':
        sparse = np.zeros_like(src)
        for y, x in np.ndindex((2, 2)):
            if pattern[2*y+x] == color:
                sparse[y::2, x::2] = src[y::2, x::2]
        kernel = (np.float32([[0,1,0],[1,4,1],[0,1,0]]) if color == 'g' else
                  np.float32([[1,2,1],[2,4,2],[1,2,1]]))/4
        linear.append(convolve(sparse, kernel))
    rgb = mix*malvar+(1-mix)*np.stack(linear, -1)
    return np.clip(rgb, 0, 1) if clip else rgb


def color(rgb, params):
    """所有图统一的色调指数、颜色矩阵和偏置；不进行逐图自动曝光/WB。"""
    return np.maximum(rgb, 0)**params['exponent'] @ np.float32(params['ccm']).T + np.float32(params['offset'])


def detail(rgb, strength):
    """固定 3×3 Gaussian 的亮度细节控制；正值锐化，负值平滑。"""
    luma = rgb @ np.float32([.299, .587, .114])
    blur = convolve(luma, np.float32([[1,2,1],[2,4,2],[1,2,1]])/16)
    return rgb + strength*(luma-blur)[..., None]


def radius2(shape, origin=(0, 0), full_shape=None):
    """以整幅图像中心为光学中心，r²=0 在中心，r²=1 在角落。"""
    fh, fw = full_shape or shape
    cy, cx = (fh-1)/2, (fw-1)/2
    y, x = np.ogrid[origin[0]:origin[0]+shape[0], origin[1]:origin[1]+shape[1]]
    return ((x-cx)**2+(y-cy)**2)/(cx*cx+cy*cy)


def vignette_gain(r2, params):
    """独立暗角参数：gain=exp(a*r²+b*r⁴)，中心固定为1，并限制放大倍数。"""
    a, b = params['coefficients']
    cap = params['max_gain']
    if not (1 <= cap <= 8) or not np.all(np.isfinite([a, b])):
        raise ValueError('无效的暗角参数')
    return np.exp(np.clip(a*r2+b*r2*r2, -np.log(cap), np.log(cap))).astype(np.float32)


def render(raw, params, vignette=None, tile=512):
    if tile < 2 or tile % 2:
        raise ValueError('tile 必须是正偶数')
    h, w = raw.shape
    if (h,w) != (params['format']['height'],params['format']['width']):
        raise ValueError('输入尺寸与参数文件不一致；ROI 改变后需要重新标定')
    bits, pattern = params['format']['bit_depth'], params['format']['bayer_pattern']
    out = np.empty((h, w, 3), np.uint8)
    for y in range(0, h, tile):
        for x in range(0, w, tile):
            y0, x0 = max(y-8, 0), max(x-8, 0)  # 偶数起点保持 Bayer 相位。
            src = raw[y0:min(y+tile+8, h), x0:min(x+tile+8, w)].astype(np.float32)/(2**bits-1)
            if vignette is not None:
                src *= vignette_gain(radius2(src.shape, (y0, x0), (h, w)), vignette)
            rgb = detail(color(demosaic(src, pattern, params['malvar_weight']), params), params['detail_strength'])
            hh, ww = min(tile, h-y), min(tile, w-x)
            rgb = rgb[y-y0:y-y0+hh, x-x0:x-x0+ww]
            out[y:y+hh, x:x+ww] = np.rint(np.clip(rgb, 0, 1)*255).astype(np.uint8)
    return out


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('raw');
    parser.add_argument('output', type=Path)
    parser.add_argument('--params', type=Path, default=DEFAULT_PARAMS_PATH,
                        help=f'ISP 参数，默认 {DEFAULT_PARAMS_PATH}')
    # 不写选项→关闭；只写 --vignette→默认文件；带路径→使用指定文件。
    parser.add_argument('--vignette', nargs='?', type=Path, const=DEFAULT_VIGNETTE_PATH, default=None,
                        help=f'开启暗角；省略路径时使用 {DEFAULT_VIGNETTE_PATH}')
    return parser.parse_args(argv)


def main():
    args = parse_args()
    params = json.loads(args.params.read_text(encoding='utf-8'))
    vig = json.loads(args.vignette.read_text(encoding='utf-8')) if args.vignette else None
    raw, fmt, _ = read_raw(args.raw, params['format'])
    if fmt != params['format']:
        raise ValueError('输入格式与拟合数据不同，请重新拟合或检查 metadata')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(render(raw, params, vig)).save(args.output)
    print(args.output)


if __name__ == '__main__':
    main()
