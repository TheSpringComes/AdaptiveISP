# 自动拟合 ISP ＋ 可选暗角补偿

**两个主文件，中文注释，只需 NumPy＋Pillow，完全使用 CPU。** 将本文件夹单独复制到其他位置也能运行，无需原 InfiniteISP 或 `isp_repro`。

- `fit.py`：自动读取样例、分组留出、拟合一套全局 ISP 参数，并单独估计暗角参数。
- `isp.py`：RAW 读取和推理；默认加载固定位置的拟合参数，**默认关闭暗角补偿**。
- `evaluate.py`：独立测试集的整图推理、误差统计和四张独立结果图片。
- `dataset/`：已复制原 `hikrobot_mini` 全部 57 组：**40 组训练、17 组测试**，保留原来的三种图像。
- `fitted/`：使用上述 40 组训练数据拟合的当前默认参数及报告。

## 直接运行随附数据

在本文件夹执行，无需重新提供数据或参数路径：

```bash
python -m pip install -r requirements.txt
python fit.py
python evaluate.py
```

已经附带拟合好的参数，可跳过 `fit.py`。拟合只使用 `dataset/train`，`dataset/test` 只用于评估，不用于选择参数。`evaluate.py` 一次运行同时生成暗角关闭和开启两种结果，无须执行两次。

每个 `outputs/test/样例名/` 保存：

- `1-InfiniteISP.png`：原 InfiniteISP 结果。
- `2-Ours.png`：Ours，关闭暗角矫正。
- `3-Ours-Vig.png`：Ours，开启暗角矫正。
- `4-Camera.png`：原相机参考图（GT）。
- `metrics_infinite.json`、`metrics_off.json`、`metrics_on.json`：三种结果相对于相机 GT 的完整图像指标。

四张 PNG 均为完整分辨率，不生成拼接图。InfiniteISP 和 Camera 直接复制原文件；报告文件名中的 `off` 表示关闭暗角矫正，`on` 表示开启。

整体报告为 `outputs/test/report_infinite.json`、`report_off.json`、`report_on.json`，指标始终使用完整分辨率。批量评价固定同时计算两种 Ours，`--vignette /path/to/vignette.json` 用于指定暗角参数，省略时使用默认文件。单图 `isp.py` 的暗角开关行为不变。

## 1. 数据格式

```text
dataset/
  任意样例名_A/
    raw.raw
    camera_rgb.png
    metadata.json          # 可选
    infinite_isp_rgb.png   # 拟合/单图推理不需要；批量评价需要
  任意样例名_B/
    raw.raw
    camera_rgb.png
```

样例名任意，无须数字编号。要求同一台相机、相同 RAW 尺寸/位深/Bayer 排列，相机 RGB 为同尺寸的 RGB8 图像。固定曝光/WB等设置有助于得到一套稳定参数；程序不会按每张图的目录名调整增益。

无 metadata 时，尺寸从 `camera_rgb.png` 读取，**位深默认 8、Bayer 默认 GBRG**，对应本次数据。其他格式必须传 `--bits 10/12`、`--bayer rggb/grbg/bggr` 等正确值。支持 unpacked uint8 或 little-endian uint16 容器；Packed/带 padding 数据不支持。

有 metadata 时优先使用原采集格式：

```json
{
  "raw_bit_depth": 8,
  "bayer_pattern": "gbrg",
  "scene_id": "scene_01",
  "raw": {"width": 5472, "height": 3648}
}
```

`scene_id` 可选。曝光、增益、相机 BlackLevel 节点不会被猜测或直接当作 RAW 码值使用。普通 `.raw` 字节流和这批 PNG 不内嵌拍摄记录；新增记录现已放在各样例的 `metadata.json` 中。57 份记录均确认 8 位 GBRG 和当前尺寸，无 `scene_id`，因此继续使用固定的场景分组文件。

完整的**单个样例**模板见 [metadata.example.json](metadata.example.json)，字段与曝光/ISO泛化说明见 [METADATA.md](METADATA.md)。示例值是假设值，没有写入任何真实样例。当前模型不以曝光/ISO为条件输入，补充这些字段不会自动切换参数。

## 2. 一条命令拟合

在本文件夹执行：

```bash
python -m pip install -r requirements.txt
python fit.py
# 使用另一套已分好的训练/测试数据：
python fit.py /path/to/train --test-data /path/to/test --output /path/to/config
# 使用尚未划分的数据，自动留出验证组：
python fit.py /path/to/dataset --output /path/to/config
```

输出：

```text
fitted/
  params.json       # 所有图统一使用：色调指数、CCM、偏置、去马赛克/细节参数
  vignette.json     # 独立暗角参数，所有图统一使用
  split.json        # 实际训练/测试（或自动验证）样例及场景分组
  report.json       # 参数搜索过程、区域指标、暗角估计的离散度与适用限制
  preview_*.jpg     # 相机参考 / 暗角关闭 / 暗角开启
```

输出目录必须位于数据目录之外；重复运行会更新同名参数和预览。`--previews 0` 跳过全分辨率推理预览，可加快纯参数拟合。

## 3. 暗角开关

关闭（默认）：

```bash
python isp.py "dataset/test/0001_白天中暗·室内/raw.raw" out_off.png
```

开启：

```bash
python isp.py "dataset/test/0001_白天中暗·室内/raw.raw" out_on.png --vignette
```

`isp.py` 顶部的 `DEFAULT_PARAMS_PATH` 和 `DEFAULT_VIGNETTE_PATH` 分别指定默认路径。目前指向本工具的 `fitted/`，不受启动时工作目录影响；可以改成任意固定绝对路径，例如 `Path(r'D:/ISP_CONFIG/params.json')` 和 `Path(r'D:/ISP_CONFIG/vignette.json')`，配置不必与代码同目录。

省略 `--vignette` 为关闭；只写 `--vignette` 使用默认暗角文件；`--vignette /path/to/vignette.json` 使用指定文件。`--params /path/to/params.json` 可覆盖默认色彩参数。推理只需要 RAW 和参数，不读取相机 RGB。`fit.py --output` 用于指定参数保存目录。

Python 调用：

```python
import json
from pathlib import Path
from isp import read_raw, render

p = json.loads(Path('fitted/params.json').read_text(encoding='utf-8'))
v = json.loads(Path('fitted/vignette.json').read_text(encoding='utf-8'))
raw, fmt, metadata = read_raw('dataset/任意样例名_A', p['format'])
rgb_off = render(raw, p)       # 关闭，返回完整分辨率 RGB uint8
rgb_on = render(raw, p, v)     # 开启
```

暗角补偿作用在 **RAW 线性域、去马赛克之前**，不是对最终 PNG 四角直接拉亮。分块运行使用整幅图像坐标计算补偿，避免每个小块各自出现一套暗角。

## 4. 拟合逻辑

基本 ISP：归一化 Bayer → Malvar/双线性混合 → 色调指数 → 3×3 CCM＋RGB 偏置 → 亮度细节控制 → RGB8。

每张图自动取五个固定位置的原分辨率区域。使用 IRLS 降低大误差像素的影响，先搜索色调指数，再搜索去马赛克比例和细节强度，并重新拟合对应 CCM。默认色调指数搜索范围为 0.35～1.8；自定义范围可修改 `fit_isp()` 中的候选值。

所有拟合、候选参数选择仅使用训练组。报告中的验证指标来自五处区域，不是整张图的指标；不做对齐，因此异步拍摄的运动和噪声也会计入误差。预览是实际全分辨率推理之后再缩小的图像。

它拟合的是这类相机的**输出外观**，不是恢复厂商全部内部 ISP；没有通用 AE/AWB、时域降噪、任意 3D LUT 或传感器物理标定。

## 5. 场景分组和复现

随附数据使用固定的 40/17 划分及 `dataset/groups.json`，程序检查同一场景组不跨训练/测试。自定义数据传入 `--test-data` 时也按目录固定，建议同时提供覆盖两边样例的 `--groups` 以检查场景隔离。

自定义数据不传 `--test-data` 时，根据 RAW 缩略图相关性自动合并近重复场景，再按固定随机种子留出约 20% 的场景组，避免相邻重复图简单随机拆分。

自动相似度无法保证识别全部相同场景。已知场景时，建议提供 `groups.json`（要覆盖全部样例）：

```json
{
  "任意样例名_A": "室内场景1",
  "任意样例名_B": "室内场景1",
  "任意样例名_C": "室外场景2"
}
```

```bash
python fit.py ../dataset --output fitted --groups groups.json --seed 42
```

优先级：`--groups` > 全部 metadata 中的 `scene_id` > 自动相似度分组。实际分组保存在 `split.json`，可复制其中的 `groups` 映射再人工修正。

只有一个场景组时默认报错；若明确只想用全部数据拟合，可传 `--validation-fraction 0`，此时没有独立验证指标。不会偷偷把同一组同时用于训练和验证。

## 6. 暗角参数的含义和限制

中心固定为图像中心，中心增益为 1；角落的归一化半径平方为 1：

```text
gain(r) = exp(a × r² + b × r⁴)
```

两个非负系数使补偿向边缘单调增大。三个颜色通道共用这一增益，不拟合彩色阴影。默认最大增益上限 3，可用 `--max-vignette-gain 2` 等调整。

无白板时，工具从训练 RAW 的绿色平面估计：对每张图拟合 log 亮度的常量、水平/竖直趋势和径向衰减，再对多图径向估计取中值。水平/竖直项仅用于减弱场景明暗梯度干扰，不会成为推理时的逐图参数。

**普通场景的真实明暗和镜头暗角不能唯一分离。** 因此 `ordinary_scene_estimate` 是有假设的补偿估计，不是已验证的真实镜头标定。角点估计的跨图离散度写入报告，不当作严格置信区间。无法估计时保留单位增益并记录 `not_fitted`，基本 ISP 仍可使用。

补偿会同时放大边缘噪声，也可能使边缘亮部饱和；已接近全黑的遮挡角落没有足够信号，乘增益不能恢复其中的细节。图像中心假设不适合明显偏心镜头；更换镜头、光圈、ROI 或相机内部处理后应重新标定。

相机参考本身也带暗角，因此主动补偿后与相机参考的 MAE 可能变大。这个 MAE **不是暗角消除质量的真值指标**，不能用它宣称画面变坏，也不能反过来证明恢复了正确照度。

### 有均匀白板 RAW 时

```text
flatfields/
  flat1.raw
  flat2.raw
```

也支持 `flatfields/样例名/raw.raw`，格式应与标定数据相同：

```bash
python fit.py ../dataset --output fitted --flatfields ../flatfields
```

这时暗角参数独立由白板估计；色彩参数仍由普通 RAW/RGB 配对数据拟合。白板应均匀照明且不过曝，**不是黑帧**。若提供的白板无有效信息会明确报错。

## 7. 当前结果与验证

- 当前划分：40 张训练、17 张独立测试；默认参数保存在 `fitted/`。
- 基本 ISP：色调指数 1.0、Malvar 权重 1.0、细节强度 -0.5；所有图片共用一套 CCM 和偏置。
- 测试区域 MAE：暗角关闭 **3.5254/255**、PSNR **25.17 dB**。这是五处区域指标，完整图像指标见 `outputs/test/report_off.json`。
- 暗角估计：`a=0`、`b≈0.619467`，角点约 **1.8579×**；默认关闭，未经白板验证。
- 完整测试结果和运行记录见 [RUN_RESULTS.md](RUN_RESULTS.md)。相机与 RAW 是短间隔异步帧，指标包含运动及拍摄差异。
- 10 项自动测试通过，包括改变独立测试集的参考图后，拟合参数和暗角参数保持完全不变。
