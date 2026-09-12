"""一条命令：读取数据 → 分组留出 → 拟合统一 ISP → 单独估计暗角 → 保存参数/报告。

输入每个样例目录包含 raw.raw 和 camera_rgb.png；不需要 InfiniteISP 输出。
所有拟合仅使用训练组；普通场景的暗角估计依赖场景统计假设，建议白板复标。
"""
import argparse
import json
import os
from pathlib import Path
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')  # 小矩阵拟合避免过多线程开销。
import numpy as np
from PIL import Image, ImageDraw
from isp import read_raw, demosaic, color, detail, render, radius2, vignette_gain, PROJECT_DIR, DEFAULT_PARAMS_PATH


def save_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def area_rgb(raw, fmt, width=192):
    """先拆 Bayer 平面，再面积平均；不能直接缩放 Bayer 马赛克。"""
    src, pattern = raw.astype(np.float32)/(2**fmt['bit_depth']-1), fmt['bayer_pattern']
    planes = {c: [] for c in 'rgb'}
    for y, x in np.ndindex((2, 2)):
        planes[pattern[y*2+x]].append(src[y::2, x::2])
    size = (min(width, raw.shape[1]//2), max(8, round(min(width, raw.shape[1]//2)*raw.shape[0]/raw.shape[1])))
    return np.stack([np.asarray(Image.fromarray(np.mean(planes[c], axis=0)).resize(size, Image.Resampling.BOX))
                     for c in 'rgb'], -1)


def load_data(dataset, bits, pattern):
    """逐张读取，内存只保留小预览和五处固定区域；不缓存全部原分辨率数据。"""
    paths = sorted(dataset.glob('*/raw.raw'))
    if not paths:
        raise ValueError('没有找到 dataset/样例名/raw.raw')
    samples, expected = [], None
    for index, path in enumerate(paths):
        with Image.open(path.with_name('camera_rgb.png')) as im:
            if im.mode != 'RGB':
                raise ValueError(f'{path.parent}: camera_rgb.png 必须是 RGB8 图像')
            defaults = dict(width=im.width, height=im.height, bit_depth=bits, bayer_pattern=pattern)
            target = np.asarray(im).copy()
        raw, fmt, meta = read_raw(path, defaults)
        if raw.shape != target.shape[:2] or (expected is not None and fmt != expected):
            raise ValueError('同一套参数要求全部样例具有相同尺寸、位深和 Bayer 排列')
        expected = fmt
        h, w = raw.shape
        patch = min(96, h-16, w-16)//2*2
        if patch < 16:
            raise ValueError('拟合图像至少为 32×32')
        records = []
        for fy, fx in ((.15,.15), (.15,.85), (.5,.5), (.85,.15), (.85,.85)):
            y = min(max(int(h*fy-patch/2)//2*2, 0), h-patch)
            x = min(max(int(w*fx-patch/2)//2*2, 0), w-patch)
            y0, x0 = max(y-8, 0), max(x-8, 0)
            src = raw[y0:min(y+patch+8,h), x0:min(x+patch+8,w)].astype(np.float32)/(2**fmt['bit_depth']-1)
            trim = (slice(y-y0, y-y0+patch), slice(x-x0, x-x0+patch))
            records.append(dict(raw=src, origin=(y0,x0), trim=trim,
                m=demosaic(src, fmt['bayer_pattern'], 1, clip=False)[trim],
                b=demosaic(src, fmt['bayer_pattern'], 0, clip=False)[trim],
                y=target[y:y+patch,x:x+patch].astype(np.float32)/255))
        samples.append(dict(name=path.parent.name, path=path, records=records, proxy=area_rgb(raw,fmt),
                            scene_id=meta.get('scene_id'), has_metadata=bool(meta)))
        print(f'读取 {index+1}/{len(paths)}: {path.parent.name}', flush=True)
    return samples, expected


def split_data(samples, groups_file, fraction, seed):
    """优先使用用户分组；否则按 RAW 缩略图的相关性合并近重复场景。"""
    n = len(samples)
    parent = list(range(n))
    def find(i):
        while parent[i] != i:
            i = parent[i]
        return i
    mapping = json.loads(groups_file.read_text(encoding='utf-8-sig')) if groups_file else None
    if mapping is not None:
        if set(mapping) != {s['name'] for s in samples} or not all(isinstance(v,str) for v in mapping.values()):
            raise ValueError('groups.json 必须给每个样例名映射一个字符串组名，不能遗漏或多写')
        labels = [mapping[s['name']] for s in samples]
        method = 'explicit_groups'
    elif all(s['scene_id'] is not None for s in samples):
        labels = [str(s['scene_id']) for s in samples]
        method = 'metadata_scene_id'
    else:
        descriptors = []
        for sample in samples:
            im = np.rint(np.clip(sample['proxy'],0,1)*255).astype(np.uint8)
            desc = np.asarray(Image.fromarray(im).resize((24,16))).astype(float)
            desc -= desc.mean((0,1), keepdims=True)
            descriptors.append(desc.ravel()/max(np.linalg.norm(desc), 1e-9))
        correlation = np.array(descriptors) @ np.array(descriptors).T
        for i in range(n):
            for j in range(i):
                if correlation[i,j] >= .94:
                    parent[find(i)] = find(j)
        labels = [str(find(i)) for i in range(n)]
        method = 'automatic_RAW_similarity_0.94'
    groups = sorted(set(labels))
    if not 0 <= fraction < 1:
        raise ValueError('validation-fraction 必须在 [0,1)')
    if fraction and len(groups) < 2:
        raise ValueError('只有一个场景组，无法留出。增加独立场景，或明确传 --validation-fraction 0')
    shuffled = np.random.default_rng(seed).permutation(groups)
    held = set(shuffled[:min(len(groups)-1, max(1, round(len(groups)*fraction)))]) if fraction else set()
    train = np.array([label not in held for label in labels])
    return train, dict(method=method, seed=seed, groups=dict(zip([s['name'] for s in samples], labels)),
                       train=[s['name'] for s,t in zip(samples,train) if t],
                       validation=[s['name'] for s,t in zip(samples,train) if not t])


def observations(samples, train, mix, seed):
    """等量抽取每张训练图的像素，防止某张图/某个分辨率主导拟合。"""
    rng = np.random.default_rng(seed)
    xs, ys = [], []
    per_image = max(32, 60000//int(train.sum()))
    for sample, use in zip(samples, train):
        if not use:
            continue
        x = np.concatenate([np.clip(mix*r['m']+(1-mix)*r['b'],0,1).reshape(-1,3) for r in sample['records']])
        y = np.concatenate([r['y'].reshape(-1,3) for r in sample['records']])
        ids = rng.choice(len(x), min(len(x),per_image), replace=False)
        xs.append(x[ids]); ys.append(y[ids])
    return np.concatenate(xs), np.concatenate(ys)


def fit_color(x, y, exponent):
    """IRLS：反复降低大误差像素的权重，减少异步运动/噪声影响。"""
    design = np.column_stack([x.astype(float)**exponent, np.ones(len(x))])
    coef = np.vstack([np.eye(3), np.zeros(3)])
    for channel in range(3):
        valid = (y[:,channel] > 2/255) & (y[:,channel] < 253/255)
        if valid.sum() < 32:
            raise ValueError('有效非黑/非饱和像素不足，无法标定颜色')
        prior = coef[:,channel].copy()
        for _ in range(8):
            residual = design @ coef[:,channel] - y[:,channel]
            weight = np.minimum(1, .015/np.maximum(abs(residual),1e-8))*valid
            # 弱正则化朝单位矩阵收缩，避免色彩分布窄时解发散。
            coef[:,channel] = np.linalg.solve(design.T@(weight[:,None]*design)+np.eye(4)*.02,
                                              design.T@(weight*y[:,channel])+.02*prior)
    return dict(exponent=float(exponent), ccm=coef[:3].T.tolist(), offset=coef[3].tolist(),
                malvar_weight=1.0, detail_strength=0.0)


def errors(pred, target):
    delta = (pred.astype(float)-target)*255
    return np.array([abs(delta).sum(), (delta*delta).sum(), np.minimum(abs(delta),10.2).sum(), delta.size])


def describe(sums):
    if sums[3] == 0:
        return None
    mae, mse, robust = sums[:3]/sums[3]
    return dict(mae=float(mae), rmse=float(np.sqrt(mse)), psnr_db=float(10*np.log10(255**2/mse)) if mse else None,
                robust_mae=float(robust), channel_values=int(sums[3]))


def patch_prediction(record, params, vig=None):
    if vig is None:
        mix = params['malvar_weight']
        rgb = np.clip(mix*record['m']+(1-mix)*record['b'],0,1)
    else:
        fmt = params['format']
        gain = vignette_gain(radius2(record['raw'].shape, record['origin'], (fmt['height'],fmt['width'])), vig)
        rgb = demosaic(record['raw']*gain, fmt['bayer_pattern'], params['malvar_weight'])[record['trim']]
    # 去掉区域边缘两像素，避免独立区域的细节滤波边界影响指标。
    return np.rint(np.clip(detail(color(rgb,params),params['detail_strength']),0,1)*255)[2:-2,2:-2]/255


def fit_isp(samples, train, fmt, seed):
    x, y = observations(samples, train, 1, seed)
    candidates = []
    for exponent in np.unique(np.r_[np.linspace(.35,1.8,16), 1, 1/2.2]):
        params = fit_color(x,y,exponent)
        score = float(np.minimum(abs(np.clip(color(x,params),0,1)-y),.04).mean())
        candidates.append((score,params))
    best = min(candidates,key=lambda v:v[0])[1]['exponent']
    for exponent in np.linspace(max(.35,best-.08),min(1.8,best+.08),9):
        params = fit_color(x,y,exponent)
        score = float(np.minimum(abs(np.clip(color(x,params),0,1)-y),.04).mean())
        candidates.append((score,params))
    chosen = min(candidates,key=lambda v:v[0])[1]
    print(f"色调指数：{chosen['exponent']:.4f}；继续拟合去马赛克/细节参数",flush=True)
    detail_candidates = []
    for mix in (0,.5,1):
        x, y = observations(samples, train, mix, seed)
        base = fit_color(x,y,chosen['exponent'])
        for strength in (-.5,0,.4,.8):
            params = dict(base, malvar_weight=mix, detail_strength=strength, format=fmt)
            total = np.zeros(4)
            for sample, use in zip(samples,train):
                if use:
                    for record in sample['records']:
                        total += errors(patch_prediction(record,params), record['y'][2:-2,2:-2])
            detail_candidates.append((describe(total),params))
    chosen = min(detail_candidates,key=lambda v:v[0]['robust_mae'])[1]
    x, _ = observations(samples,train,chosen['malvar_weight'],seed)
    design = np.column_stack([x.astype(float)**chosen['exponent'],np.ones(len(x))])
    condition = float(np.linalg.cond(design))
    report = dict(tone_candidates=[dict(exponent=p['exponent'], training_robust_mae=s*255) for s,p in candidates],
                  detail_candidates=[dict(malvar_weight=p['malvar_weight'], detail_strength=p['detail_strength'],training=m)
                                     for m,p in detail_candidates],
                  color_design_rank=int(np.linalg.matrix_rank(design)),
                  color_design_condition=condition if np.isfinite(condition) else None,
                  weak_color_coverage=condition>1e5)
    return chosen, report


def radial_estimate(proxies, max_gain=3, flat=False):
    """拟合 log(G)=图像常量+水平/竖直亮度趋势-a*r²-b*r⁴。

    普通场景无法唯一辨识光学暗角：这里只用多图中值降低场景偏差。
    白板输入可减小歧义；仍需照明均匀、未饱和，不能是暗场照片。
    """
    profiles, coefficients = [], []
    grid = np.linspace(0,1,33)
    for proxy in proxies:
        green = proxy[...,1]
        r2 = radius2(green.shape)
        yy, xx = np.meshgrid(np.linspace(-1,1,len(green)),np.linspace(-1,1,green.shape[1]),indexing='ij')
        valid = (green > .03) & (green < .90)
        if valid.sum() < 200 or not np.any(valid & (r2>.7)) or not np.any(valid & (r2<.15)):
            continue
        design = np.stack([np.ones_like(r2),xx,yy,r2,r2*r2],-1)[valid]
        target = np.log(green[valid])
        coef = np.linalg.lstsq(design,target,rcond=None)[0]
        for _ in range(6):
            residual = design@coef-target
            scale = max(.05,1.4826*np.median(abs(residual-np.median(residual))))
            weight = np.minimum(1,1.5*scale/np.maximum(abs(residual),1e-8))
            coef = np.linalg.solve(design.T@(weight[:,None]*design)+np.eye(5)*1e-7,design.T@(weight*target))
        coefficients.append(-coef[3:])
        profiles.append(-coef[3]*grid-coef[4]*grid*grid)
    if not profiles:
        raise ValueError('暗角拟合缺少同时覆盖中心/边缘的有效亮度；请提供白板 RAW 或更丰富场景')
    profile = np.median(profiles,axis=0)
    design = np.column_stack([grid,grid*grid])
    # 两变量非负最小二乘：内部解+边界解，避免依赖 SciPy 优化器。
    options = [np.zeros(2)]
    unconstrained = np.linalg.lstsq(design,profile,rcond=None)[0]
    if np.all(unconstrained>=0): options.append(unconstrained)
    for i in range(2):
        c = np.zeros(2); c[i] = max(0,design[:,i]@profile/(design[:,i]@design[:,i])); options.append(c)
    coef = min(options,key=lambda c:np.mean((design@c-profile)**2))
    uncapped_gain = float(np.exp(np.clip(coef.sum(),-30,30)))
    coef *= min(1,np.log(max_gain)/max(coef.sum(),1e-12))
    spread = np.percentile(np.array(profiles)[:,-1],[10,50,90]).tolist()
    params = dict(schema_version=1, model='exp(a*r2+b*r2_squared)', coefficients=coef.tolist(), max_gain=max_gain,
                  center='image_center', method='flat_field' if flat else 'ordinary_scene_estimate',
                  reference='center_gain_1', default_enabled=False,
                  note='普通场景的照明/构图与镜头暗角不可唯一分离；这是受最大增益限制的估计，非真实平场标定。' if not flat else
                       '假设输入均匀照明白板；黑帧、照明不均匀或饱和会造成偏差。')
    diagnostics = dict(usable_images=len(profiles), individual_coefficients=[c.tolist() for c in coefficients],
                       log_corner_gain_percentiles_10_50_90=spread, unconstrained_corner_gain=uncapped_gain,
                       actual_corner_gain=float(np.exp(coef.sum())), capped=uncapped_gain>max_gain,
                       physical_identifiability='conditional_on_uniform_flat_field' if flat else 'not_identifiable_from_ordinary_scenes')
    return params, diagnostics


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('dataset',type=Path,nargs='?',help='训练目录；省略时用随附 dataset/train 和 dataset/test')
    p.add_argument('--test-data',type=Path,help='独立测试目录；只评估，不参与参数拟合或选择')
    p.add_argument('--output',type=Path,default=DEFAULT_PARAMS_PATH.parent)
    p.add_argument('--bits',type=int,choices=[8,10,12],default=8)
    p.add_argument('--bayer',choices=['gbrg','grbg','rggb','bggr'],default='gbrg')
    p.add_argument('--groups',type=Path,help='JSON：样例名→场景组名；可选但比自动分组可靠')
    p.add_argument('--validation-fraction',type=float,default=.2);
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--flatfields',type=Path,help='可选：均匀照明白板 RAW 所在目录，支持子目录/raw.raw')
    p.add_argument('--max-vignette-gain',type=float,default=3);
    p.add_argument('--previews',type=int,default=3)
    a = p.parse_args()
    if not 1 <= a.max_vignette_gain <= 8 or a.previews < 0:
        p.error('max-vignette-gain 需为1~8；previews不能为负')
    # 防止参数/图片覆盖源数据；输出目录只存本工具产物。
    dataset = (a.dataset or PROJECT_DIR/'dataset/train').resolve()
    test_dir = a.test_data or (PROJECT_DIR/'dataset/test' if a.dataset is None else None)
    test_dir = test_dir.resolve() if test_dir else None
    output = a.output.resolve()
    if any(d == output or d in output.parents for d in [dataset,test_dir] if d):
        p.error('输出目录必须位于数据目录之外')
    samples, fmt = load_data(dataset,a.bits,a.bayer)
    evaluation_name = 'test' if test_dir else 'validation'
    if test_dir:
        testing, test_fmt = load_data(test_dir,a.bits,a.bayer)
        if test_fmt != fmt: p.error('训练和测试 RAW 格式不同')
        train_count = len(samples)
        samples += testing
        names = [s['name'] for s in samples]
        if len(set(names)) != len(names): p.error('训练/测试样例名重复，请检查是否泄漏同一图像')
        train = np.arange(len(samples)) < train_count
        groups_file = a.groups or (PROJECT_DIR/'dataset/groups.json' if a.dataset is None else None)
        groups = json.loads(groups_file.read_text(encoding='utf-8')) if groups_file else {}
        if groups:
            if set(groups) != set(names): p.error('分组文件必须覆盖全部训练和测试样例')
            if {groups[n] for n in names[:train_count]} & {groups[n] for n in names[train_count:]}:
                p.error('相同场景组同时出现在训练/测试集')
        split = dict(method='explicit_train_test_folders',groups=groups,seed=a.seed,
                     train=names[:train_count],test=names[train_count:],test_used_for_fitting=False)
    else:
        train, split = split_data(samples,a.groups,a.validation_fraction,a.seed)
    print(f'训练 {train.sum()} 张，留出 {(~train).sum()} 张；全部共用同一套参数',flush=True)
    params, search = fit_isp(samples,train,fmt,a.seed)
    params.update(schema_version=1, fit_scope='global_for_all_images', seed=a.seed)
    proxies = [s['proxy'] for s,t in zip(samples,train) if t]
    if a.flatfields:
        flatpaths = sorted(a.flatfields.glob('*.raw'))+sorted(a.flatfields.glob('*/raw.raw'))
        if not flatpaths: raise ValueError('白板目录没有 RAW 文件')
        proxies = []
        for path in flatpaths:
            raw, current, _ = read_raw(path,fmt)
            if current != fmt: raise ValueError('白板 RAW 格式必须与标定数据一致')
            proxies.append(area_rgb(raw,fmt))
    try:
        vig, vig_report = radial_estimate(proxies,a.max_vignette_gain,bool(a.flatfields))
    except ValueError as error:
        if a.flatfields:  # 显式白板输入无效时要报错，不能悄悄当成成功。
            raise
        # 可选暗角估计失败不影响基本 ISP：保存单位增益，并明确记录原因。
        vig = dict(schema_version=1,coefficients=[0,0],max_gain=a.max_vignette_gain,
                   method='insufficient_data_identity',default_enabled=False,note=str(error))
        vig_report = dict(usable_images=0,actual_corner_gain=1,status='not_fitted',reason=str(error))
    report = dict(split=split, search=search, vignette=vig_report,
                  metrics_scope='五处固定原分辨率区域，去掉2像素边界；非整图指标；未配准的异步图像',
                  missing_metadata_samples=[s['name'] for s in samples if not s['has_metadata']],
                  format_defaults_when_missing=dict(bits=a.bits,bayer=a.bayer,dimensions='camera_rgb.png'),samples=[])
    sums = {split_name:{mode:np.zeros(4) for mode in ('off','on')} for split_name in ('train',evaluation_name)}
    for sample, is_train in zip(samples,train):
        totals = {mode:np.zeros(4) for mode in ('off','on')}
        for record in sample['records']:
            for mode, shading in [('off',None),('on',vig)]:
                totals[mode] += errors(patch_prediction(record,params,shading),record['y'][2:-2,2:-2])
        name = 'train' if is_train else evaluation_name
        for mode in totals: sums[name][mode] += totals[mode]
        report['samples'].append(dict(sample=sample['name'],split=name,**{k:describe(v) for k,v in totals.items()}))
    report['metrics'] = {s:{k:describe(v) for k,v in modes.items()} for s,modes in sums.items()}
    report['vignette_metric_note'] = '相机参考也可能有暗角；主动补偿后，与相机的MAE上升不等于补偿失败，也不证明真实亮度恢复。'
    output.mkdir(parents=True,exist_ok=True)
    for name,data in [('params.json',params),('vignette.json',vig),('report.json',report),('split.json',split)]:
        save_json(output/name,data)
    # 预览从全分辨率推理结果缩小，确实经过最终运行路径；左参考/中关闭/右开启。
    indices = np.linspace(0,len(samples)-1,min(a.previews,len(samples)),dtype=int)
    for index in indices:
        sample = samples[index]; raw,_,_ = read_raw(sample['path'],fmt)
        with Image.open(sample['path'].with_name('camera_rgb.png')) as im: reference=im.copy()
        views = [reference,Image.fromarray(render(raw,params)),Image.fromarray(render(raw,params,vig))]
        width = 720; height = round(width*fmt['height']/fmt['width'])
        preview = Image.new('RGB',(width*3,height+28),'#202020'); draw=ImageDraw.Draw(preview)
        for i,(im,label) in enumerate(zip(views,('Camera reference','ISP / vignette OFF','ISP / vignette ON'))):
            preview.paste(im.resize((width,height),Image.Resampling.BOX),(i*width,28)); draw.text((i*width+8,8),label,fill='white')
        preview.save(output/f"preview_{index:04d}.jpg",quality=92)
    print(json.dumps(dict(parameters=str(output/'params.json'),vignette=str(output/'vignette.json'),
                         evaluation_set=evaluation_name,evaluation=report['metrics'][evaluation_name],
                         vignette_corner_gain=vig_report['actual_corner_gain']),indent=2))


if __name__ == '__main__':
    main()
