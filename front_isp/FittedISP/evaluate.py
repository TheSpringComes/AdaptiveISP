"""整图测试，每个样例保存 InfiniteISP / Ours / Ours-Vig / Camera 四张 PNG。"""
import argparse
import hashlib
import json
import shutil
from pathlib import Path
import numpy as np
from PIL import Image
from isp import PROJECT_DIR, DEFAULT_PARAMS_PATH, DEFAULT_VIGNETTE_PATH, read_raw, render


def metric(a, b):
    absolute=squared=count=0
    for y in range(0,len(a),128):  # 按行累计，避免多个整图 float64 数组占用内存。
        d=a[y:y+128].astype(np.float64)-b[y:y+128]
        absolute+=abs(d).sum(); squared+=(d*d).sum(); count+=d.size
    mse=float(squared/count)
    return dict(mae=float(absolute/count),rmse=float(np.sqrt(mse)),
                psnr_db=float(10*np.log10(255**2/mse)) if mse else None,channel_values=count)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('dataset',type=Path,nargs='?',default=PROJECT_DIR/'dataset/test')
    p.add_argument('--params',type=Path,default=DEFAULT_PARAMS_PATH)
    p.add_argument('--vignette',type=Path,nargs='?',const=DEFAULT_VIGNETTE_PATH,default=DEFAULT_VIGNETTE_PATH,
                   help='暗角参数文件；测试固定同时输出关闭/开启两种结果')
    p.add_argument('--output',type=Path,default=PROJECT_DIR/'outputs/test')
    a=p.parse_args(); dataset=a.dataset.resolve(); output=a.output.resolve()
    if output==dataset or dataset in output.parents: p.error('输出目录不可位于测试数据目录内')
    params=json.loads(a.params.read_text(encoding='utf-8'))
    vig=json.loads(a.vignette.read_text(encoding='utf-8'))
    paths=sorted(dataset.glob('*/raw.raw'))
    if not paths: p.error('测试目录中没有 */raw.raw')
    # 提前检查所需参考文件，避免跑到中途才发现缺失。
    for path in paths:
        for name in ('camera_rgb.png','infinite_isp_rgb.png'):
            if not path.with_name(name).is_file(): p.error(f'缺少 {path.with_name(name)}')
    rows={mode:[] for mode in ('infinite','off','on')}
    for i,path in enumerate(paths):
        raw,fmt,_=read_raw(path,params['format'])
        if fmt!=params['format']: raise ValueError(f'格式与拟合参数不一致：{path}')
        with Image.open(path.with_name('camera_rgb.png')) as im:
            if im.mode!='RGB': raise ValueError('camera_rgb.png 必须为 RGB 图像')
            target=np.asarray(im).copy()
        if target.shape!=(fmt['height'],fmt['width'],3): raise ValueError('参考图与 RAW 尺寸不同')
        folder=output/path.parent.name; folder.mkdir(parents=True,exist_ok=True)
        # 原始参考直接复制；两种预测保存完整分辨率，不生成拼接图。
        shutil.copy2(path.with_name('infinite_isp_rgb.png'),folder/'1-InfiniteISP.png')
        shutil.copy2(path.with_name('camera_rgb.png'),folder/'4-Camera.png')
        for mode in ('infinite','off','on'):
            if mode=='infinite':
                with Image.open(path.with_name('infinite_isp_rgb.png')) as im:
                    if im.mode!='RGB': raise ValueError('infinite_isp_rgb.png 必须为 RGB 图像')
                    predicted=np.asarray(im).copy()
            else:
                predicted=render(raw,params,vig if mode=='on' else None)
                filename='2-Ours.png' if mode=='off' else '3-Ours-Vig.png'
                Image.fromarray(predicted).save(folder/filename)
            if predicted.shape!=target.shape: raise ValueError(f'{mode} 与 GT 尺寸不同：{path}')
            row=dict(sample=path.parent.name,**metric(predicted,target)); rows[mode].append(row)
            (folder/f'metrics_{mode}.json').write_text(json.dumps(row,indent=2,ensure_ascii=False),encoding='utf-8')
            del predicted
        scores=' / '.join(f'{mode}={rows[mode][-1]["mae"]:.4f}' for mode in rows)
        print(f'{i+1}/{len(paths)} {path.parent.name}: MAE {scores}',flush=True)
    for mode,scores in rows.items():
        total=sum(r['channel_values'] for r in scores)
        mse=sum(r['rmse']**2*r['channel_values'] for r in scores)/total
        report=dict(samples=scores,count=len(scores),vignette_enabled=mode=='on',
                method=dict(infinite='InfiniteISP',off='Ours',on='Ours (Vignette corrected)')[mode],
                params_sha256=hashlib.sha256(a.params.read_bytes()).hexdigest(),
                vignette_sha256=hashlib.sha256(a.vignette.read_bytes()).hexdigest() if mode=='on' else None,
                scope='完整分辨率、未配准，参数固定；测试图不参与拟合',
                pooled=dict(mae=sum(r['mae']*r['channel_values'] for r in scores)/total,
                            rmse=float(np.sqrt(mse)),psnr_db=float(10*np.log10(255**2/mse)) if mse else None),
                note='开启补偿会主动改变相机也保留的暗角；与相机的误差不代表真实平场恢复质量。')
        (output/f'report_{mode}.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
        print(mode,json.dumps(report['pooled'],indent=2))


if __name__=='__main__': main()
