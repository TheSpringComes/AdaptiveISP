import math
import cv2
import torch
import os
import sys
import random
import numpy as np
import threading
import yaml


def _deep_merge(base: dict, override: dict) -> dict:
    """递归合并 override 到 base（override 优先，dict 深合并，其余整体替换）。

    list（operators / rules / groups 等）一律整体替换——消融变体重新
    列出完整列表比"按索引合并"更可读也更安全。
    """
    out = dict(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _load_yaml_with_base(path: str, _seen: tuple = ()) -> dict:
    """读一个 yaml；若声明 `base:` 键则先递归加载 base 再做 override 合并。

    `base: configs/base/human.yaml`（相对仓库根或相对本文件所在目录均可）。
    禁止继承环；`base` 键本身在合并完成后从结果中移除。
    """
    with open(path, "r") as f:
        data = yaml.safe_load(f) or {}

    base_ref = data.pop('base', None)
    if base_ref is None:
        return data
    if not isinstance(base_ref, str):
        raise ValueError(f"{path}: 'base' 必须是一个 yaml 路径字符串")

    # 相对路径：先按相对仓库根解析（与 --cfg 的用法一致），再按相对
    # 本文件所在目录解析（允许 base/ 目录内部互相引用）。
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cand = [base_ref,
            os.path.join(repo_root, base_ref),
            os.path.join(os.path.dirname(os.path.abspath(path)), base_ref)]
    base_path = next((c for c in cand if os.path.isfile(c)), None)
    if base_path is None:
        raise FileNotFoundError(f"{path}: base 配置不存在: {base_ref}")
    base_path = os.path.abspath(base_path)
    if base_path in _seen:
        raise ValueError(f"config 继承环: {' -> '.join(_seen + (base_path,))}")

    base_data = _load_yaml_with_base(base_path, _seen + (os.path.abspath(path),))
    if not isinstance(base_data, dict):
        raise ValueError(f"{base_path}: base 配置顶层必须是映射")
    return _deep_merge(base_data, data)


def load_config(path: str):
    """Load a config yaml (or fall back to a python module for legacy .py paths).

    yaml 支持 `base:` 继承：变体配置只写与 base 的差异，加载时递归合并
    （见 `_load_yaml_with_base`）。

    Returns a `Dict` for dot-attribute access, with derived fields populated:
      - `num_state_dim`  (defaults to 3 + len(operators))
      - `z_dim`          (defaults to 3 + len(operators) * z_dim_per_filter)
    Requires the config to define an `operators` list; raises `ValueError`
    otherwise. Shared by `engine.trainer`, `engine.trainer_human`, and
    `engine.evaluator` so all three paths see identical derived fields.
    """
    if path.endswith(".yaml") or path.endswith(".yml"):
        data = _load_yaml_with_base(path)
        cfg = Dict(data)
    else:
        # Legacy: python module import (e.g., --cfg config)
        import importlib
        cfg = importlib.import_module(path).cfg

    if 'operators' not in cfg:
        raise ValueError(f"config missing 'operators' list: {path}")
    if 'num_state_dim' not in cfg:
        cfg.num_state_dim = 3 + len(cfg.operators)
    if 'z_dim' not in cfg:
        cfg.z_dim = 3 + len(cfg.operators) * cfg.get('z_dim_per_filter', 16)
    return cfg


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Seed all RNGs for reproducibility.

    Must be called BEFORE any model / dataset / dataloader construction.
    Covers: python random, numpy, torch CPU, torch CUDA (all devices),
    PYTHONHASHSEED, and cuDNN determinism flags.
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def save_img(img, img_path, save_path, prefix=None, format="CHW", is_train=True):
    if isinstance(img, torch.Tensor):
        img = img.detach().cpu().numpy()
    # print(img.shape, len(img.shape))
    if len(img.shape) > 3:
        img = img.squeeze(0)
    if format.upper() == "CHW":
        img = np.transpose(img, (1, 2, 0))
    img[np.isnan(img)] = 0.
    # print(img.shape, format)
    img = np.clip(img, a_min=0.0, a_max=1.0)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    _, fullflname = os.path.split(img_path)
    fname, ext = os.path.splitext(fullflname)

    if is_train:
        os.makedirs(os.path.join(save_path, fname), exist_ok=True)
        cv2.imwrite(os.path.join(save_path, fname, fname + ('' if prefix is None else f'_{prefix}.png')), img * 255.0)
    else:
        cv2.imwrite(os.path.join(save_path, fname + ('' if prefix is None else f'_{prefix}')) + ext, img * 255.0)


import matplotlib.pyplot as plt
def show(x, title="a", format="HWC", is_finish=True):
    if len(x.shape) > 3:
        print(f"Warning input image shape is {x.shape}, just show first image")
        x = x[0]
    if format == 'CHW':
        x = np.transpose(x, (1, 2, 0))
    plt.figure()
    plt.cla()
    plt.title(title)
    plt.imshow(x)
    if is_finish:
        plt.show()


# based on https://stackoverflow.com/questions/2352181/how-to-use-a-dot-to-access-members-of-dictionary
class Dict(dict):
    """
      Example:
      m = Dict({'first_name': 'Eduardo'}, last_name='Pool', age=24, sports=['Soccer'])
      """

    def __init__(self, *args, **kwargs):
        super(Dict, self).__init__(*args, **kwargs)
        for arg in args:
            if isinstance(arg, dict):
                for k, v in arg.items():
                    self[k] = v

        if kwargs:
            for k, v in kwargs.items():
                self[k] = v

    def __getattr__(self, attr):
        return self[attr]

    def __setattr__(self, key, value):
        self.__setitem__(key, value)

    def __setitem__(self, key, value):
        super(Dict, self).__setitem__(key, value)
        self.__dict__.update({key: value})

    def __delattr__(self, item):
        self.__delitem__(item)

    def __delitem__(self, key):
        super(Dict, self).__delitem__(key)
        del self.__dict__[key]


def make_image_grid(images, per_row=2, padding=2): #  per_row =8
    npad = ((0, 0), (padding, padding), (padding, padding), (0, 0))
    images = np.pad(images, pad_width=npad, mode='constant', constant_values=1.0)
    assert images.shape[0] % per_row == 0
    num_rows = images.shape[0] // per_row
    image_rows = []
    for i in range(num_rows):
        image_rows.append(np.hstack(images[i * per_row:(i + 1) * per_row]))
    return np.vstack(image_rows)


class Tee(object):

    def __init__(self, name):
        self.file = open(name, 'w')
        self.stdout = sys.stdout
        self.stderr = sys.stderr
        sys.stdout = self
        sys.stderr = self

    def __del__(self):
        self.file.close()

    def write(self, data):
        self.file.write(data)
        self.stdout.write(data)
        self.file.flush()
        self.stdout.flush()

    def write_to_file(self, data):
        self.file.write(data)

    def flush(self):
        self.file.flush()


def merge_dict(a, b):
    ret = a.copy()
    for key, val in list(b.items()):
        if key in ret:
            assert False, 'Item ' + key + 'already exists'
        else:
            ret[key] = val
    return ret


def lerp(a, b, l):
    return (1 - l) * a + l * b



class AsyncTaskManager:

    def __init__(self, target, args=(), kwargs={}):
        self.target = target
        self.args = args
        self.kwargs = kwargs
        self.condition = threading.Condition()
        self.result = None
        self.thread = threading.Thread(target=self.worker)
        self.stopped = False
        self.thread.daemon = True
        self.thread.start()

    def worker(self):
        while True:
            self.condition.acquire()
            while self.result is not None:
                if self.stopped:
                    self.condition.release()
                    return
                self.condition.notify()
                self.condition.wait()
            self.condition.notify()
            self.condition.release()

            result = (self.target(*self.args, **self.kwargs),)

            self.condition.acquire()
            self.result = result
            self.condition.notify()
            self.condition.release()

    def get_next(self):
        self.condition.acquire()
        while self.result is None:
            self.condition.notify()
            self.condition.wait()
        result = self.result[0]
        self.result = None
        self.condition.notify()
        self.condition.release()
        return result

    def stop(self):
        while self.thread.is_alive():
            self.condition.acquire()
            self.stopped = True
            self.condition.notify()
            self.condition.release()
