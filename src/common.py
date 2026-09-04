"""通用工具：随机种子固定与路径配置加载。

所有实验脚本入口必须调用 set_seed()；所有路径必须经 load_paths() 获取，
禁止在代码中硬编码（原始数据位于中文路径，集中在 configs/paths.json 管理）。

路径分两组（见 configs/paths.json）：
- external：原始数据/文献，相对 data_root（文档仓）解析，只读；
- internal：本项目产物（清洗数据/结果/图），相对代码仓根（PROJECT_ROOT）解析，目录自动创建。
"""
import json
import random
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]  # .../v6.0_src
CONFIG_DIR = PROJECT_ROOT / "configs"


def set_seed(seed: int) -> None:
    """固定 random/numpy/torch 随机种子，保证实验可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def load_paths() -> dict[str, Path]:
    """读取 configs/paths.json，返回解析后的绝对路径字典。

    external 组相对 data_root（只读，不创建）；internal 组相对代码仓根，*_dir 自动创建。
    """
    with open(CONFIG_DIR / "paths.json", encoding="utf-8") as f:
        cfg = json.load(f)

    data_root = Path(cfg["data_root"])
    paths: dict[str, Path] = {"data_root": data_root}
    for key, value in cfg.get("external", {}).items():
        paths[key] = data_root / value
    for key, value in cfg.get("internal", {}).items():
        p = PROJECT_ROOT / value
        if key.endswith("_dir"):
            p.mkdir(parents=True, exist_ok=True)
        paths[key] = p
    return paths


def load_config(name: str) -> dict:
    """读取 configs/ 下指定实验配置（JSON），如 load_config("exp001_baseline_rf")。"""
    with open(CONFIG_DIR / f"{name}.json", encoding="utf-8") as f:
        return json.load(f)
