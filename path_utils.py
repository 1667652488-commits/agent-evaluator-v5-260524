#!/usr/bin/env python3
"""
路径适配工具 — 自动适配 Windows / Linux / 不同运行环境
"""
import os
import sys
import json
from pathlib import Path

# 自动检测环境
IS_WINDOWS = sys.platform.startswith("win")
IS_LINUX = sys.platform.startswith("linux")

# 项目根目录 = 本文件所在目录的父目录
PROJECT_ROOT = Path(__file__).parent.resolve()


def load_config(config_path: str = "./config.json") -> dict:
    """加载全局配置文件"""
    p = Path(config_path)
    if not p.exists():
        p = PROJECT_ROOT / "config.json"
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def get_path(relative_path: str) -> Path:
    """
    将相对路径转换为绝对路径，基于项目根目录。
    自动处理 Windows / Linux 路径差异。
    """
    # 清理路径中的环境特定前缀（如 /mnt/...）
    cleaned = relative_path.replace("\\", "/")
    if cleaned.startswith("/mnt/") and IS_WINDOWS:
        # WSL 路径映射回 Windows
        cleaned = cleaned.replace("/mnt/c", "C:").replace("/mnt/d", "D:")
    # 统一使用 Path 处理
    target = PROJECT_ROOT / cleaned
    return target.resolve()


def ensure_dir(path: Path) -> Path:
    """确保目录存在，不存在则创建"""
    path.mkdir(parents=True, exist_ok=True)
    return path


def adapt_env_path(raw_path: str) -> str:
    """
    根据当前运行环境调整路径字符串。
    例如：在 Linux 上保留 /mnt/...，在 Windows 上转为 C:\...
    """
    if IS_WINDOWS:
        # Linux 绝对路径 → 尝试映射到 WSL 盘符
        if raw_path.startswith("/mnt/"):
            drive = raw_path[5].upper()
            rest = raw_path[6:].replace("/", "\\")
            return f"{drive}:\\{rest}"
        if raw_path.startswith("/root/"):
            # 无法直接映射，返回原样让调用方处理
            return raw_path
    return raw_path


# 预定义常用路径
def data_preprocessing_dir() -> Path:
    return get_path("data_preprocessing")


def agent_dir() -> Path:
    return get_path("agent")


def evaluator_dir() -> Path:
    return get_path("evaluator")


def optimizer_dir() -> Path:
    return get_path("optimizer")


def pipeline_dir() -> Path:
    return get_path("pipeline")
