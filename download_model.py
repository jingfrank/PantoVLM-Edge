#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
一键下载 Qwen-VL 官方大模型权重脚本 (download_model.py)
默认使用国内 ModelScope 极速镜像源
"""

import os
import sys
import argparse

def download_qwen_model(model_name="Qwen/Qwen3-VL-4B-Instruct", save_dir="./models/Qwen3-VL-4B-Instruct"):
    print("--------------------------------------------------")
    print(f"[Qwen-VL 下载器] 正在检查/安装 modelscope 镜像工具...")
    os.system(f"{sys.executable} -m pip install modelscope -q")

    try:
        from modelscope import snapshot_download
        abs_save_dir = os.path.abspath(save_dir)
        print(f"[Qwen-VL 下载器] 目标模型: {model_name}")
        print(f"[Qwen-VL 下载器] 保存目标路径: {abs_save_dir}")
        print(f"[Qwen-VL 下载器] 正在从 ModelScope 镜像源下载 {model_name}，请稍候...\n")

        res_path = snapshot_download(model_name, local_dir=abs_save_dir)
        print(f"\n✅ [Qwen-VL 下载器] 模型权重成功下载完成！")
        print(f"   ├─ 绝对路径: {res_path}")
        print(f"   └─ 您可使用 TensorRT-LLM 命令将该权重编译为 .engine 硬件加速文件。")
        print("--------------------------------------------------\n")
    except Exception as e:
        print(f"\n❌ [Qwen-VL 下载器] 下载失败: {e}", file=sys.stderr)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Qwen-VL 大模型权重一键下载器")
    parser.add_argument("--model", default="Qwen/Qwen3-VL-4B-Instruct", help="ModelScope 模型 ID")
    parser.add_argument("--out", default="./models/Qwen3-VL-4B-Instruct", help="本地输出路径")
    args = parser.parse_args()

    download_qwen_model(args.model, args.out)

