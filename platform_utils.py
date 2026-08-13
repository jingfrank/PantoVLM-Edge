#!/usr/bin/env python3
"""运行时硬件平台自动检测与配置适配 (platform_utils.py)"""

import platform

def detect_platform():
    """自动检测当前硬件平台。"""
    arch = platform.machine()
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            if "Orin" in gpu_name or "Tegra" in gpu_name:
                return "orin"
    except ImportError:
        pass
    return "orin" if arch == "aarch64" else "x86"

def get_gpu_memory_fraction(platform_id=None):
    """根据平台返回合适的 GPU 显存配额。"""
    pid = platform_id or detect_platform()
    return 0.50 if pid == "orin" else 0.50

def get_trt_engine_suffix():
    """返回当前平台的 TRT 引擎文件后缀。"""
    import torch
    sm = torch.cuda.get_device_capability(0)
    return f"sm{sm[0]}{sm[1]}"

def supports_flash_attention():
    """检测是否支持 FlashAttention-2 (sm >= 80)。"""
    import torch
    sm = torch.cuda.get_device_capability(0)
    return sm[0] >= 8
