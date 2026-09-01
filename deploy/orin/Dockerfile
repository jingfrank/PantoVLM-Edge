# ==============================================================================
# 受电弓网故障监测系统 - Jetson AGX Orin 官方 vLLM 一体化完整镜像
# 基础镜像: ghcr.io/nvidia-ai-iot/vllm:r36.4.tegra-aarch64-cu126-22.04
# 内置环境: JetPack 6.x (r36.4) + CUDA 12.6 + PyTorch + Jetson 适配版 vLLM
# 内置模型: Qwen3-VL-4B-Instruct-AWQ-4bit
# ==============================================================================

FROM ghcr.io/nvidia-ai-iot/vllm:r36.4.tegra-aarch64-cu126-22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Shanghai
ENV PYTHONUNBUFFERED=1
ENV VLLM_MODEL_PATH=/app/models/Qwen3-VL-4B-Instruct-AWQ-4bit
ENV VLLM_MODEL_NAME=Qwen3-VL-4B-Instruct
ENV VLLM_GPU_MEMORY=0.8
ENV VLLM_MAX_MODEL_LEN=4096

WORKDIR /app

# 1. 安装弓网监测业务 Python 依赖包 (复用容器内自带的 Jetson 优化版 PyTorch)
RUN pip install --no-cache-dir \
    -i https://pypi.tuna.tsinghua.edu.cn/simple \
    openpyxl \
    pyyaml \
    Pillow \
    requests \
    opencv-python-headless

# 2. 复制预下载的 AWQ 4bit 大模型权重与源码
COPY models/Qwen3-VL-4B-Instruct-AWQ-4bit /app/models/Qwen3-VL-4B-Instruct-AWQ-4bit
COPY . /app

# 3. 设置启动入口权限
RUN chmod +x /app/deploy/orin/entrypoint.sh

# 4. 容器入口
ENTRYPOINT ["/app/deploy/orin/entrypoint.sh"]
