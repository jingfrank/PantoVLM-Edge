# ==============================================================================
# 受电弓网故障监测系统 - Jetson AGX Orin 边缘部署 Dockerfile (v2.0)
# 基于 pantograph-subsystem-jetson:v1.0 增量升级 Python 3.10 与 Qwen3-VL 现代大模型依赖
# ==============================================================================

FROM pantograph-subsystem-jetson:v1.0

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Shanghai
ENV PYTHONUNBUFFERED=1

# 1. 下载并安装 Standalone Python 3.10 (ARM64 aarch64 纯净运行时)
RUN python3 -c "import urllib.request; urllib.request.urlretrieve('https://ghproxy.net/https://github.com/astral-sh/python-build-standalone/releases/download/20250115/cpython-3.10.16+20250115-aarch64-unknown-linux-gnu-install_only.tar.gz', '/tmp/python.tar.gz')" && \
    mkdir -p /opt/python310 && \
    tar -xzf /tmp/python.tar.gz -C /opt/python310 --strip-components=1 && \
    rm /tmp/python.tar.gz

# 2. 安装现代 PyTorch 与 Qwen3-VL 依赖 (配置国内清华镜像源主通道 + PyTorch CPU aarch64 扩展源，设置长超时保护)
RUN /opt/python310/bin/pip install --no-cache-dir --default-timeout=1000 \
        -i https://pypi.tuna.tsinghua.edu.cn/simple \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        torch torchvision && \
    /opt/python310/bin/pip install --no-cache-dir --default-timeout=1000 \
        -i https://pypi.tuna.tsinghua.edu.cn/simple \
        "transformers>=4.57.0" accelerate ultralytics opencv-python-headless openpyxl Pillow pyyaml

# 3. 设置系统默认 Python 为 Python 3.10
ENV PATH="/opt/python310/bin:$PATH"

WORKDIR /app
COPY . /app

CMD ["python3", "main.py", "config/config_prod.yaml"]
