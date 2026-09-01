#!/bin/bash
# ==============================================================================
# 受电弓网故障监测系统 - Jetson AGX Orin 边缘端一键加载与部署脚本
# 适用设备: NVIDIA Jetson AGX Orin (JetPack 6.x / L4T r36.x)
# ==============================================================================

set -e

IMAGE_TAR="${1:-pantograph-vllm-orin-v1.0.tar}"
IMAGE_TAG="pantograph-vllm-orin:v1.0"
CONTAINER_NAME="gongwang_monitor_vllm"

echo "======================================================================"
echo "🚀 [Jetson AGX Orin 部署] 弓网故障监测系统 (vLLM AWQ 版)"
echo "======================================================================"

# 1. 检查 Docker 及 NVIDIA Runtime
if ! command -v docker &> /dev/null; then
    echo "❌ 错误: 未安装 Docker，请先安装 Docker 并配置 nvidia-container-runtime！"
    exit 1
fi

if ! docker info 2>/dev/null | grep -i -q 'runtimes:.*nvidia'; then
    echo "⚠️ 警告: Docker 未检测到默认 nvidia runtime，请确认 /etc/docker/daemon.json 中包含 nvidia runtime。"
fi

# 2. 加载镜像归档包
if [ -f "${IMAGE_TAR}" ]; then
    echo ">> [1/3] 正在从 ${IMAGE_TAR} 加载 Docker 镜像 (docker load)..."
    sudo docker load -i "${IMAGE_TAR}"
else
    echo ">> 镜像归档 ${IMAGE_TAR} 不存在，检查本地镜像库中是否存在 ${IMAGE_TAG}..."
    if ! sudo docker image inspect "${IMAGE_TAG}" > /dev/null 2>&1; then
        echo "❌ 错误: 本地未找到镜像 ${IMAGE_TAG}，请传入有效的 tar 包路径！"
        echo "用法: sudo ./deploy_on_jetson.sh [镜像包路径.tar]"
        exit 1
    fi
fi

# 3. 创建本地持久化缓存与结果目录
RESULT_DIR="/hirain/hi_opt/nvme1/Algo-result"
mkdir -p "${RESULT_DIR}"
echo ">> [2/3] 确认本地结果存储路径: ${RESULT_DIR}"

# 4. 停止并移除已有容器
if sudo docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo ">> 停止已有容器: ${CONTAINER_NAME}..."
    sudo docker stop "${CONTAINER_NAME}" || true
    sudo docker rm "${CONTAINER_NAME}" || true
fi

# 5. 启动一体化监测系统容器
echo ">> [3/3] 启动弓网监测系统容器..."
sudo docker run -d \
    --name "${CONTAINER_NAME}" \
    --restart always \
    --runtime=nvidia \
    --network host \
    -e NVIDIA_VISIBLE_DEVICES=all \
    -e NVIDIA_DRIVER_CAPABILITIES=all \
    -e TZ=Asia/Shanghai \
    -e PYTHONUNBUFFERED=1 \
    -e VLLM_GPU_MEMORY=0.8 \
    -e VLLM_MAX_MODEL_LEN=4096 \
    -v /etc/localtime:/etc/localtime:ro \
    -v /etc/timezone:/etc/timezone:ro \
    -v "${RESULT_DIR}":/app/local_cache/Algo-result \
    "${IMAGE_TAG}"

echo "======================================================================"
echo "🎉 部署完成！容器 ${CONTAINER_NAME} 已在后台运行。"
echo "----------------------------------------------------------------------"
echo "查看运行日志: sudo docker logs -f ${CONTAINER_NAME}"
echo "停止服务:     sudo docker stop ${CONTAINER_NAME}"
echo "重启服务:     sudo docker restart ${CONTAINER_NAME}"
echo "======================================================================"
