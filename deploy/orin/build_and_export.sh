#!/bin/bash
# ==============================================================================
# 受电弓网故障监测系统 - Jetson AGX Orin 镜像打包与导出脚本 (build_and_export.sh)
# ==============================================================================

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_ROOT}"

echo "======================================================================"
echo "📦 [构建引擎] 开始制作 Jetson AGX Orin 弓网一体化完整 Docker 镜像"
echo "======================================================================"

BASE_TAR="deploy/orin/images/vllm_jetson_orin_base.tar"
BASE_TAG="ghcr.io/nvidia-ai-iot/vllm:r36.4.tegra-aarch64-cu126-22.04"
TARGET_IMAGE="pantograph-vllm-orin:v1.0"
OUTPUT_TAR="deploy/orin/images/pantograph-vllm-orin-v1.0.tar"

# 1. 检查基础镜像
if ! docker image inspect "${BASE_TAG}" > /dev/null 2>&1; then
    if [ -f "${BASE_TAR}" ]; then
        echo ">> 正在从本地归档加载基础镜像: ${BASE_TAR}..."
        docker load -i "${BASE_TAR}"
    else
        echo "❌ 错误: 未找到基础镜像 ${BASE_TAG} 且 ${BASE_TAR} 不存在！"
        exit 1
    fi
fi

# 2. 检查 AWQ 模型文件
MODEL_FILE="models/Qwen3-VL-4B-Instruct-AWQ-4bit/model.safetensors"
if [ ! -f "${MODEL_FILE}" ]; then
    echo "❌ 错误: 模型权重文件 ${MODEL_FILE} 不存在或尚未下载完毕！"
    echo "   请确保 models/Qwen3-VL-4B-Instruct-AWQ-4bit 目录包含完整 safetensors 权重。"
    exit 1
fi

echo ">> 正在构建 ARM64 一体化镜像: ${TARGET_IMAGE}..."
docker build \
    --platform linux/arm64 \
    -f deploy/orin/Dockerfile \
    -t "${TARGET_IMAGE}" \
    .

echo ">> 正在导出为 Jetson 离线加载包: ${OUTPUT_TAR} (请稍候)..."
docker save -o "${OUTPUT_TAR}" "${TARGET_IMAGE}"

echo "----------------------------------------------------------------------"
echo "✅ 构建并导出成功！"
echo "📁 导出文件路径: ${PROJECT_ROOT}/${OUTPUT_TAR}"
ls -lh "${OUTPUT_TAR}"
echo "======================================================================"
