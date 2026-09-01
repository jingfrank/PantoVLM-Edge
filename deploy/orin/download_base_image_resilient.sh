#!/bin/bash
DIR_CACHE="/mnt/T4/jyb/gongwang_jetson_vllm/deploy/orin/images/vllm_dir_cache"
TAR_OUT="/mnt/T4/jyb/gongwang_jetson_vllm/deploy/orin/images/vllm_jetson_orin_base.tar"
SRC_IMAGE="docker://ghcr.io/nvidia-ai-iot/vllm:r36.4.tegra-aarch64-cu126-22.04"
TAG="ghcr.io/nvidia-ai-iot/vllm:r36.4.tegra-aarch64-cu126-22.04"

mkdir -p "${DIR_CACHE}"

echo "🚀 [基础镜像下载器] 使用 Clash Verge 网络，同步分层至本地持久化目录 ${DIR_CACHE}..."

while true; do
    /tmp/skopeo copy \
        --insecure-policy \
        --retry-times 20 \
        --override-arch arm64 \
        --override-os linux \
        "${SRC_IMAGE}" \
        dir:"${DIR_CACHE}"
    
    EXIT_CODE=$?
    if [ $EXIT_CODE -eq 0 ]; then
        echo "✅ 所有分层已下载完毕！正在封包为 tar 归档: ${TAR_OUT}..."
        /tmp/skopeo copy \
            --insecure-policy \
            dir:"${DIR_CACHE}" \
            docker-archive:"${TAR_OUT}":"${TAG}"
        echo "🎉 基础镜像归档制作完成: ${TAR_OUT}"
        break
    else
        echo "⚠️ 连接重试中 (code: ${EXIT_CODE})，5秒后继续从断点续传..."
        sleep 5
    fi
done
