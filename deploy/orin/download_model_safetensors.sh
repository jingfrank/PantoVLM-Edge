#!/bin/bash
TARGET_DIR="/mnt/T4/jyb/gongwang_jetson_vllm/models/Qwen3-VL-4B-Instruct-AWQ-4bit"
TARGET_FILE="${TARGET_DIR}/model.safetensors"
URL="https://huggingface.co/cyankiwi/Qwen3-VL-4B-Instruct-AWQ-4bit/resolve/main/model.safetensors"
PROXY="http://127.0.0.1:7897"

echo "🚀 开始下载 Qwen3-VL-4B-Instruct-AWQ-4bit model.safetensors (支持断点续传与无限重试)..."

while true; do
    curl -x "${PROXY}" \
         -L -C - \
         --connect-timeout 30 \
         --max-time 1800 \
         --retry 100 \
         --retry-delay 3 \
         -o "${TARGET_FILE}" \
         "${URL}"
    
    EXIT_CODE=$?
    if [ $EXIT_CODE -eq 0 ]; then
        echo "✅ model.safetensors 下载完毕！"
        break
    else
        echo "⚠️ 下载中断 (code: ${EXIT_CODE})，5秒后自动断点续传..."
        sleep 5
    fi
done
