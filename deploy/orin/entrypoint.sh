#!/bin/bash
# ==============================================================================
# 受电弓网故障监测系统 - Jetson AGX Orin 容器自启动入口 (entrypoint.sh)
# 1. 自动拉起后台 vLLM 推理服务 (AWQ-4bit 量化大模型)
# 2. 探针等待 vLLM API 就绪
# 3. 拉起弓网主监测进程 (main.py)
# ==============================================================================

set -e

echo "======================================================================"
echo "🚀 [Jetson AGX Orin] 弓网故障监测系统正在启动..."
echo "======================================================================"

# 配置参数 (支持环境变量覆盖)
MODEL_PATH="${VLLM_MODEL_PATH:-/app/models/Qwen3-VL-4B-Instruct-AWQ-4bit}"
SERVED_MODEL_NAME="${VLLM_MODEL_NAME:-Qwen3-VL-4B-Instruct}"
GPU_MEM="${VLLM_GPU_MEMORY:-0.8}"
MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-4096}"
VLLM_PORT="${VLLM_PORT:-8000}"
CONFIG_FILE="${APP_CONFIG:-config/config_prod.yaml}"

# 1. 检查模型目录是否存在
if [ ! -d "${MODEL_PATH}" ]; then
    echo "⚠️ 警告: 模型路径 ${MODEL_PATH} 不存在，尝试降级为在线或默认路径..."
fi

# 2. 启动 vLLM 推理服务 (后台运行)
echo ">> [1/3] 正在启动 vLLM 核心推理引擎 (GPU Mem: ${GPU_MEM}, Port: ${VLLM_PORT})..."
vllm serve "${MODEL_PATH}" \
    --host 0.0.0.0 \
    --port "${VLLM_PORT}" \
    --served-model-name "${SERVED_MODEL_NAME}" \
    --gpu-memory-utilization "${GPU_MEM}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --trust-remote-code \
    --enforce-eager &
VLLM_PID=$!

# 3. 等待 vLLM 服务探针就绪
echo ">> [2/3] 正在等待 vLLM 推理服务初始化 (http://127.0.0.1:${VLLM_PORT}/v1/models)..."
MAX_RETRIES=90
RETRY_COUNT=0
READY=0

while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    if curl -s "http://127.0.0.1:${VLLM_PORT}/v1/models" | grep -q "${SERVED_MODEL_NAME}"; then
        READY=1
        break
    fi
    RETRY_COUNT=$((RETRY_COUNT + 1))
    echo "   ⏳ vLLM 预热与显存编译中 (${RETRY_COUNT}/${MAX_RETRIES})..."
    sleep 3
done

if [ $READY -eq 1 ]; then
    echo "✅ [vLLM 就绪] 大模型服务已成功拉起并处于监听状态！"
else
    echo "⚠️ [警告] 超过等待时间，vLLM 未返回成功状态，尝试继续拉起主程序..."
fi

# 4. 捕获退出信号并优雅关闭
cleanup() {
    echo "🛑 接收到终止信号，正在退出 vLLM 服务..."
    kill -TERM "$VLLM_PID" 2>/dev/null || true
    wait "$VLLM_PID" 2>/dev/null || true
    exit 0
}

trap cleanup SIGTERM SIGINT SIGHUP

# 容器保持运行，守护 vLLM 后台进程，等待外部调用 start.sh
echo ">> [3/3] vLLM 守护进程运行中... (算法随时可通过 start.sh 拉起)"
wait "$VLLM_PID"
