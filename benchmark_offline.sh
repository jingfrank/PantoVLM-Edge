#!/bin/bash
# ==============================================================================
# 动车组受电弓故障监测系统 - 本地视频性能基准评测自动化脚本 (方式一)
# benchmark_offline.sh
# ==============================================================================

set -e

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$BASE_DIR"

VIDEO_PATH="${1:-./弓网典型视频/foreign_1.mp4}"
MAX_SEC="${2:-}"

mkdir -p ./benchmark_results
TS=$(date +"%Y%m%d_%H%M%S")
TEGRA_LOG="./benchmark_results/tegrastats_offline_${TS}.log"

echo "================================================================================"
echo " 🚀 启动【方式一：本地视频压测】 (Jetson AGX Orin / TITAN RTX 统一评测)"
echo " 📹 测试视频源: ${VIDEO_PATH}"
echo "================================================================================"

# 检测是否在 Jetson 平台并具备 sudo tegrastats 权限
HAS_TEGRA=false
if command -v tegrastats &> /dev/null; then
    HAS_TEGRA=true
    echo "[1/3] 检测到 Jetson 硬件环境，启动底层 tegrastats 资源采样器 (500ms)..."
    sudo tegrastats --interval 500 --logfile "$TEGRA_LOG" &
    TEGRA_PID=$!
else
    echo "[1/3] 非 Jetson 环境或未安装 tegrastats，跳过 tegrastats 后台采样"
fi

cleanup() {
    if [ "$HAS_TEGRA" = true ] && [ ! -z "$TEGRA_PID" ]; then
        echo -e "\n[*] 正在停止 tegrastats 监控 (PID: $TEGRA_PID)..."
        sudo kill -9 $TEGRA_PID 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

echo "[2/3] 开始执行端到端视频多阶段推理压测..."
EXTRA_ARGS=""
if [ ! -z "$MAX_SEC" ]; then
    EXTRA_ARGS="--max_sec $MAX_SEC"
fi

python3 run_full_video_test.py --video "$VIDEO_PATH" --output_dir ./benchmark_results $EXTRA_ARGS

echo "[3/3] 压测完成，正在整理硬件资源开销..."
if [ "$HAS_TEGRA" = true ] && [ -f "$TEGRA_LOG" ]; then
    PEAK_RAM=$(grep "RAM " "$TEGRA_LOG" | awk '{print $2}' | cut -d'/' -f1 | sort -n | tail -1)
    TOTAL_RAM=$(grep "RAM " "$TEGRA_LOG" | head -1 | awk '{print $2}' | cut -d'/' -f2 | sed 's/MB//')
    echo "--------------------------------------------------------------------------------"
    echo " 📊 [Jetson AGX Orin 硬件层实测资源开销]"
    if [ ! -z "$PEAK_RAM" ]; then
        PEAK_RAM_GB=$(awk "BEGIN {printf \"%.2f\", $PEAK_RAM/1024}")
        echo "   • 整机物理内存/统一显存占用峰值 : ${PEAK_RAM} MB (${PEAK_RAM_GB} GB) / ${TOTAL_RAM} MB"
    fi
    echo "   • tegrastats 详细日志已留存至   : $TEGRA_LOG"
    echo "--------------------------------------------------------------------------------"
fi

echo "✅ 本地视频基准评测成功结束！报告与 CSV 请查看 ./benchmark_results/"
