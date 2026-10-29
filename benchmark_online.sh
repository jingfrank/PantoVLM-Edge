#!/bin/bash
# ==============================================================================
# 动车组受电弓故障监测系统 - 在线视频流性能基准评测自动化脚本 (方式二)
# benchmark_online.sh
# ==============================================================================

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$BASE_DIR"

mkdir -p ./benchmark_results
TS=$(date +"%Y%m%d_%H%M%S")
TEGRA_LOG="./benchmark_results/tegrastats_online_${TS}.log"
RSS_LOG="./benchmark_results/memory_rss_online_${TS}.csv"
SYS_LOG="./benchmark_results/system_runtime_${TS}.log"

echo "================================================================================"
echo " 🚀 启动【方式二：在线视频流监测实测】 (Jetson AGX Orin / 车载工况评测)"
echo " 📁 运行时性能日志保存目录: ./benchmark_results"
echo "================================================================================"

# 1. 启动 Jetson 硬件级资源监控 (tegrastats)
HAS_TEGRA=false
if command -v tegrastats &> /dev/null; then
    HAS_TEGRA=true
    echo "[1/3] 启动 Jetson 硬件层 tegrastats 资源采样器 (1000ms)..."
    sudo tegrastats --interval 1000 --logfile "$TEGRA_LOG" &
    TEGRA_PID=$!
else
    echo "[1/3] 非 Jetson 环境或未安装 tegrastats，跳过 tegrastats"
fi

# 2. 启动生产监测主程序 (start.sh)
echo "[2/3] 启动动车组弓网 AI 监测服务 (start.sh)..."
./start.sh > "$SYS_LOG" 2>&1 &
MAIN_PID=$!

echo "================================================================================"
echo "  AI 监测服务已在后台运行 (PID: $MAIN_PID)"
echo "  - 终端正在实时采样进程内存并监控运行状态..."
echo "  - 提示: 请让系统稳定推流运行 5~15 分钟以上以获得充分的评测样本。"
echo "  - 测试完成时，请在此终端按 [Ctrl + C] 正常结束评测。"
echo "================================================================================"

echo "Timestamp,RSS_MB" > "$RSS_LOG"

stop_and_summary() {
    echo -e "\n\n[3/3] 收到停止信号，正在平稳关闭 AI 识别引擎并生成评测汇总..."
    
    # 优雅关闭算法服务 (通过 close.py 发送 0x24 停止信令或 SIGINT)
    if [ -f "close.py" ]; then
        python3 close.py 2>/dev/null || true
    fi
    kill -INT $MAIN_PID 2>/dev/null || true

    # 等待算法主进程退出并保存落盘报告
    TIMEOUT=15
    while kill -0 $MAIN_PID 2>/dev/null && [ $TIMEOUT -gt 0 ]; do
        sleep 1
        TIMEOUT=$((TIMEOUT - 1))
    done

    if kill -0 $MAIN_PID 2>/dev/null; then
        kill -9 $MAIN_PID 2>/dev/null || true
    fi

    # 停止 tegrastats
    if [ "$HAS_TEGRA" = true ] && [ ! -z "$TEGRA_PID" ]; then
        sudo kill -9 $TEGRA_PID 2>/dev/null || true
    fi

    echo "--------------------------------------------------------------------------------"
    echo " 🏁 在线流实测硬件与资源消耗汇总看板"
    echo "--------------------------------------------------------------------------------"
    
    # 统计进程 RSS 峰值
    if [ -f "$RSS_LOG" ]; then
        MAX_RSS=$(tail -n +2 "$RSS_LOG" | awk -F',' '{print $2}' | sort -n | tail -1)
        if [ ! -z "$MAX_RSS" ]; then
            MAX_RSS_GB=$(awk "BEGIN {printf \"%.2f\", $MAX_RSS/1024}")
            echo "   • Python 进程物理内存峰值 (RSS)   : ${MAX_RSS} MB (${MAX_RSS_GB} GB)"
        fi
    fi

    # 统计 Jetson 硬件层整机内存峰值
    if [ "$HAS_TEGRA" = true ] && [ -f "$TEGRA_LOG" ]; then
        PEAK_RAM=$(grep "RAM " "$TEGRA_LOG" | awk '{print $2}' | cut -d'/' -f1 | sort -n | tail -1)
        TOTAL_RAM=$(grep "RAM " "$TEGRA_LOG" | head -1 | awk '{print $2}' | cut -d'/' -f2 | sed 's/MB//')
        if [ ! -z "$PEAK_RAM" ]; then
            PEAK_RAM_GB=$(awk "BEGIN {printf \"%.2f\", $PEAK_RAM/1024}")
            echo "   • Jetson 整机物理内存/显存占用峰值 : ${PEAK_RAM} MB (${PEAK_RAM_GB} GB) / ${TOTAL_RAM} MB"
        fi
        echo "   • tegrastats 硬件监控日志路径      : $TEGRA_LOG"
    fi

    LATEST_MD=$(ls -t ./benchmark_results/online_benchmark_*.md 2>/dev/null | head -1 || true)
    if [ ! -z "$LATEST_MD" ]; then
        echo "   • 在线流延迟与吞吐性能报告已落盘至 : $LATEST_MD"
        echo "--------------------------------------------------------------------------------"
        echo " 📋 核心评测指标预览 (取自生成的 Markdown 报告):"
        echo "--------------------------------------------------------------------------------"
        grep -E "平均单窗口|等效平均单帧|全链路端到端|在线流式处理|采样队列积压量" "$LATEST_MD" || true
    fi
    echo "================================================================================"
    echo "✅ 在线视频流性能测试完成！完整数据已存入 ./benchmark_results/"
    exit 0
}

trap stop_and_summary INT TERM

# 循环采样进程 RSS 物理内存 (每 2 秒一次)
while kill -0 $MAIN_PID 2>/dev/null; do
    MEM_KB=$(cat /proc/$MAIN_PID/status 2>/dev/null | grep VmRSS | awk '{print $2}')
    if [ ! -z "$MEM_KB" ]; then
        MEM_MB=$((MEM_KB / 1024))
        echo "$(date '+%H:%M:%S'),$MEM_MB" >> "$RSS_LOG"
    fi
    sleep 2
done

stop_and_summary
