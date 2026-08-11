#!/bin/bash
BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="${BASE_DIR}/all.log"

echo "$(date) [gongwang_monitor] 关闭服务 ==========" >> "$LOG_FILE"
docker exec gongwang_monitor bash -c "cd /app && python3 close.py $@" >> "$LOG_FILE" 2>&1
echo "[gongwang_monitor] AI 识别服务已干净停止" >> "$LOG_FILE"
