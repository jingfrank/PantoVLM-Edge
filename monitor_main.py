#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
动车组弓网监测子系统 - AI 监测服务主入口 (monitor_main.py)
整合：
1. YOLOv11 受电弓 AI 实时故障识别引擎 (ai_detector.py)
2. 车载 UDP 8080 端口 0xFF 0x20 标准故障告警上报
3. PHM 心跳上报 + 故障结果上报 (算法 ID 10008, phm_service.py)
"""

import os
import sys
import time
import yaml
import socket
import signal
import threading

src_dir = os.path.dirname(os.path.abspath(__file__))
if os.path.exists(os.path.join(src_dir, "config_prod.yaml")) or os.path.exists(os.path.join(src_dir, "AppConfig.xlsx")):
    app_dir = src_dir
else:
    app_dir = os.path.dirname(src_dir)
sys.path.insert(0, app_dir)
sys.path.insert(0, src_dir)

from ai_detector import PantographFaultDetector
from phm_service import create_monitor_phm_reporter
from ftp_uploader import init_ftp_uploader

stop_event = threading.Event()


def signal_handler(signum, frame):
    print("\n[监测服务] 收到终止信号，正在安全停止 AI 故障识别引擎...")
    stop_event.set()


def load_config():
    candidates = [
        os.path.join(app_dir, "config", "config_prod.yaml"),
        os.path.join(app_dir, "config_prod.yaml"),
        os.path.join(app_dir, "config", "config_dev.yaml"),
        os.path.join(app_dir, "config_dev.yaml"),
    ]
    default_cfg_path = next((p for p in candidates if os.path.exists(p)), candidates[0])
    config_file = sys.argv[1] if len(sys.argv) > 1 and os.path.exists(sys.argv[1]) else default_cfg_path

    if not os.path.exists(config_file):
        print(f"[CRITICAL] 无法找到配置文件: {config_file}", file=sys.stderr)
        sys.exit(1)

    with open(config_file, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f), config_file


def main():
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    global_cfg, config_file = load_config()

    print("=" * 80)
    print(" 动车组弓网监测子系统 - AI 监测服务启动 (Monitor) ")
    print(f" 载入配置文件: {config_file}")
    print("=" * 80)

    udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    udp_sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)

    init_ftp_uploader(global_cfg.get("ftp_config", {}))

    phm_reporter = None
    ai_threads = []
    detectors = []
    vs_cfg = global_cfg.get("video_stream_config", {})
    streams = vs_cfg.get("streams", [])

    algo_cfg = global_cfg.get("base_algorithm_config", {})
    engine_backend = algo_cfg.get("engine_backend", "vllm")
    if engine_backend == "vllm":
        from vllm_client_engine import TwoStageVLLMEngine
        shared_vlm_engine = TwoStageVLLMEngine(algo_cfg)
        print("[Engine] 🚀 正在使用 vLLM 高性能微服务推理后端")
    else:
        from twostage_vlm_engine import TwoStageVLMEngine
        shared_vlm_engine = TwoStageVLMEngine(algo_cfg)
        print("[Engine] 🐢 正在使用 Transformers 本地加载推理后端")

    for stream_cfg in streams:
        detector = PantographFaultDetector(global_cfg, stream_cfg, udp_sock, stop_event, vlm_engine=shared_vlm_engine)
        detectors.append(detector)
        t = threading.Thread(
            target=detector.run,
            daemon=True,
            name=f"AI-Detector-CAR{detector.car_number}-CAM{detector.camera_id}"
        )
        ai_threads.append(t)
        t.start()
        print(f"[AI-识别引擎已拉起] 流地址: {stream_cfg.get('rtsp_url')} -> 算法结果路径: {stream_cfg.get('result_base_dir')}")

    if not ai_threads:
        print("[WARNING] 未配置任何 video_stream_config.streams，监测服务无工作流。")

    phm_reporter = create_monitor_phm_reporter(global_cfg, detectors, package_dir=app_dir)
    if phm_reporter:
        phm_reporter.start()

    print("\n✅ [监测服务] AI 故障识别引擎与 PHM 心跳/故障上报已成功运行！")

    try:
        while not stop_event.is_set():
            time.sleep(0.5)
    finally:
        print("\n[监测服务] 正在关闭 AI 识别引擎...")
        stop_event.set()
        if phm_reporter:
            phm_reporter.stop()
        udp_sock.close()
        print("[监测服务] 安全退出完成。")


if __name__ == "__main__":
    main()
