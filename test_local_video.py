#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
弓网典型视频本地离线测试脚本 (test_local_video.py)
用于使用 ./弓网典型视频 目录下的测试视频验证 Qwen3-VL 4B TensorRT 引擎
"""

import os
import sys
import argparse
import cv2
import time
import yaml

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from trt_qwen3_vl import TensorRTQwen3VLEngine

FAULT_CN_MAP = {
    "spk": "大火花",
    "str": "受电弓结构异常",
    "dirt": "画面脏污",
    "tilt": "受电弓倾斜",
    "obj": "悬挂异物",
    "flt": "异常飘弓"
}

def run_video_test(video_path, config_path="config_prod.yaml"):
    full_video_path = os.path.abspath(video_path)
    if not os.path.exists(full_video_path):
        print(f"[ERROR] 找不到指定的视频文件: {full_video_path}")
        return

    config_full_path = os.path.abspath(config_path)
    with open(config_full_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    algo_cfg = cfg["base_algorithm_config"]
    print(f"\n=======================================================")
    print(f"[Qwen3-VL 4B 本地视频测试] 加载配置: {config_full_path}")
    print(f"[Qwen3-VL 4B 本地视频测试] 目标视频: {full_video_path}")
    print(f"=======================================================\n")

    engine = TensorRTQwen3VLEngine(algo_cfg)

    cap = cv2.VideoCapture(full_video_path)
    if not cap.isOpened():
        print(f"[ERROR] cv2.VideoCapture 无法打开视频文件: {full_video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[TEST] 视频成功打开 -> 帧率: {fps:.1f} FPS | 总帧数: {total_frames}\n")

    frame_idx = 0
    fault_summary = {v: 0 for v in FAULT_CN_MAP.values()}

    t_start = time.time()
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            frame_idx += 1
            t0 = time.time()
            res = engine.infer_frame(frame)
            latency_ms = (time.time() - t0) * 1000

            detected = [FAULT_CN_MAP[k] for k, v in res.items() if v == 1]
            if detected:
                for d in detected:
                    fault_summary[d] += 1
                
                # 自动保存告警快照至 local_cache/Algo-result/月-日/ 目录
                month_day = time.strftime("%m-%d")
                save_dir = os.path.join(BASE_DIR, "local_cache", "Algo-result", month_day)
                os.makedirs(save_dir, exist_ok=True)
                
                ts_str = time.strftime("%H%M%S")
                fault_tag = "_".join(detected)
                video_name = os.path.splitext(os.path.basename(video_path))[0]
                pic_filename = f"{ts_str}-frame{frame_idx:04d}-{video_name}-{fault_tag}.jpg"
                pic_path = os.path.join(save_dir, pic_filename)

                # 绘制 3+2 水印并落盘
                from ai_detector import draw_osd_watermark
                watermarked_frame = draw_osd_watermark(frame, train_id="CR400BF", car_number="06", speed=0, camera_id=0)
                cv2.imwrite(pic_path, watermarked_frame)

                print(f"[Frame {frame_idx:04d}/{total_frames:04d}] [{latency_ms:5.1f}ms] 🚨 触发故障: {', '.join(detected)}")
                print(f"  └─ 📸 告警快照已保存至: {pic_path}")
            else:
                if frame_idx % 30 == 0 or frame_idx == 1:
                    print(f"[Frame {frame_idx:04d}/{total_frames:04d}] [{latency_ms:5.1f}ms] 图像正常")


    except KeyboardInterrupt:
        print("\n[TEST] 用户手动中断测试。")

    t_total = time.time() - t_start
    cap.release()

    print(f"\n=======================================================")
    print(f"[测试完成总结]")
    print(f"  ├─ 处理总帧数: {frame_idx} / {total_frames}")
    print(f"  ├─ 总消耗耗时: {t_total:.2f} 秒")
    print(f"  ├─ 平均处理帧率: {frame_idx / t_total:.2f} FPS" if t_total > 0 else "")
    print(f"  └─ 故障统计结果: {fault_summary}")
    print(f"=======================================================\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="弓网典型视频 Qwen3-VL 4B 离线测试工具")
    parser.add_argument("--video", default="./弓网典型视频/spark_1.mp4", help="测试视频文件路径")
    parser.add_argument("--config", default="config_prod.yaml", help="配置文件路径")
    args = parser.parse_args()
    run_video_test(args.video, args.config)
