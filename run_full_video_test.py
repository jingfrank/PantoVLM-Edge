#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
完整视频端到端压测脚本 (run_full_video_test.py)
--------------------------------------------------------------------------------
100% 严格对齐 rtsp_twostage_vlm.py 的级联检测逻辑与事件截图保存：
- 先执行 Stage 1 (PROMPT_STAGE1)
- 级联触发：
  * 若 dirt == 1: 跳过 Stage 2
  * 若 spk == 1: 执行 Stage 2A 火花 <box> 定位 (PROMPT_STAGE2_SPARK)
  * 若 spk == 0 且 dirt == 0: 执行 Stage 2B 悬挂异物 <point> 定位 (PROMPT_STAGE2_FOREIGN)
- 故障检出时，自动绘制与保存 4 帧高清事件标定截图并生成 CSV 报告
"""

import os
import sys
import time
import csv
import datetime
import cv2
import argparse
import numpy as np
from vllm_client_engine import TwoStageVLLMEngine


def save_annotated_event_frames(frames, parsed_res, save_dir, event_id):
    """保存 4 帧独立的高清事件截图 (100% 对齐 rtsp_twostage_vlm.py save_event_frames)"""
    os.makedirs(save_dir, exist_ok=True)
    annotated_frames = [f.copy() for f in frames]

    spk = parsed_res.get("spk", 0)
    dirt = parsed_res.get("dirt", 0)
    obj = parsed_res.get("obj", 0)
    box_coords = parsed_res.get("box_coords")
    point_coords = parsed_res.get("foreign_point")

    saved_files = []

    for idx, f in enumerate(annotated_frames, start=1):
        h, w = f.shape[:2]

        # 1. 绘制 Box (Spark)
        if spk == 1 and box_coords:
            nx1, ny1, nx2, ny2 = box_coords
            x1 = max(0, min(w, int(nx1 / 1000.0 * w)))
            y1 = max(0, min(h, int(ny1 / 1000.0 * h)))
            x2 = max(0, min(w, int(nx2 / 1000.0 * w)))
            y2 = max(0, min(h, int(ny2 / 1000.0 * h)))
            cv2.rectangle(f, (x1, y1), (x2, y2), (0, 0, 255), 3)
            cv2.putText(f, "Spark", (x1, max(y1 - 10, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        # 2. 绘制 Point (Foreign Object)
        elif obj == 1 and point_coords:
            nx, ny = point_coords
            px = int(nx / 1000.0 * w)
            py = int(ny / 1000.0 * h)
            px = max(0, min(w - 1, px))
            py = max(0, min(h - 1, py))

            cv2.circle(f, (px, py), 15, (0, 0, 255), 3)
            cv2.circle(f, (px, py), 4, (0, 0, 255), -1)
            cv2.line(f, (px - 25, py), (px + 25, py), (0, 0, 255), 2)
            cv2.line(f, (px, py - 25), (px, py + 25), (0, 0, 255), 2)
            cv2.putText(f, f"Foreign Object Point ({px},{py})",
                        (max(px - 100, 10), max(py - 30, 30)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # 3. 绘制 Dirt
        elif dirt == 1:
            cv2.putText(f, "WARNING: Dirt / Blur Detected", (50, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 165, 255), 2)

        # 4. 标注状态水印
        cv2.putText(f, f"EVENT: {event_id} | Frame {idx}/{len(frames)} | {parsed_res.get('status_str', '')}",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)

        img_name = f"event_{event_id}_f{idx}.jpg"
        img_path = os.path.join(save_dir, img_name)
        cv2.imwrite(img_path, f)
        saved_files.append(img_name)

    return saved_files


def run_full_test(video_path, max_duration=None, output_dir="./test_results"):
    if not os.path.exists(video_path):
        print(f"[ERROR] 找不到视频文件: {video_path}")
        return

    video_name = os.path.splitext(os.path.basename(video_path))[0]
    now_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(output_dir, exist_ok=True)
    snapshot_dir = os.path.join(output_dir, f"events_{video_name}_{now_str}")

    csv_path = os.path.join(output_dir, f"result_{video_name}_{now_str}.csv")
    md_path = os.path.join(output_dir, f"report_{video_name}_{now_str}.md")

    print("=" * 95)
    print(" 🎬 Qwen3-VL (vLLM 加速) 完整视频端到端故障诊断测试 (100% 对齐 rtsp_twostage_vlm 级联逻辑)")
    print(f" 📹 视频源: {video_path}")
    print(f" 📁 结果保存目录: {output_dir}")
    print("=" * 95)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] 无法打开视频: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_duration_s = total_frames / fps

    print(f" -> 视频总时长: {int(video_duration_s//60):02d}:{int(video_duration_s%60):02d} ({video_duration_s:.1f}s), 总帧数: {total_frames}, 帧率: {fps:.1f} FPS\n")

    engine = TwoStageVLLMEngine()

    window_buffer = []
    window_count = 0
    fault_event_count = 0
    records = []
    total_latency_ms_list = []

    print(f"{'时间':^8} | {'窗口':^6} | {'S1耗时':^8} | {'S2耗时':^8} | {'总耗时':^8} | {'总体诊断结论':<24} | {'定位坐标':^16}")
    print("-" * 95)

    frame_idx = 0
    sample_stride = max(1, int(round(fps)))  # 严格每 1 秒抽取 1 帧
    start_wall_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        current_sec = frame_idx / fps
        if max_duration and current_sec > max_duration:
            break

        if frame_idx % sample_stride == 0:
            window_buffer.append(frame)

            if len(window_buffer) == 4:
                window_count += 1
                res = engine.infer_window(window_buffer, timestamp=current_sec)
                current_window_frames = [f for f in window_buffer]
                window_buffer.clear()

                total_latency_ms_list.append(res["total_latency_ms"])

                time_str = f"{int(current_sec//60):02d}:{int(current_sec%60):02d}"
                s1_time = f"{res['stage1_latency_ms']}ms"
                s2_time = f"{res['stage2_latency_ms']}ms" if res['stage2_latency_ms'] > 0 else "跳过"
                tot_time = f"{res['total_latency_ms']}ms"

                status_str = res["status_str"]
                has_fault = (res["spk"] == 1 or res["dirt"] == 1 or res["obj"] == 1)

                coord_str = "-"
                if res["box_coords"]:
                    coord_str = f"Box{tuple(res['box_coords'])}"
                elif res["foreign_point"]:
                    coord_str = f"Pt{tuple(res['foreign_point'])}"

                print(f"{time_str:^8} | #{window_count:<5} | {s1_time:^8} | {s2_time:^8} | {tot_time:^8} | {status_str:<24} | {coord_str:^16}")

                # 若触发故障，保存事件全量 4 帧截图
                saved_snapshots = []
                if has_fault:
                    fault_event_count += 1
                    event_id = f"win{window_count}_{time_str.replace(':', 'm')}s"
                    saved_snapshots = save_annotated_event_frames(
                        current_window_frames, res, snapshot_dir, event_id
                    )

                # 记录明细数据
                records.append({
                    "window_id": window_count,
                    "time_str": time_str,
                    "current_sec": round(current_sec, 2),
                    "s1_latency_ms": res["stage1_latency_ms"],
                    "s2_latency_ms": res["stage2_latency_ms"],
                    "total_latency_ms": res["total_latency_ms"],
                    "spk": res["spk"],
                    "dirt": res["dirt"],
                    "foreign_obj": res["obj"],
                    "box_coords": str(res["box_coords"]) if res["box_coords"] else "",
                    "point_coords": str(res["foreign_point"]) if res["foreign_point"] else "",
                    "status": status_str,
                    "snapshots": ",".join(saved_snapshots),
                    "reasoning": res["reasoning"],
                })

        frame_idx += 1

    cap.release()
    total_wall_time = time.time() - start_wall_time

    # 1. 写入 CSV 文件
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        fieldnames = [
            "window_id", "time_str", "current_sec", "s1_latency_ms", "s2_latency_ms",
            "total_latency_ms", "spk", "dirt", "foreign_obj", "box_coords", "point_coords",
            "status", "snapshots", "reasoning"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    # 2. 写入 Markdown 汇总报告
    avg_lat = np.mean(total_latency_ms_list) if total_latency_ms_list else 0.0
    med_lat = np.median(total_latency_ms_list) if total_latency_ms_list else 0.0

    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# 视频检测评估报告: `{video_name}.mp4`\n\n")
        f.write(f"- **测试时间**: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"- **检测逻辑**: `100% 对齐 rtsp_twostage_vlm.py 级联模式 (vLLM 加速)`\n")
        f.write(f"- **视频总时长**: {int(video_duration_s//60):02d}:{int(video_duration_s%60):02d} ({video_duration_s:.1f}s)\n")
        f.write(f"- **实际推理耗时**: {int(total_wall_time//60):02d}:{int(total_wall_time%60):02d} ({total_wall_time:.1f}s)\n")
        f.write(f"- **处理窗口总数**: {window_count} 个\n")
        f.write(f"- **平均单窗口耗时**: {avg_lat:.1f} ms (中位数: {med_lat:.1f} ms)\n")
        f.write(f"- **检出故障事件总数**: {fault_event_count} 次\n\n")
        f.write("## 故障检出明细表\n\n")
        f.write("| 窗口 # | 时间轴 | 耗时 | 总体结论 | 定位坐标 | 推理原因 | 事件截图 |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n")
        for r in records:
            if r["spk"] == 1 or r["dirt"] == 1 or r["foreign_obj"] == 1:
                coord = r["box_coords"] if r["box_coords"] else (r["point_coords"] if r["point_coords"] else "-")
                f.write(f"| #{r['window_id']} | {r['time_str']} | {r['total_latency_ms']}ms | `{r['status']}` | `{coord}` | {r['reasoning']} | `{r['snapshots']}` |\n")

    print("-" * 95)
    print("\n" + "=" * 95)
    print(" 🏁 完整视频测试统计与落盘汇总")
    print("=" * 95)
    print(f"• 测试视频: {video_path}")
    print(f"• 视频总时长: {int(video_duration_s//60):02d}:{int(video_duration_s%60):02d} | 实际评测耗时: {int(total_wall_time//60):02d}:{int(total_wall_time%60):02d}")
    print(f"• 评估窗口总数: {window_count} 个")
    print(f"• 平均单窗口耗时: {avg_lat:.1f} ms (中位数: {med_lat:.1f} ms)")
    print(f"• 检出故障事件总数: {fault_event_count} 次")
    print("-" * 95)
    print(f"📄 CSV 详细数据已保存至: {csv_path}")
    print(f"📝 Markdown 报告已保存至: {md_path}")
    if fault_event_count > 0:
        print(f"📸 故障 4 帧全量截图目录: {snapshot_dir}")
    print("=" * 95)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Full Video Test with vLLM (Alinged with rtsp_twostage_vlm.py)")
    parser.add_argument("--video", type=str, default="./弓网典型视频/foreign_1.mp4", help="Video path")
    parser.add_argument("--max_sec", type=float, default=None, help="Max duration in seconds")
    parser.add_argument("--output_dir", type=str, default="./test_results", help="Output directory")
    args = parser.parse_args()

    run_full_test(args.video, max_duration=args.max_sec, output_dir=args.output_dir)
