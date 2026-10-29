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
import psutil
try:
    import torch
except ImportError:
    torch = None

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
    s1_latency_ms_list = []
    s2_latency_ms_list = []

    process = psutil.Process(os.getpid())
    initial_ram_mb = process.memory_info().rss / (1024 * 1024)
    peak_ram_mb = initial_ram_mb

    if torch and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        initial_vram_mb = torch.cuda.memory_allocated() / (1024 * 1024)
    else:
        initial_vram_mb = 0.0

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

                tot_ms = res["total_latency_ms"]
                s1_ms = res["stage1_latency_ms"]
                s2_ms = res["stage2_latency_ms"]

                total_latency_ms_list.append(tot_ms)
                s1_latency_ms_list.append(s1_ms)
                if s2_ms > 0:
                    s2_latency_ms_list.append(s2_ms)

                cur_ram = process.memory_info().rss / (1024 * 1024)
                if cur_ram > peak_ram_mb:
                    peak_ram_mb = cur_ram

                time_str = f"{int(current_sec//60):02d}:{int(current_sec%60):02d}"
                s1_time = f"{s1_ms}ms"
                s2_time = f"{s2_ms}ms" if s2_ms > 0 else "跳过"
                tot_time = f"{tot_ms}ms"

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
                    "s1_latency_ms": s1_ms,
                    "s2_latency_ms": s2_ms,
                    "total_latency_ms": tot_ms,
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

    # 2. 统计计算
    tot_arr = np.array(total_latency_ms_list) if total_latency_ms_list else np.array([0.0])
    s1_arr = np.array(s1_latency_ms_list) if s1_latency_ms_list else np.array([0.0])
    s2_arr = np.array(s2_latency_ms_list) if s2_latency_ms_list else np.array([0.0])

    avg_tot = float(np.mean(tot_arr))
    p50_tot = float(np.percentile(tot_arr, 50))
    p90_tot = float(np.percentile(tot_arr, 90))
    p99_tot = float(np.percentile(tot_arr, 99))
    std_tot = float(np.std(tot_arr))
    single_frame_lat = avg_tot / 4.0

    avg_s1 = float(np.mean(s1_arr))
    p50_s1 = float(np.percentile(s1_arr, 50))
    p99_s1 = float(np.percentile(s1_arr, 99))

    avg_s2 = float(np.mean(s2_arr)) if len(s2_latency_ms_list) > 0 else 0.0
    p50_s2 = float(np.percentile(s2_arr, 50)) if len(s2_latency_ms_list) > 0 else 0.0
    p99_s2 = float(np.percentile(s2_arr, 99)) if len(s2_latency_ms_list) > 0 else 0.0

    total_eval_frames = window_count * 4
    offline_fps = total_eval_frames / max(0.001, total_wall_time)
    window_qps = window_count / max(0.001, total_wall_time)

    peak_vram_mb = 0.0
    if torch and torch.cuda.is_available():
        peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

    # 3. 写入 Markdown 汇总报告 (包含系统评测指标看板)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# 动车组受电弓故障监测系统 - 离线视频基准评测报告\n\n")
        f.write(f"- **测试视频文件**: `{video_path}`\n")
        f.write(f"- **测试执行时间**: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"- **视频总时长**: {int(video_duration_s//60):02d}:{int(video_duration_s%60):02d} ({video_duration_s:.1f}s), 总帧数: {total_frames}\n")
        f.write(f"- **实际压测耗时**: {int(total_wall_time//60):02d}:{int(total_wall_time%60):02d} ({total_wall_time:.1f}s)\n")
        f.write(f"- **处理窗口总数**: {window_count} 个 (等效评测帧数: {total_eval_frames} 帧)\n")
        f.write(f"- **检出故障事件总数**: {fault_event_count} 次\n\n")

        f.write("## 1. 核心系统评测指标汇总表 (System Benchmark Scorecard)\n\n")
        f.write("| 评测维度 | 指标项 | 评测实测值 | 性能判定与说明 |\n")
        f.write("| :--- | :--- | :--- | :--- |\n")
        f.write(f"| **处理延迟** | **平均单窗口耗时 (Mean)** | **{avg_tot:.2f} ms** | 4 帧动静级联诊断整体平均时延 |\n")
        f.write(f"| | **等效平均单帧耗时** | **{single_frame_lat:.2f} ms / 帧** | 均摊至每张输入图像的处理时延 |\n")
        f.write(f"| | **中位数耗时 (P50)** | **{p50_tot:.2f} ms** | 50% 样本处于该时延以内 |\n")
        f.write(f"| | **90分位耗时 (P90)** | **{p90_tot:.2f} ms** | 90% 样本处于该时延以内 |\n")
        f.write(f"| | **99分位耗时 (P99)** | **{p99_tot:.2f} ms** | 长尾极限时延，衡量系统抗抖动能力 |\n")
        f.write(f"| | **时延标准差 (Jitter Std)** | **{std_tot:.2f} ms** | 时延稳定性评估 |\n")
        f.write(f"| | Stage 1 (时序分类) 平均耗时 | {avg_s1:.2f} ms (P99: {p99_s1:.2f} ms) | 4 帧视频输入诊断火花与脏污 |\n")
        f.write(f"| | Stage 2 (细粒度定位) 平均耗时 | {avg_s2:.2f} ms (P99: {p99_s2:.2f} ms) | 触发异物高精定位时的单帧耗时 |\n")
        f.write(f"| **吞吐能力** | **纯算力极限吞吐率 (FPS)** | **{offline_fps:.2f} FPS** | 离线满载无间隔连续推帧处理速率 |\n")
        f.write(f"| | **窗口吞吐率 (QPS)** | **{window_qps:.2f} Windows/s** | 每秒可完整完成的时序窗口诊断数 |\n")
        f.write(f"| **资源开销** | **主机物理内存峰值 (RAM RSS)** | **{peak_ram_mb:.2f} MB ({peak_ram_mb/1024:.2f} GB)** | 进程常驻物理内存峰值 |\n")
        if peak_vram_mb > 0:
            f.write(f"| | **GPU 显存活跃峰值 (VRAM Alloc)** | **{peak_vram_mb:.2f} MB ({peak_vram_mb/1024:.2f} GB)** | PyTorch / CUDA 实际张量分配峰值 |\n")
        f.write("\n---\n\n")

        f.write("## 2. 故障事件检出明细表\n\n")
        f.write("| 窗口 # | 时间轴 | 耗时 | 总体结论 | 定位坐标 | 推理原因 | 事件截图 |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n")
        for r in records:
            if r["spk"] == 1 or r["dirt"] == 1 or r["foreign_obj"] == 1:
                coord = r["box_coords"] if r["box_coords"] else (r["point_coords"] if r["point_coords"] else "-")
                f.write(f"| #{r['window_id']} | {r['time_str']} | {r['total_latency_ms']}ms | `{r['status']}` | `{coord}` | {r['reasoning']} | `{r['snapshots']}` |\n")

    print("-" * 95)
    print("\n" + "=" * 95)
    print(" 🏁 完整视频评测统计与指标看板 (System Evaluation Scorecard)")
    print("=" * 95)
    print(f"• 测试视频: {video_path}")
    print(f"• 评估窗口数: {window_count} 个 (等效帧数: {total_eval_frames}) | 实际用时: {total_wall_time:.2f}s")
    print(f"-------------------------------------------------------------------------------------------")
    print(f" [1] 处理延迟指标 (Latency)")
    print(f"     - 平均单窗口处理耗时 (Mean) : {avg_tot:.2f} ms")
    print(f"     - 等效单帧处理耗时 (Per-frame): {single_frame_lat:.2f} ms / 帧")
    print(f"     - 中位数 (P50) / 90分位 (P90): {p50_tot:.2f} ms / {p90_tot:.2f} ms")
    print(f"     - 99分位长尾时延 (P99)      : {p99_tot:.2f} ms (标准差: {std_tot:.2f} ms)")
    print(f"     - Stage 1 (时序分类) 耗时   : 均值 {avg_s1:.1f} ms | P99 {p99_s1:.1f} ms")
    print(f"     - Stage 2 (细粒度定位) 耗时 : 均值 {avg_s2:.1f} ms | P99 {p99_s2:.1f} ms")
    print(f"-------------------------------------------------------------------------------------------")
    print(f" [2] 吞吐能力指标 (Throughput)")
    print(f"     - 纯算力离线极限吞吐 (FPS)  : {offline_fps:.2f} FPS")
    print(f"     - 时序窗口诊断能力 (QPS)    : {window_qps:.2f} Windows/s")
    print(f"-------------------------------------------------------------------------------------------")
    print(f" [3] 内存与显存占用 (Resources)")
    print(f"     - 主机物理内存峰值 (RSS)   : {peak_ram_mb:.2f} MB ({peak_ram_mb/1024:.2f} GB)")
    if peak_vram_mb > 0:
        print(f"     - GPU 显存活跃峰值 (VRAM)   : {peak_vram_mb:.2f} MB ({peak_vram_mb/1024:.2f} GB)")
    print(f"-------------------------------------------------------------------------------------------")
    print(f"• 故障事件检出次数: {fault_event_count} 次")
    print(f"📄 CSV 明细数据已保存至: {csv_path}")
    print(f"📝 Markdown 评测报告已保存至: {md_path}")
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
