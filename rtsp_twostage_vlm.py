#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于 Qwen3-VL 的 RTSP 视频流时序滑动窗口故障检测系统
融合了 test_twostage_inference.py 的统一两阶段检测逻辑（原版 Prompt 不变，采用视频理解模式）
"""

import os
import sys
import time
import json
import re
import queue
import threading
import argparse
import cv2
import torch
import numpy as np
from PIL import Image

try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

# ==================== 默认参数配置 ====================
DEFAULT_MODEL_PATH = "./models/Qwen3-VL-4B-Instruct"
WINDOW_SIZE = 4  # 窗口大小：4 帧
SAMPLE_INTERVAL = 1  # 采样间隔：1 秒 / 帧 (1 FPS)

# =============================================
# Stage 1 Prompt: 火花 & 脏污全局分类 (保持原版内容)
# =============================================
PROMPT_STAGE1 = (
    "你是受电弓故障诊断专家。请观察图像，判断：\n"
    "1. 火花(spk)：有无刺眼强光或电弧放电？\n"
    "2. 脏污(dirt)：画面是否影响了视野，极度模糊、失焦、或被水珠/雾气/污泥遮挡？\n"
    "请先给出一句话的推理分析(reasoning)，描述画面的清晰度，然后再输出JSON。\n"
    "示例 JSON:\n"
    '{"reasoning": "画面整体极度模糊，无法看清受电弓细节，判定为脏污。", "spk": 0, "dirt": 1}'
)

# =============================================
# Stage 2A Prompt: 火花 <box> 矩形框定位 (保持原版内容)
# =============================================
PROMPT_STAGE2_SPARK = (
    "你是一个受电弓和接触网故障诊断专家。\n"
    "图像中发生了大火花或电弧放电。请仔细观察并找出火花的位置。\n"
    "如果找到了火花，请输出一段描述，并在最后必须使用XML格式给出火花的坐标，"
    "格式为：<box>x_min, y_min, x_max, y_max</box>。\n"
    "如果没有找到，请直接回复 'Normal'。"
)

# =============================================
# Stage 2B Prompt: 悬挂异物 <point> 点定位 (保持原版内容)
# =============================================
PROMPT_STAGE2_FOREIGN = (
    "你是一个视觉定位专家"
    "观察受电弓碳滑板（最上方的弧形横梁）及其附近的接触网区域。\n"
    "找出不属于受电弓结构的悬挂异物（如编织袋碎片、棉纱线头、鸟窝、羽毛等），"
    "若有异物，输出中心点坐标：<point>x, y</point>（0-1000归一化，原点左上角）。\n"
    "金属部件（弓角、线夹、螺栓、卡扣、铰接关节等）不是异物\n"
    "若无异物，只回复 'Normal'。"
)


# ==================== RTSP 异步多线程拉流器 ====================
class RTSPStreamSampler:
    def __init__(self, source_url, sample_interval=1.0, is_file=False):
        self.source_url = source_url
        self.sample_interval = sample_interval
        self.is_file = is_file
        self.running = False
        self.cap = None
        self.frame_queue = queue.Queue(maxsize=30)
        self.thread = None
        self.last_sample_time = 0.0

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()
        return self

    def _open_stream(self):
        print(f"[RTSP拉流] 正在连接视频源: {self.source_url} ...", flush=True)
        cap = cv2.VideoCapture(self.source_url)
        if not self.is_file:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    def _worker(self):
        self.cap = self._open_stream()
        if not self.cap.isOpened():
            print(
                f"[错误] 无法连接到视频源: {self.source_url}",
                file=sys.stderr,
                flush=True,
            )
            self.running = False
            return

        fps = self.cap.get(cv2.CAP_PROP_FPS) or 25.0
        frame_delay = 1.0 / fps if self.is_file else 0.001

        print(
            f"[RTSP拉流] 视频源连接成功! 采样率: 每 {self.sample_interval} 秒抽取 1 帧",
            flush=True,
        )
        self.last_sample_time = time.time()
        frame_count = 0

        while self.running:
            if not self.cap.isOpened():
                print("[警告] 视频流断开，正在尝试重连...", flush=True)
                time.sleep(1.0)
                self.cap = self._open_stream()
                continue

            ret, frame = self.cap.read()
            if not ret or frame is None:
                if self.is_file:
                    print("[RTSP拉流] 本地视频文件读取完毕。", flush=True)
                    break
                else:
                    time.sleep(0.05)
                    continue

            now = time.time()
            if self.is_file:
                if frame_count % int(round(fps * self.sample_interval)) == 0:
                    self._push_frame(frame, now)
                frame_count += 1
                time.sleep(frame_delay)
            else:
                if (now - self.last_sample_time) >= self.sample_interval:
                    self._push_frame(frame, now)
                    self.last_sample_time = now

        if self.cap:
            self.cap.release()
        self.running = False

    def _push_frame(self, frame, timestamp):
        if not self.frame_queue.full():
            self.frame_queue.put((frame, timestamp))
        else:
            try:
                self.frame_queue.get_nowait()
                self.frame_queue.put((frame, timestamp))
            except queue.Empty:
                pass

    def get_frame(self, timeout=2.0):
        try:
            return self.frame_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def stop(self):
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)


# ==================== Qwen3-VL 模型推理 ====================
def load_vlm_model(model_path=DEFAULT_MODEL_PATH, gpu_mem_fraction=0.65):
    print("=" * 70, flush=True)
    print(f"正在加载 Qwen3-VL 视频理解大模型...", flush=True)
    print(f"模型路径: {model_path}", flush=True)

    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(gpu_mem_fraction, device=0)
        gpu_name = torch.cuda.get_device_name(0)
        print(
            f"GPU 设备: {gpu_name} (分配显存上限: {int(gpu_mem_fraction * 100)}%)",
            flush=True,
        )

    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

    t0 = time.time()
    processor = AutoProcessor.from_pretrained(model_path)

    # 视频分辨率上限，控制视觉 token、显存和推理延迟。
    if hasattr(processor, "video_processor") and processor.video_processor is not None:
        processor.video_processor.size = {
            "shortest_edge": 64 * 32 * 32,
            "longest_edge": int(960 * 720 * 2),
        }

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path,
        dtype=torch.float16,
        device_map="cuda:0",
        low_cpu_mem_usage=True,
    )
    model.eval()
    print(f"模型加载成功！初始化耗时: {time.time() - t0:.2f}s", flush=True)
    print("=" * 70, flush=True)
    return model, processor


def infer_video(model, processor, frames_bgr_list, prompt, max_tokens=128):
    """
    将 4 帧 BGR 图像作为视频送入 Qwen3-VL 推理
    """
    pil_frames = [
        Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in frames_bgr_list
    ]
    content = [
        {"type": "video", "video": pil_frames, "fps": 1.0},
        {"type": "text", "text": prompt},
    ]
    messages = [{"role": "user", "content": content}]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(
        text=[text],
        videos=[pil_frames],
        padding=True,
        return_tensors="pt",
    )
    inputs = {k: v.to("cuda") for k, v in inputs.items()}

    t0 = time.time()
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs, max_new_tokens=max_tokens, do_sample=False
        )
    latency_ms = (time.time() - t0) * 1000

    input_len = inputs["input_ids"].shape[1]
    new_ids = generated_ids[:, input_len:]
    output_text = processor.batch_decode(new_ids, skip_special_tokens=True)[0].strip()

    return output_text, latency_ms


# ==================== 结果保存 ====================
def save_event_frames(frames, parsed_json, save_dir, event_id):
    """保存 4 帧独立的高清事件截图 (每帧单独保存，保留原始分辨率与标定)"""
    annotated_frames = [f.copy() for f in frames]

    # 解析故障
    spk = parsed_json.get("spk", 0)
    dirt = parsed_json.get("dirt", 0)
    obj = parsed_json.get("foreign_object", 0)

    box_coords = parsed_json.get("box_coords")
    point_coords = parsed_json.get("point_coords")

    labels = []
    if spk == 1:
        labels.append("SPARK")
    if dirt == 1:
        labels.append("DIRT")
    if obj == 1:
        labels.append("FOREIGN_OBJ")
    status_str = " | ".join(labels) if labels else "DETECTED"

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
            cv2.putText(
                f,
                "Spark",
                (x1, max(y1 - 10, 0)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 0, 255),
                2,
            )

        # 2. 绘制 Point (Foreign Object)
        elif obj == 1 and point_coords:
            nx, ny = point_coords
            # 1. 映射到像素坐标 (对齐 test_point_foreign_object.py)
            px = int(nx / 1000.0 * w)
            py = int(ny / 1000.0 * h)

            # 2. 限制坐标在有效图像范围内
            px = max(0, min(w - 1, px))
            py = max(0, min(h - 1, py))

            # 3. 绘制同心圆与 Crosshair 准星十字线 (对齐 test_point_foreign_object.py)
            cv2.circle(f, (px, py), 15, (0, 0, 255), 3)
            cv2.circle(f, (px, py), 4, (0, 0, 255), -1)
            cv2.line(f, (px - 25, py), (px + 25, py), (0, 0, 255), 2)
            cv2.line(f, (px, py - 25), (px, py + 25), (0, 0, 255), 2)
            cv2.putText(
                f,
                f"Foreign Object Point ({px},{py})",
                (max(px - 100, 10), max(py - 30, 30)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
            )

        # 3. 绘制 Dirt (如果发生脏污)
        elif dirt == 1:
            cv2.putText(
                f,
                "WARNING: Dirt / Blur Detected",
                (50, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 165, 255),
                2,
            )

        # 4. 标注当前事件与帧序号水印
        cv2.putText(
            f,
            f"EVENT: {event_id} | Frame {idx}/{len(frames)} | {status_str}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

        img_path = os.path.join(save_dir, f"event_{event_id}_f{idx}.jpg")
        cv2.imwrite(img_path, f)

    print(
        f"  💾 🚨 事件 4 帧全量截图已保存至: {save_dir}/event_{event_id}_f[1-4].jpg",
        flush=True,
    )


# ==================== 主滑动窗口检测循环 ====================
def run_twostage_sliding_window(
    source,
    model_path=DEFAULT_MODEL_PATH,
    save_dir="./local_cache/rtsp_twostage_results",
):
    os.makedirs(save_dir, exist_ok=True)
    is_file = os.path.exists(source) or (
        not source.startswith("rtsp://") and not source.startswith("http://")
    )

    model, processor = load_vlm_model(model_path)
    sampler = RTSPStreamSampler(
        source, sample_interval=SAMPLE_INTERVAL, is_file=is_file
    ).start()

    print(
        "\n🚀 [系统就绪] 开始执行 4 帧滑动窗口 + 两阶段综合检测 (采用原版 Prompt + 视频推理)..."
    )
    print("-" * 70, flush=True)

    window_buffer = []
    timestamp_buffer = []
    window_count = 0

    try:
        while sampler.running or not sampler.frame_queue.empty():
            item = sampler.get_frame(timeout=3.0)
            if item is None:
                if not sampler.running:
                    break
                continue

            frame, ts = item
            window_buffer.append(frame)
            timestamp_buffer.append(ts)

            current_time_str = time.strftime("%H:%M:%S", time.localtime(ts))
            print(
                f"[{current_time_str}] 📥 采集成帧 ({len(window_buffer)}/{WINDOW_SIZE})",
                flush=True,
            )

            if len(window_buffer) == WINDOW_SIZE:
                window_count += 1
                print(
                    f"\n⚡ ===== [第 {window_count} 个时序窗口] 送入 Qwen3-VL 进行 Stage 1 全局诊断 ====="
                )

                # ================= Stage 1 =================
                s1_text, s1_ms = infer_video(
                    model, processor, window_buffer, PROMPT_STAGE1, max_tokens=128
                )

                try:
                    s1_res = json.loads(s1_text)
                except Exception:
                    m = re.search(r"\{[^}]+\}", s1_text)
                    s1_res = json.loads(m.group()) if m else {}

                spk = s1_res.get("spk", 0)
                dirt = s1_res.get("dirt", 0)
                s1_reasoning = s1_res.get("reasoning", "")
                print(f"  ├─ S1 耗时: {s1_ms:.0f} ms | 推理: {s1_reasoning}")

                # ================= Stage 2 =================
                s2_text, s2_ms = "", 0
                box_match = None
                point_match = None
                obj_detected = 0

                if dirt == 1:
                    s2_text = "Skipped (极度脏污短路保护)"
                elif spk == 1:
                    print(f"  ⚡ 触发 Stage 2: 火花 <box> 定位...")
                    s2_text, s2_ms = infer_video(
                        model,
                        processor,
                        window_buffer,
                        PROMPT_STAGE2_SPARK,
                        max_tokens=128,
                    )
                    box_pattern1 = (
                        r"<box>\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*</box>"
                    )
                    box_pattern2 = r"\((\d+)\s*,\s*(\d+)\)\s*,?\s*\((\d+)\s*,\s*(\d+)\)"
                    box_match = re.search(box_pattern1, s2_text)
                    if not box_match:
                        box_match = re.search(box_pattern2, s2_text)
                else:
                    print(f"  🔍 触发 Stage 2: 悬挂异物 <point> 定位...")
                    s2_text, s2_ms = infer_video(
                        model,
                        processor,
                        window_buffer,
                        PROMPT_STAGE2_FOREIGN,
                        max_tokens=256,
                    )
                    point_pattern1 = r"<point>\s*(\d+)\s*,\s*(\d+)\s*</point>"
                    point_pattern2 = r"\((\d+)\s*,\s*(\d+)\)"
                    point_match = re.search(point_pattern1, s2_text)
                    if not point_match:
                        point_match = re.search(point_pattern2, s2_text)
                    if point_match:
                        obj_detected = 1

                if s2_text:
                    s2_short = s2_text[:80] + "..." if len(s2_text) > 80 else s2_text
                    print(f"  ├─ S2 耗时: {s2_ms:.0f} ms | 输出: {s2_short}")

                # ================= 状态汇总与保存 =================
                flags = []
                if spk == 1:
                    flags.append("🔥 大火花")
                if dirt == 1:
                    flags.append("🌫️ 脏污")
                if box_match and spk == 1:
                    flags.append("🎯 火花定位(Box)")
                elif point_match and obj_detected == 1:
                    flags.append("🎯 悬挂异物(Point)")

                status_str = " | ".join(flags) if flags else "✅ 正常"
                print(f"  └─ 总体结论: {status_str}")

                if spk == 1 or dirt == 1 or obj_detected == 1:
                    parsed_result = {
                        "spk": spk,
                        "dirt": dirt,
                        "foreign_object": obj_detected,
                        "box_coords": list(map(int, box_match.groups()))
                        if box_match
                        else None,
                        "point_coords": list(map(int, point_match.groups()))
                        if point_match
                        else None,
                    }
                    save_event_frames(
                        window_buffer, parsed_result, save_dir, int(time.time())
                    )

                print("=" * 70, "\n", flush=True)

                window_buffer.clear()
                timestamp_buffer.clear()

    except KeyboardInterrupt:
        print("\n[退出] 接收到用户中断信号，正在退出...", flush=True)
    finally:
        sampler.stop()
        print("[完成] 检测流程已停止。", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Qwen3-VL 视频流时序滑动窗口 + 两阶段综合检测 (原版Prompt)"
    )
    parser.add_argument(
        "--source",
        type=str,
        default="./弓网典型视频/foreign_1.mp4",
        help="RTSP流地址或本地视频文件路径",
    )
    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument(
        "--save_dir", type=str, default="./local_cache/rtsp_twostage_results"
    )
    args = parser.parse_args()

    run_twostage_sliding_window(args.source, args.model_path, args.save_dir)
