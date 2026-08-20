#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于 Qwen3-VL 的视频流悬挂异物专项测试脚本
参考 rtsp_twostage_vlm.py 编写，使用结构化 JSON Prompt，结合时序与空间定位能力进行专门的异物验证。
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
from PIL import Image

try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

DEFAULT_MODEL_PATH = "./models/Qwen3-VL-4B-Thinking/"
WINDOW_SIZE = 4
SAMPLE_INTERVAL = 1.0

# =============================================
# 悬挂异物专项 Prompt (强制 JSON 输出 + 强制 CoT)
# =============================================
PROMPT_FOREIGN_OBJECT = (
    "你是一个专业的铁路受电弓与接触网视觉巡检专家。\n"
    "【任务目标】\n"
    "排查画面中受电弓上是否存在“悬挂异物”（如塑料袋、风筝、鸟巢、树枝、气球、防尘网等开集物体）。\n\n"
    "输出格式\n"
    "按照以下 JSON 格式输出最后的结果：\n\n"
    "{\n"
    '  "has_foreign_object": true/false,\n'
    '  "object_category": "异物的精细类别（如：蓝色塑料薄膜/鸟巢/树枝/风筝/未知编织物）。如果为 false，则填 null",\n'
    '  "description": "对异物的颜色、形态、大小以及缠绕状态的详细自然语言描述。如果为 false，则填 null",\n'
    '  "bounding_box": [ymin, xmin, ymax, xmax],\n'
    "}"
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

        print(f"[RTSP拉流] 连接成功! 采样间隔: {self.sample_interval} 秒", flush=True)
        self.last_sample_time = time.time()
        frame_count = 0

        while self.running:
            if not self.cap.isOpened():
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
                # 文件模式：按照间隔抽帧
                if frame_count % max(1, int(round(fps * self.sample_interval))) == 0:
                    self._push_frame(frame, now)
                frame_count += 1
                time.sleep(frame_delay)
            else:
                # RTSP流模式：按时间间隔采样
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


# ==================== 模型推理与图像保存 ====================
def load_vlm_model(model_path=DEFAULT_MODEL_PATH, gpu_mem_fraction=0.65):
    print("=" * 70, flush=True)
    print(f"正在加载 Qwen3-VL 思考模型...", flush=True)
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(gpu_mem_fraction, device=0)

    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

    processor = AutoProcessor.from_pretrained(model_path)
    if hasattr(processor, "video_processor") and processor.video_processor is not None:
        processor.video_processor.size = {
            "shortest_edge": 64 * 32 * 32,
            "longest_edge": int(960 * 720 * 2),
        }
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path, dtype=torch.float16, device_map="cuda:0", low_cpu_mem_usage=True
    )
    model.eval()
    return model, processor


def infer_video(model, processor, frames_bgr_list, prompt, max_tokens=1024):
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
        text=[text], videos=[pil_frames], padding=True, return_tensors="pt"
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


def save_event_frames(frames, parsed_json, save_dir, event_id):
    annotated_frames = [f.copy() for f in frames]
    has_obj = parsed_json.get("has_foreign_object", False)
    cat = parsed_json.get("object_category") or "Unknown"
    box = parsed_json.get("bounding_box")
    sev = parsed_json.get("severity_assessment") or "low"

    for idx, f in enumerate(annotated_frames, start=1):
        h, w = f.shape[:2]
        # Qwen3-VL 的 bounding box 格式解析: 通常是 [ymin, xmin, ymax, xmax] 归一化到 0-1000
        if has_obj and box and len(box) == 4 and isinstance(box[0], (int, float)):
            ny1, nx1, ny2, nx2 = box
            x1 = max(0, min(w, int(nx1 / 1000.0 * w)))
            y1 = max(0, min(h, int(ny1 / 1000.0 * h)))
            x2 = max(0, min(w, int(nx2 / 1000.0 * w)))
            y2 = max(0, min(h, int(ny2 / 1000.0 * h)))

            color = (0, 0, 255) if sev == "high" else (0, 165, 255)  # BGR
            cv2.rectangle(f, (x1, y1), (x2, y2), color, 3)
            # 添加底色文字标签
            label = f"{cat} ({sev})"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
            cv2.rectangle(f, (x1, max(y1 - 30, 0)), (x1 + tw, max(y1, 30)), color, -1)
            cv2.putText(
                f,
                label,
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 255, 255),
                2,
            )

        status_str = f"Obj: {cat}" if has_obj else "Normal"
        text_color = (0, 255, 0) if not has_obj else (0, 0, 255)
        cv2.putText(
            f,
            f"EVENT: {event_id} | Frame {idx}/{len(frames)} | {status_str}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            text_color,
            2,
            cv2.LINE_AA,
        )

        img_path = os.path.join(save_dir, f"foreign_event_{event_id}_f{idx}.jpg")
        cv2.imwrite(img_path, f)


# ==================== 主测试流程 ====================
def run_test(source, model_path, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    is_file = os.path.exists(source) or not source.startswith(("rtsp://", "http://"))

    model, processor = load_vlm_model(model_path)
    sampler = RTSPStreamSampler(
        source, sample_interval=SAMPLE_INTERVAL, is_file=is_file
    ).start()

    print("\n🚀 [系统就绪] 开始执行 悬挂异物 专项测试 (使用结构化 JSON Prompt)...")
    print("-" * 70, flush=True)

    window_buffer = []
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
            current_time_str = time.strftime("%H:%M:%S", time.localtime(ts))
            print(
                f"[{current_time_str}] 📥 采集成帧 ({len(window_buffer)}/{WINDOW_SIZE})",
                flush=True,
            )

            if len(window_buffer) == WINDOW_SIZE:
                window_count += 1
                print(
                    f"\n⚡ ===== [第 {window_count} 个时序窗口] Qwen3-VL 悬挂异物检测 ====="
                )

                s_text, s_ms = infer_video(
                    model,
                    processor,
                    window_buffer,
                    PROMPT_FOREIGN_OBJECT,
                    max_tokens=1024,
                )

                # 提取并打印 Thinking 过程
                think_match = re.search(r"<think>([\s\S]*?)</think>", s_text)
                think_content = think_match.group(1).strip() if think_match else ""

                # 安全解析 JSON
                try:
                    m = re.search(r"\{[\s\S]*\}", s_text)
                    s_res = json.loads(m.group()) if m else json.loads(s_text)
                except Exception as e:
                    print(f"  [JSON 解析错误]: {e}\n  [原始输出]: {s_text}")
                    s_res = {}

                print(f"  ├─ 推理耗时: {s_ms:.0f} ms")
                if think_content:
                    # 将较长的思考过程截断或简单打印
                    print(
                        f"  ├─ 内部深度思考 (Thinking): \n{think_content}\n  ├─ --------------------"
                    )

                print(
                    f"  ├─ 是否有异物 (Has Object): {s_res.get('has_foreign_object', False)}"
                )

                if s_res.get("has_foreign_object"):
                    print(f"  ├─ 异物类别: {s_res.get('object_category', '')}")
                    print(f"  ├─ 坐标定位: {s_res.get('bounding_box', [])}")
                    print(f"  ├─ 危险等级: {s_res.get('severity_assessment', '')}")

                    save_event_frames(window_buffer, s_res, save_dir, int(time.time()))
                    print(f"  └─ 🚨 检测到异物，已保存画框截图至: {save_dir}")
                else:
                    print(f"  └─ ✅ 正常。")

                print("=" * 70, "\n", flush=True)
                window_buffer.clear()

    except KeyboardInterrupt:
        print("\n[退出] 接收到用户中断信号，正在退出...", flush=True)
    finally:
        sampler.stop()
        print("[完成] 测试流程已停止。", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Qwen3-VL 悬挂异物专项测试")
    parser.add_argument(
        "--source", type=str, default="./弓网典型视频/foreign_1.mp4", help="视频源"
    )
    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument(
        "--save_dir", type=str, default="./local_cache/foreign_test_results"
    )
    args = parser.parse_args()

    run_test(args.source, args.model_path, args.save_dir)
