#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于 Qwen3-VL 的 RTSP 视频流时序滑动窗口故障检测系统 (集成 Visual Grounding 点定位 Point)
(rtsp_sliding_window_vlm.py)

设计规格：
1. 采样频率：每秒采样 1 帧 (1 FPS)
2. 窗口大小 (Window Size)：4 帧 (覆盖 4 秒连续时序画面)
3. 滑动步长 (Stride)：4 帧 (步长等于窗口大小，无重复帧，每 4 秒触发一次时序分析)
4. 多帧时序分析与视觉定位：
   - 【大火花 (spark)】：受电弓碳滑板与接触网间剧烈燃弧、拉弧放电分析
   - 【悬挂异物 (foreign_object)】：
     * 引入 Visual Grounding 中心点定位约束：<point>x, y</point> (0-1000 归一化)
     * if foreign object does not exist in a frame, skip that frame
     * 幻觉过滤：未定位到目标中心点时不予告警，杜绝模型幻觉误报
   - 【镜头脏污 (dirt)】：镜头表面水滴、油膜、泥斑遮挡或严重污损
5. 架构特性：
   - 多线程解耦：独立后台拉流线程持续清空 RTSP 缓冲，杜绝视频流时延堆积
   - 结构化 JSON 约束输出 + 目标点自动提取与图像可视化标定
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
SAMPLE_INTERVAL = 1.0  # 采样间隔：1 秒 / 帧 (1 FPS)

# 专为 3 项故障定制的 Visual Grounding 时序分析 Prompt (Point 点定位)
TEMPORAL_FAULT_PROMPT = (
    "这是动车列车车顶摄像头拍摄的受电弓视频图像\n"
    "输入是由连续4秒（每秒1帧，共4帧）构成的时序视频片段。请仔细排查以下3类故障：\n\n"
    "1. 【大火花 (spark)】：判断受电弓碳滑板与上方接触网导线接触点是否存在明显的燃弧、大火花放电（注意区分正常金属反光与真实火花放电）。(0/1)\n"
    "2. 【镜头脏污 (dirt)】：判断摄像机镜头表面是否有明显的水渍、污泥、油斑遮挡或严重污损模糊。(0/1)\n"
    "3. 【悬挂异物 (foreign_object)】：受电弓碳滑板（最上方的弧形横梁）上高度怀疑有附着异物（塑料薄膜、编织袋、风筝线、鸟巢、树枝等）。\n"
    "detect and locate that foreign object, output form：frame <序号> <point>x, y</point>（x, y 坐标均在 0 到 1000 之间）；若没有外来异物，foreign_point 必须填 null。\n\n"
    "if foreign object does not exist in a frame, skip that frame。\n"
    "请严格按以下 JSON 格式输出：\n"
    "{\n"
    '  "spark": 0,\n'
    '  "dirt": 0,\n'
    '  "foreign_object": 1,\n'
    '  "foreign_point": "frame 2 <point>x, y</point>",\n'
    '  "details": {\n'
    '    "spark_desc": "描述火花状态",\n'
    '    "dirt_desc": "描述镜头清晰度",\n'
    '    "foreign_desc": "描述异物特征与定位位置"\n'
    "  },\n"
    '  "summary": "正常/大火花/悬挂异物/镜头脏污"\n'
    "}"
)


# ==================== RTSP 异步多线程拉流器 ====================
class RTSPStreamSampler:
    """
    RTSP 异步拉流与定频采样器
    功能：
    1. 独立线程循环 grab() 最新视频帧，彻底解决 OpenCV 读 RTSP 造成的缓冲区滞后问题
    2. 按照指定采样周期 (默认 1 秒) 精准抽取 1 帧存入采样队列
    """

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
            # RTSP 缓冲区优化参数
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
                # 本地视频文件模式：按帧序号精准抽帧
                if frame_count % int(round(fps * self.sample_interval)) == 0:
                    self._push_frame(frame, now)
                frame_count += 1
                time.sleep(frame_delay)
            else:
                # RTSP 实时流模式：按时钟间隔抽样
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


# ==================== Qwen3-VL 模型加载与推理 ====================
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
    
    # 🎯 性能优化：配置视频分辨率上限（720P 均衡档）
    # 避免默认配置将 2048x1536 视频超采样至 4096x3072（产生 2.4 万 Tokens、5.4s 延迟）
    # 设置后 4 帧视频等效约为 1344x960（~2500 Tokens），单次推理降至 ~500ms
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


def infer_window_frame_list(
    model, processor, frames_bgr_list, prompt=TEMPORAL_FAULT_PROMPT, max_tokens=384
):
    """
    将 4 帧 BGR 图像组装成 frame_list 喂入 Qwen3-VL 进行时序分析与视觉定位 (Point)
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
        text=[text], videos=[pil_frames], padding=True, return_tensors="pt"
    )
    inputs = {k: v.to("cuda") for k, v in inputs.items()}

    t0 = time.time()
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,  # 贪婪解码保证工业判定确定性
        )
    latency_ms = (time.time() - t0) * 1000

    input_len = inputs["input_ids"].shape[1]
    new_ids = generated_ids[:, input_len:]
    output_text = processor.batch_decode(new_ids, skip_special_tokens=True)[0].strip()

    # 尝试将模型输出解析为结构化 JSON
    parsed_json = None
    try:
        clean_text = output_text
        if "```json" in clean_text:
            clean_text = clean_text.split("```json")[1].split("```")[0].strip()
        elif "```" in clean_text:
            clean_text = clean_text.split("```")[1].split("```")[0].strip()

        json_match = re.search(r"\{[\s\S]*\}", clean_text)
        if json_match:
            parsed_json = json.loads(json_match.group(0))
        else:
            parsed_json = json.loads(clean_text)
    except Exception:
        pass

    if parsed_json is None:
        parsed_json = {
            "spark": 0,
            "dirt": 0,
            "foreign_object": 0,
            "foreign_point": None,
            "details": {},
            "summary": "解析失败",
        }

    # ==================== 视觉定位坐标后处理与幻觉校验 (Point) ====================
    search_target = str(parsed_json.get("foreign_point") or "") + " " + output_text
    point_match = re.search(r"<point>\s*(\d+)\s*,\s*(\d+)\s*</point>", search_target)

    # 提取帧序号 frame_num (如 frame 2, 默认为 4)
    frame_num = 4
    fm = re.search(r"(?i)frame\s*(\d)", search_target)
    if fm:
        try:
            fn = int(fm.group(1))
            if 1 <= fn <= 4:
                frame_num = fn
        except:
            pass

    if point_match:
        nx = int(point_match.group(1))
        ny = int(point_match.group(2))
        parsed_json["foreign_object"] = 1
        parsed_json["foreign_point_coords"] = (nx, ny)
        parsed_json["frame_num"] = frame_num
        parsed_json["foreign_point"] = f"frame {frame_num} <point>{nx}, {ny}</point>"
    else:
        # 兼容括号 (x, y) 格式
        paren_match = re.search(r"\((\d+)\s*,\s*(\d+)\)", search_target)
        if paren_match:
            nx = int(paren_match.group(1))
            ny = int(paren_match.group(2))
            parsed_json["foreign_object"] = 1
            parsed_json["foreign_point_coords"] = (nx, ny)
            parsed_json["frame_num"] = frame_num
            parsed_json["foreign_point"] = (
                f"frame {frame_num} <point>{nx}, {ny}</point>"
            )
        else:
            # 核心幻觉抑制：如果模型回答了 foreign_object=1 但未能给出合法的定位点坐标，自动降级为 0
            if parsed_json.get("foreign_object") == 1:
                print(
                    "[Visual Grounding 拦截] ⚠️ 模型标记了异物但未给出有效 <point> 坐标定位，判定为部件结构幻觉，已自动滤除。"
                )
                parsed_json["foreign_object"] = 0
            parsed_json["foreign_point_coords"] = None
            parsed_json["frame_num"] = None
            parsed_json["foreign_point"] = None

    return output_text, parsed_json, latency_ms


# ==================== 主滑动窗口检测循环 ====================
def run_sliding_window_detection(
    source, model_path=DEFAULT_MODEL_PATH, save_dir="./events"
):
    os.makedirs(save_dir, exist_ok=True)
    is_file = os.path.exists(source) or (
        not source.startswith("rtsp://") and not source.startswith("http://")
    )

    # 1. 载入 Qwen3-VL 模型
    model, processor = load_vlm_model(model_path)

    # 2. 启动异步定频拉流器 (1 秒抽 1 帧)
    sampler = RTSPStreamSampler(
        source, sample_interval=SAMPLE_INTERVAL, is_file=is_file
    ).start()

    print(
        "\n🚀 [系统就绪] 开始执行 4 帧滑动窗口时序监测 (集成 Visual Grounding Point 定位)..."
    )
    print(
        f"📌 监控目标: 1. 大火花(spark) | 2. 悬挂异物(foreign_object) | 3. 镜头脏污(dirt)"
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

            # 当填满 4 帧窗口时 (Stride = 4，无重叠)，触发一次 Qwen3-VL 视频理解
            if len(window_buffer) == WINDOW_SIZE:
                window_count += 1
                print(
                    f"\n⚡ ===== [第 {window_count} 个时序窗口 (4秒跨度)] 送入 Qwen3-VL 进行多帧理解与视觉定位 (Point) ====="
                )

                raw_output, parsed_json, latency_ms = infer_window_frame_list(
                    model, processor, window_buffer, prompt=TEMPORAL_FAULT_PROMPT
                )

                # 打印诊断报告
                print(f"⏱️ 推理耗时: {latency_ms:.1f} ms")
                print("📊 诊断结果:")
                spk = parsed_json.get("spark", 0)
                obj = parsed_json.get("foreign_object", 0)
                dirt = parsed_json.get("dirt", 0)
                summary = parsed_json.get("summary", "未知")
                details = parsed_json.get("details", {})
                point_coords = parsed_json.get("foreign_point_coords")

                spk_tag = "🔴【发现大火花】" if spk else "🟢 无火花"
                dirt_tag = "🟡【镜头脏污遮挡】" if dirt else "🟢 镜头清晰"
                if obj and point_coords:
                    obj_tag = f"🔴【发现异物悬挂】(定位点 <point>: X={point_coords[0]}, Y={point_coords[1]})"
                else:
                    obj_tag = "🟢 无异物"

                print(f"  ├─ 火花检测: {spk_tag} - {details.get('spark_desc', '')}")
                print(f"  ├─ 异物定位: {obj_tag} - {details.get('foreign_desc', '')}")
                print(f"  ├─ 脏污检测: {dirt_tag} - {details.get('dirt_desc', '')}")
                print(f"  └─ 总体结论: {summary}")

                # 若发现故障，保存带定位标定的高清原图
                if spk or obj or dirt:
                    event_id = int(time.time())
                    save_event_collage(window_buffer, parsed_json, save_dir, event_id)

                print("=" * 70, "\n", flush=True)

                # 清空当前窗口
                window_buffer.clear()
                timestamp_buffer.clear()

    except KeyboardInterrupt:
        print("\n[退出] 接收到用户中断信号，正在退出...", flush=True)
    finally:
        sampler.stop()
        print("[完成] 检测流程已停止。", flush=True)


def save_event_collage(frames, result_json, save_dir, event_id):
    """当出现故障时，将 4 帧拼接并在图像上绘制异物中心点与十字准星"""
    annotated_frames = [f.copy() for f in frames]
    point_coords = result_json.get("foreign_point_coords")

    if point_coords is not None:
        nx, ny = point_coords
        for f in annotated_frames:
            h, w = f.shape[:2]
            px = int(np.clip(nx / 1000.0 * w, 0, w - 1))
            py = int(np.clip(ny / 1000.0 * h, 0, h - 1))

            # 绘制醒目的红色十字准星瞄准圈
            cv2.circle(f, (px, py), 20, (0, 255, 255), 1)  # 黄色外圈
            cv2.circle(f, (px, py), 14, (0, 0, 255), 2)  # 红色主圈
            cv2.circle(f, (px, py), 4, (0, 0, 255), -1)
            cv2.line(f, (px - 26, py), (px + 26, py), (0, 0, 255), 2)
            cv2.line(f, (px, py - 26), (px, py + 26), (0, 0, 255), 2)
            cv2.putText(
                f,
                f"POINT ({nx},{ny})",
                (max(10, px - 40), max(25, py - 25)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

    # 缩小每张图以便于拼接成 2x2 拼图
    thumb_w, thumb_h = 480, 270
    resized_frames = [cv2.resize(f, (thumb_w, thumb_h)) for f in annotated_frames]

    top_row = np.hstack([resized_frames[0], resized_frames[1]])
    bottom_row = np.hstack([resized_frames[2], resized_frames[3]])
    collage = np.vstack([top_row, bottom_row])

    # 绘制状态水印
    summary_text = result_json.get("summary", "故障告警")
    cv2.putText(
        collage,
        f"FAULT ALERT: {summary_text}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )

    img_path = os.path.join(save_dir, f"event_{event_id}_collage.jpg")
    json_path = os.path.join(save_dir, f"event_{event_id}_data.json")

    cv2.imwrite(img_path, collage)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result_json, f, ensure_ascii=False, indent=2)

    print(f"  💾 🚨 故障事件（含中心点 <point> 标定）已归档: {img_path}", flush=True)


# ==================== 入口函数 ====================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Qwen3-VL 4帧滑动窗口时序视频理解与视觉定位 (Point)"
    )
    parser.add_argument(
        "--source",
        type=str,
        default="./弓网典型视频/foreign_1.mp4",
        help="视频源：支持 RTSP 地址 (rtsp://...) 或 本地 MP4 视频路径",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=DEFAULT_MODEL_PATH,
        help="Qwen3-VL-4B 本地权重目录",
    )
    parser.add_argument(
        "--save_dir", type=str, default="./events", help="故障事件与多帧拼图保存目录"
    )

    args = parser.parse_args()
    run_sliding_window_detection(args.source, args.model_path, args.save_dir)
