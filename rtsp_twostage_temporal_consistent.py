#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
受电弓网动静分离 + 方案1: 时序一致性滤波追踪器 (rtsp_twostage_temporal_consistent.py)
--------------------------------------------------------------------------------
解决痛点：
1. 单帧高清定位：保留 100% 原始像素细节，精准定位羽毛/线头等微小异物；
2. 时序一致性滤波：通过跨时序窗口的坐标空间邻近度与连续性校验（Temporal Consistency），
   彻底过滤掉单帧偶发反光、拖影、本体幻觉等瞬间误报！
"""

import os
import sys
import time
import json
import re
import math
import argparse
import threading
import queue
from datetime import datetime

import cv2
import torch
import numpy as np
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

DEFAULT_MODEL_PATH = "./models/Qwen3-VL-4B-Instruct"
WINDOW_SIZE = 4           # 每个诊断时序窗口包含的帧数
SAMPLE_INTERVAL = 1.0     # 采样间隔：1 秒 / 帧 (1 FPS)

# ================= 时序一致性追踪参数 =================
MATCH_DISTANCE_THRESH = 80.0   # 归一化坐标空间距离容差 (0-1000 坐标系下，相当于全图宽高的 8%)
CONFIRM_HIT_THRESH = 2         # 确认阈值：连续或近邻 2 次检出同一位置即确认故障
CANDIDATE_TTL_SECONDS = 6.0    # 候选点生存时间 (秒)，超时未再次命中则自动平息过滤

# ================= Prompts =================
PROMPT_STAGE1 = (
    "你是受电弓故障诊断专家。请观察图像，判断：\n"
    "1. 火花(spk)：有无刺眼强光或电弧放电？\n"
    "2. 脏污(dirt)：画面是否影响了视野，极度模糊、失焦、或被水珠/雾气/污泥遮挡？\n"
    "请先给出一句话的推理分析(reasoning)，描述画面的清晰度，然后再输出JSON。\n"
    "示例 JSON:\n"
    '{"reasoning": "画面整体极度模糊，无法看清受电弓细节，判定为脏污。", "spk": 0, "dirt": 1}'
)

PROMPT_STAGE2_SPARK = (
    "你是一个受电弓和接触网故障诊断专家。\n"
    "图像中发生了大火花或电弧放电。请仔细观察并找出火花的位置。\n"
    "如果找到了火花，请输出一段描述，并在最后必须使用XML格式给出火花的坐标，"
    "格式为：<box>x_min, y_min, x_max, y_max</box>。\n"
    "如果没有找到，请直接回复 'Normal'。"
)

PROMPT_STAGE2_FOREIGN = (
    "你是一个受电弓与接触网视觉定位专家。\n"
    "请仔细观察受电弓碳滑板（最上方的弧形横梁）及其正上方的接触网区域。\n"
    "如果发现任何不属于受电弓本体结构的附着物或悬挂物（如附着在滑板上的白色羽毛/絮状物、塑料薄膜、编织袋碎片、棉纱线头等），"
    "请使用XML格式给出该异物的中心点坐标，格式为：<point>x, y</point>（0到1000归一化，原点在左上角）。\n\n"
    "【负样本约束】受电弓本体金属部件（弓角、线夹、螺栓、卡扣、接头等）绝不是异物！列车经过站台、灯光密集区、高压线密集区时，因高速运动产生的光源拖影也不是异物。若画面正常无外来异物，请直接回复 'Normal'。"
)


class ForeignObjectCandidate:
    """单个异物候选点跟踪实体"""
    def __init__(self, pt, timestamp, window_id):
        self.point = pt  # (x, y) 归一化坐标 (0-1000)
        self.first_seen_ts = timestamp
        self.last_seen_ts = timestamp
        self.hit_count = 1
        self.window_ids = [window_id]
        self.is_confirmed = False

    def distance_to(self, other_pt):
        dx = self.point[0] - other_pt[0]
        dy = self.point[1] - other_pt[1]
        return math.sqrt(dx * dx + dy * dy)

    def update(self, pt, timestamp, window_id):
        # 移动平均平滑坐标
        self.point = (
            int(round(self.point[0] * 0.4 + pt[0] * 0.6)),
            int(round(self.point[1] * 0.4 + pt[1] * 0.6)),
        )
        self.last_seen_ts = timestamp
        self.hit_count += 1
        self.window_ids.append(window_id)


class TemporalConsistencyTracker:
    """
    【时序一致性追踪器】
    管理异物候选点生命周期，过滤单帧偶发误报，确认持续性真阳性故障。
    """
    def __init__(self, match_dist=MATCH_DISTANCE_THRESH, confirm_thresh=CONFIRM_HIT_THRESH, ttl=CANDIDATE_TTL_SECONDS):
        self.match_dist = match_dist
        self.confirm_thresh = confirm_thresh
        self.ttl = ttl
        self.candidates = []
        self.history_confirmed = []

    def update(self, raw_point, current_ts, window_id):
        """
        输入单帧检出的原始点坐标 raw_point: (x, y) or None
        返回: (status, confirmed_point, candidate_info, expired_log)
        status:
          - "CONFIRMED": 经过时序连续确认的真实故障
          - "SUSPECT": 首次或正在观察中的疑似候选点 (暂不误报)
          - "CLEARED": 本次无异物或误报已自动平息
        """
        # 1. 先清理超时的候选点 (未持续命中的单帧瞬间误报)
        active_candidates = []
        expired_candidates = []
        for cand in self.candidates:
            if (current_ts - cand.last_seen_ts) <= self.ttl:
                active_candidates.append(cand)
            else:
                expired_candidates.append(cand)
        self.candidates = active_candidates

        expired_log = []
        for exp in expired_candidates:
            if not exp.is_confirmed:
                expired_log.append(f"点 {exp.point} (仅命中 {exp.hit_count} 次)")

        # 2. 如果本窗口无检出
        if raw_point is None:
            return "CLEARED", None, None, expired_log

        # 3. 尝试与现有候选点进行空间邻近匹配
        best_match = None
        min_d = float("inf")
        for cand in self.candidates:
            d = cand.distance_to(raw_point)
            if d < self.match_dist and d < min_d:
                min_d = d
                best_match = cand

        if best_match is not None:
            # 命中已有候选点，累计时序一致性
            best_match.update(raw_point, current_ts, window_id)
            if best_match.hit_count >= self.confirm_thresh:
                best_match.is_confirmed = True
                return "CONFIRMED", best_match.point, best_match, expired_log
            else:
                return "SUSPECT", best_match.point, best_match, expired_log
        else:
            # 首次检出新候选点，建立追踪观察
            new_cand = ForeignObjectCandidate(raw_point, current_ts, window_id)
            self.candidates.append(new_cand)
            if self.confirm_thresh <= 1:
                new_cand.is_confirmed = True
                return "CONFIRMED", new_cand.point, new_cand, expired_log
            else:
                return "SUSPECT", new_cand.point, new_cand, expired_log


class RTSPStreamSampler:
    """异步多线程 RTSP / 本地视频拉流器"""
    def __init__(self, source_url, sample_interval=1.0, is_file=False):
        self.source_url = source_url
        self.sample_interval = sample_interval
        self.is_file = is_file
        self.running = False
        self.cap = None
        self.frame_queue = queue.Queue(maxsize=30)
        self.last_sample_time = 0.0

    def start(self):
        self.running = True
        self.cap = cv2.VideoCapture(self.source_url)
        if not self.cap.isOpened():
            raise RuntimeError(f"无法打开视频源: {self.source_url}")
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def _worker(self):
        fps = self.cap.get(cv2.CAP_PROP_FPS) or 25.0
        frame_delay = 1.0 / fps if self.is_file else 0.001
        frame_count = 0

        while self.running:
            ret, frame = self.cap.read()
            if not ret or frame is None:
                if self.is_file:
                    self.frame_queue.put(None)
                    break
                else:
                    time.sleep(1.0)
                    continue

            now = time.time()
            if self.is_file:
                sample_stride = max(1, int(round(fps * self.sample_interval)))
                if frame_count % sample_stride == 0:
                    self._push_frame(frame, now)
                frame_count += 1
                time.sleep(frame_delay)
            else:
                if (now - self.last_sample_time) >= self.sample_interval:
                    self._push_frame(frame, now)
                    self.last_sample_time = now
                time.sleep(frame_delay)

    def _push_frame(self, frame, timestamp):
        if self.frame_queue.full():
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                pass
        self.frame_queue.put((frame, timestamp))

    def get_frame(self, timeout=5.0):
        try:
            return self.frame_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def stop(self):
        self.running = False
        if self.cap:
            self.cap.release()


def load_vlm_model(model_path):
    print(f"[Model] 正在向 GPU 载入 Qwen3-VL 4B (FP16): {model_path} ...")
    processor = AutoProcessor.from_pretrained(model_path)
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
    print("[Model] ✅ 模型加载完成！")
    return model, processor


def infer_video(model, processor, frames_bgr, prompt, max_tokens=128):
    pil_frames = [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in frames_bgr]
    messages = [{"role": "user", "content": [{"type": "video", "video": pil_frames, "fps": 1.0}, {"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], videos=[pil_frames], padding=True, return_tensors="pt").to("cuda")

    t0 = time.time()
    with torch.no_grad():
        out_ids = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)
    latency_ms = (time.time() - t0) * 1000

    in_len = inputs["input_ids"].shape[1]
    res_text = processor.batch_decode(out_ids[:, in_len:], skip_special_tokens=True)[0].strip()
    return res_text, latency_ms


def infer_single_image(model, processor, frame_bgr, prompt, max_tokens=128):
    pil_img = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    messages = [{"role": "user", "content": [{"type": "image", "image": pil_img}, {"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[pil_img], padding=True, return_tensors="pt").to("cuda")

    t0 = time.time()
    with torch.no_grad():
        out_ids = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)
    latency_ms = (time.time() - t0) * 1000

    in_len = inputs["input_ids"].shape[1]
    res_text = processor.batch_decode(out_ids[:, in_len:], skip_special_tokens=True)[0].strip()
    return res_text, latency_ms


def save_annotated_event(frames, status_label, point_coords, output_dir="./local_cache/rtsp_temporal_results"):
    os.makedirs(output_dir, exist_ok=True)
    ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    saved_paths = []
    
    for i, frame in enumerate(frames):
        vis_frame = frame.copy()
        h, w = vis_frame.shape[:2]

        if point_coords:
            px = max(0, min(w - 1, int(point_coords[0] / 1000.0 * w)))
            py = max(0, min(h - 1, int(point_coords[1] / 1000.0 * h)))
            cv2.circle(vis_frame, (px, py), 22, (0, 0, 255), 3)
            cv2.circle(vis_frame, (px, py), 12, (0, 255, 255), 2)
            cv2.line(vis_frame, (px - 35, py), (px + 35, py), (0, 0, 255), 2)
            cv2.line(vis_frame, (px, py - 35), (px, py + 35), (0, 0, 255), 2)
            cv2.putText(
                vis_frame,
                f"FOREIGN ({point_coords[0]}, {point_coords[1]})",
                (max(10, px - 70), max(35, py - 30)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
            )

        cv2.putText(
            vis_frame,
            f"EVENT: {status_label} | Frame {i+1}/{len(frames)}",
            (30, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 255),
            2,
        )
        out_path = os.path.join(output_dir, f"event_{ts_str}_f{i+1}.jpg")
        cv2.imwrite(out_path, vis_frame)
        saved_paths.append(out_path)

    return saved_paths


def main():
    parser = argparse.ArgumentParser(description="受电弓动静分离 + 方案1时序一致性滤波检测系统")
    parser.add_argument("--source", type=str, default="./弓网典型视频/foreign_1.mp4", help="视频路径或 RTSP URL")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_PATH, help="Qwen3-VL 权重目录")
    parser.add_argument("--confirm-hits", type=int, default=2, help="时序确认阈值(默认连续/近邻2次确认)")
    args = parser.parse_args()

    model, processor = load_vlm_model(args.model)
    tracker = TemporalConsistencyTracker(confirm_thresh=args.confirm_hits)

    is_file = os.path.isfile(args.source) or not args.source.startswith("rtsp://")
    sampler = RTSPStreamSampler(args.source, sample_interval=SAMPLE_INTERVAL, is_file=is_file)
    sampler.start()

    print("\n" + "=" * 80)
    print(" 🚀 动静分离 + 方案 1: 时序一致性滤波系统已就绪 ")
    print(f" 📹 视频源: {args.source}")
    print(f" 🛡️ 时序确认阈值: {args.confirm_hits} 次空间邻近命中 | 空间距离容差: {MATCH_DISTANCE_THRESH} (0-1000 坐标系)")
    print("=" * 80 + "\n")

    window_buffer = []
    window_count = 0

    try:
        while True:
            item = sampler.get_frame(timeout=5.0)
            if item is None:
                if is_file:
                    print("\n🏁 本地测试视频已全部处理完毕。")
                    break
                continue

            frame, ts = item
            window_buffer.append(frame)

            if len(window_buffer) == WINDOW_SIZE:
                window_count += 1
                curr_time_str = datetime.now().strftime("%H:%M:%S")

                print(f"\n⚡ ===== [第 {window_count} 个时序窗口] [{curr_time_str}] Stage 1 & Stage 2 检测 =====")

                # ================= Stage 1 (视频动态时序诊断) =================
                s1_text, s1_ms = infer_video(model, processor, window_buffer, PROMPT_STAGE1, max_tokens=128)
                try:
                    s1_res = json.loads(s1_text)
                except Exception:
                    m = re.search(r"\{[^}]+\}", s1_text)
                    s1_res = json.loads(m.group()) if m else {}

                spk = s1_res.get("spk", 0)
                dirt = s1_res.get("dirt", 0)
                reasoning = s1_res.get("reasoning", "")
                print(f"  ├─ S1 (视频流) 耗时: {s1_ms:.0f} ms | 推理: {reasoning}")

                # ================= Stage 2 (细粒度单图高精度定位) =================
                raw_point = None
                s2_text = ""
                s2_ms = 0

                if dirt == 1:
                    s2_text = "Skipped (画面脏污短路保护)"
                elif spk == 1:
                    s2_text, s2_ms = infer_video(model, processor, window_buffer, PROMPT_STAGE2_SPARK, max_tokens=128)
                else:
                    # 🎯 最后一帧高清单图定位
                    last_frame = window_buffer[-1]
                    s2_text, s2_ms = infer_single_image(model, processor, last_frame, PROMPT_STAGE2_FOREIGN, max_tokens=128)

                    pt_match = re.search(r"<point>\s*(\d+)\s*,\s*(\d+)\s*</point>", s2_text)
                    if not pt_match:
                        pt_match = re.search(r"\((\d+)\s*,\s*(\d+)\)", s2_text)
                    if pt_match and "Normal" not in s2_text:
                        coords = tuple(map(int, pt_match.groups()))
                        if not (coords[0] == 0 and coords[1] == 0):
                            raw_point = coords

                if s2_text:
                    s2_short = s2_text[:80] + "..." if len(s2_text) > 80 else s2_text
                    print(f"  ├─ S2 (定位) 耗时: {s2_ms:.0f} ms | 输出: {s2_short}")

                # ================= 方案 1: 时序一致性校验 =================
                status, confirmed_pt, cand, expired_log = tracker.update(raw_point, time.time(), window_count)

                # 打印已自动平息的单帧偶发误报
                for exp_str in expired_log:
                    print(f"  🛡️ [时序滤波] 成功过滤瞬时偶发误报: {exp_str} (因未在后续窗口持续出现)")

                if status == "CONFIRMED":
                    print(f"  └─ 总体结论: 🎯 悬挂异物 [时序已确认! 连续命中 {cand.hit_count} 次，坐标 {confirmed_pt}]")
                    saved = save_annotated_event(window_buffer, "FOREIGN_CONFIRMED", confirmed_pt)
                    print(f"  💾 🚨 真实故障切片已归档至: {saved[0]}")
                elif status == "SUSPECT":
                    print(f"  └─ 总体结论: ⏳ 疑似异物候选 [时序观察中 命中 {cand.hit_count}/{args.confirm_hits} 次，坐标 {cand.point}，暂不报虚警]")
                elif spk == 1:
                    print("  └─ 总体结论: 🔥 大火花报警")
                elif dirt == 1:
                    print("  └─ 总体结论: 🌫️ 画面脏污报警")
                else:
                    print("  └─ 总体结论: ✅ 正常 (无异物/无火花/无脏污)")

                print("=" * 75)
                window_buffer.clear()

    finally:
        sampler.stop()


if __name__ == "__main__":
    main()
