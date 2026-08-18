#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
对比测试：针对 foreign_1.mp4 验证「视频模式 vs 最后一帧单图模式 vs RoI 局部特写模式」在两阶段检测中的实际效果与耗时
"""

import os
import sys
import time
import re
import cv2
import torch
import numpy as np
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

MODEL_PATH = "./models/Qwen3-VL-4B-Instruct"
VIDEO_PATH = "./弓网典型视频/foreign_1.mp4"
SAVE_DIR = "./local_cache/image_vs_video_results"
os.makedirs(SAVE_DIR, exist_ok=True)

# Prompts
PROMPT_STAGE1 = (
    "你是受电弓故障诊断专家。请观察图像，判断：\n"
    "1. 火花(spk)：有无刺眼强光或电弧放电？\n"
    "2. 脏污(dirt)：画面是否影响了视野，极度模糊、失焦、或被水珠/雾气/污泥遮挡？\n"
    "请先给出一句话的推理分析(reasoning)，描述画面的清晰度，然后再输出JSON。\n"
    "示例 JSON:\n"
    '{"reasoning": "画面整体极度模糊，无法看清受电弓细节，判定为脏污。", "spk": 0, "dirt": 1}'
)

# 纯单图全景 Prompt
PROMPT_STAGE2_FOREIGN_IMAGE = (
    "你是一个受电弓与接触网视觉定位专家。\n"
    "请仔细观察受电弓碳滑板（最上方的弧形横梁）及其正上方的接触网区域。\n"
    "如果发现任何不属于受电弓本体结构的附着物或悬挂物（如附着在滑板上的白色羽毛/絮状物、塑料薄膜、编织袋碎片、棉纱线头等），"
    "请使用XML格式给出该异物的中心点坐标，格式为：<point>x, y</point>（0到1000归一化，原点在左上角）。\n\n"
    "【负样本约束】受电弓本体金属部件（弓角、线夹、螺栓、卡扣、接头等）绝不是异物！列车经过站台、灯光密集区、高压线密集区时，因高速运动产生的光源拖影也不是异物。若画面正常无外来异物，请直接回复 'Normal'。"
)

# RoI 局部特写 Prompt
PROMPT_STAGE2_FOREIGN_ROI = (
    "你是一个视觉定位专家。这是一张受电弓碳滑板（最上方横梁）与接触网的局部放大特写图。\n"
    "请仔细观察碳滑板横梁上方是否有外来附着物或悬挂物（如白色羽毛、编织袋碎片、棉纱线头等）。\n"
    "如果发现异物，请先简要描述，并在最后使用XML格式输出坐标：<point>x, y</point>（0-1000归一化）。若无异物请回复 'Normal'。"
)


def load_model():
    print(f"正在加载模型: {MODEL_PATH} ...", flush=True)
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    
    # 设置 720P 视频处理器限制（用于 Stage 1）
    if hasattr(processor, "video_processor") and processor.video_processor is not None:
        processor.video_processor.size = {
            "shortest_edge": 64 * 32 * 32,
            "longest_edge": int(960 * 720 * 2),
        }

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        dtype=torch.float16,
        device_map="cuda:0",
        low_cpu_mem_usage=True,
    )
    model.eval()
    return model, processor


def infer_video(model, processor, frames_bgr, prompt, max_tokens=128):
    pil_frames = [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in frames_bgr]
    content = [
        {"type": "video", "video": pil_frames, "fps": 1.0},
        {"type": "text", "text": prompt},
    ]
    text = processor.apply_chat_template([{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], videos=[pil_frames], padding=True, return_tensors="pt").to("cuda")

    t0 = time.time()
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)
    ms = (time.time() - t0) * 1000
    res = processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()
    return res, ms


def infer_single_image(model, processor, frame_bgr, prompt, max_tokens=128):
    pil_img = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    content = [
        {"type": "image", "image": pil_img},
        {"type": "text", "text": prompt},
    ]
    text = processor.apply_chat_template([{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[pil_img], return_tensors="pt").to("cuda")

    t0 = time.time()
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)
    ms = (time.time() - t0) * 1000
    res = processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()
    return res, ms


def infer_roi_image(model, processor, frame_bgr, prompt, max_tokens=128):
    h, w = frame_bgr.shape[:2]
    # 裁剪受电弓碳滑板上部区域: x [25%, 75%], y [18%, 55%]
    x1, y1, x2, y2 = int(0.25 * w), int(0.18 * h), int(0.75 * w), int(0.55 * h)
    roi_bgr = frame_bgr[y1:y2, x1:x2]
    
    pil_roi = Image.fromarray(cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB))
    content = [
        {"type": "image", "image": pil_roi},
        {"type": "text", "text": prompt},
    ]
    text = processor.apply_chat_template([{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[pil_roi], return_tensors="pt").to("cuda")

    t0 = time.time()
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)
    ms = (time.time() - t0) * 1000
    res = processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()
    
    # 解析并映射回原图坐标
    pt_match = re.search(r"<point>\s*(\d+)\s*,\s*(\d+)\s*</point>", res)
    mapped_point = None
    if pt_match:
        rx, ry = int(pt_match.group(1)), int(pt_match.group(2))
        full_x = int((0.25 + (rx / 1000.0) * 0.50) * 1000)
        full_y = int((0.18 + (ry / 1000.0) * 0.37) * 1000)
        mapped_point = (full_x, full_y)
        
    return res, ms, mapped_point


def main():
    model, processor = load_model()
    
    cap = cv2.VideoCapture(VIDEO_PATH)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    
    # 抽取第 8, 9, 10 个滑动窗口（每个窗口 4 帧，每秒 1 帧）
    # 对应秒数：28~32s, 32~36s, 36~40s
    test_windows = [
        ("窗口 8 (28-32s 前导正常/微弱帧)", 28),
        ("窗口 9 (32-36s 异物核心关键帧)", 32),
        ("窗口 10 (36-40s 异物持续帧)", 36),
    ]

    print("\n" + "=" * 80)
    print("开始对比评测：视频模式 vs 全图单图模式 vs RoI 局部特写模式")
    print("=" * 80 + "\n")

    for w_name, start_sec in test_windows:
        frames = []
        for i in range(4):
            frame_idx = int((start_sec + i) * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                frames.append(frame)
        
        if len(frames) < 4:
            continue
            
        print(f"🎬 【{w_name}】测试开始:")
        
        # 1. 视频模式推理
        res_v, ms_v = infer_video(model, processor, frames, PROMPT_STAGE2_FOREIGN_IMAGE, max_tokens=128)
        print(f"   [模式 1: 4帧视频模式]     耗时: {ms_v:5.0f}ms | 结果: {res_v}")

        # 2. 最后一帧高清单图模式
        res_s, ms_s = infer_single_image(model, processor, frames[-1], PROMPT_STAGE2_FOREIGN_IMAGE, max_tokens=128)
        print(f"   [模式 2: 最后一帧高清单图] 耗时: {ms_s:5.0f}ms | 结果: {res_s}")

        # 3. 碳滑板 RoI 局部特写模式
        res_r, ms_r, mapped_pt = infer_roi_image(model, processor, frames[-1], PROMPT_STAGE2_FOREIGN_ROI, max_tokens=128)
        pt_info = f" (映射全图: {mapped_pt})" if mapped_pt else ""
        print(f"   [模式 3: 碳滑板 RoI 特写]   耗时: {ms_r:5.0f}ms | 结果: {res_r}{pt_info}")

        # 保存高清标注对比图
        vis_frame = frames[-1].copy()
        h, w = vis_frame.shape[:2]
        
        # 绘制模式 2 点 (绿色)
        m2_match = re.search(r"<point>\s*(\d+)\s*,\s*(\d+)\s*</point>", res_s)
        if m2_match:
            px = int(int(m2_match.group(1)) / 1000.0 * w)
            py = int(int(m2_match.group(2)) / 1000.0 * h)
            cv2.circle(vis_frame, (px, py), 16, (0, 255, 0), 2)
            cv2.putText(vis_frame, f"M2-Image: ({px},{py})", (px + 20, py - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # 绘制模式 3 点 (红色十字准星)
        if mapped_pt:
            px = int(mapped_pt[0] / 1000.0 * w)
            py = int(mapped_pt[1] / 1000.0 * h)
            cv2.circle(vis_frame, (px, py), 12, (0, 0, 255), 2)
            cv2.line(vis_frame, (px - 20, py), (px + 20, py), (0, 0, 255), 2)
            cv2.line(vis_frame, (px, py - 20), (px, py + 20), (0, 0, 255), 2)
            cv2.putText(vis_frame, f"M3-RoI: ({px},{py})", (px + 20, py + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        save_path = os.path.join(SAVE_DIR, f"{w_name[:5]}.jpg")
        cv2.imwrite(save_path, vis_frame)
        print(f"   💾 对比标注图已保存: {save_path}\n")

    cap.release()
    print("=" * 80)
    print("对比评测完成！")
    print("=" * 80)


if __name__ == "__main__":
    main()
