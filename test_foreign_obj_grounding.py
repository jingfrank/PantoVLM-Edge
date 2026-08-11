#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qwen3-VL-4B 悬挂异物检测专用测试脚本 - Visual Grounding 方案 (test_foreign_obj_grounding.py)
激发模型输出目标检测框，而不是仅仅回答是或否。
"""

import os
import sys
import time
import json
import re
import cv2
import torch
from PIL import Image


def run_qwen3_vl_grounding(
    video_path="./弓网典型视频/foreign_1.mp4",
    model_path="./models/Qwen3-VL-4B-Instruct",
):
    print(f"=======================================================")
    print(f"[Qwen3-VL-4B] 正在初始化 Visual Grounding 脚本...")
    print(f"  ├─ 模型权重路径: {model_path}")
    print(f"  └─ 测试视频路径: {video_path}")
    print(f"=======================================================")

    if not torch.cuda.is_available():
        print("[ERROR] 未检测到 CUDA GPU，请确认 NVIDIA 驱动正常！")
        return

    # 显存安全保护
    torch.cuda.set_per_process_memory_fraction(0.50, device=0)
    print("[Qwen3-VL-4B] 🛡️ 已启用 50% GPU 显存配额保护锁 (~12GB)")

    from transformers import (
        Qwen3VLForConditionalGeneration,
        Qwen3VLProcessor,
        AutoProcessor,
    )

    try:
        processor = AutoProcessor.from_pretrained(model_path)
        print("[Qwen3-VL-4B] 正在向 GPU 加载模型 (FP16)...")
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path,
            dtype=torch.float16,
            device_map="cuda:0",
            low_cpu_mem_usage=True,
        )
        model.eval()
        print(f"✅ [Qwen3-VL-4B] 模型加载成功！")

    except Exception as e:
        print(f"❌ [Qwen3-VL-4B] 权重加载异常: {e}")
        import traceback
        traceback.print_exc()
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] 无法打开视频: {video_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    print(f"[Qwen3-VL-4B] 视频读取成功 | 总帧数: {total_frames} | FPS: {fps:.1f}")

    # 创建保存目录
    save_dir = "./local_cache/grounding_results"
    os.makedirs(save_dir, exist_ok=True)
    print(f"[Qwen3-VL-4B] 结果图片将保存至: {save_dir}")

    frame_idx = 0
    sample_interval = 25  # 每 25 帧（1 秒）采样 1 帧
    inferred = 0
    detected_count = 0

    while cap.isOpened():
        ret, raw_frame = cap.read()
        if not ret or raw_frame is None:
            break

        frame_idx += 1
        if frame_idx % sample_interval != 0 and frame_idx != 1:
            continue

        height, width, _ = raw_frame.shape
        pil_img = Image.fromarray(cv2.cvtColor(raw_frame, cv2.COLOR_BGR2RGB))

        # 专为 Visual Grounding 设计的 Prompt
        prompt = (
            "你是一个受电弓和接触网故障诊断专家。接触网线路上可能会挂有非金属的悬挂异物（如塑料薄膜、编织袋、风筝线、树枝、鸟窝等）。\n"
            "请仔细寻找图像中的悬挂异物。\n"
            f"图像的宽度是 {width} 像素，高度是 {height} 像素。\n"
            "如果找到了异物，请输出一段描述，并在最后使用XML格式给出异物在图像上的绝对像素坐标，格式为：<box>x_min, y_min, x_max, y_max</box>。\n"
            "如果没有找到异物，请直接回复 'Normal'。"
        )

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_img},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = processor(
            text=[text], images=[pil_img], padding=True, return_tensors="pt"
        )
        inputs = {k: v.to("cuda") for k, v in inputs.items()}

        t0 = time.time()
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=128)
        latency_ms = (time.time() - t0) * 1000

        input_len = inputs["input_ids"].shape[1]
        new_ids = generated_ids[:, input_len:]
        output_text = processor.batch_decode(new_ids, skip_special_tokens=True)[0].strip()

        # 解析坐标框
        # 寻找类似于 <box>100, 200, 300, 400</box> 或者 <box>100,200,300,400</box>
        box_pattern = r'<box>\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*</box>'
        match = re.search(box_pattern, output_text)

        if match:
            x1, y1, x2, y2 = map(int, match.groups())
            detected_count += 1
            print(f"[Frame {frame_idx:04d}] [{latency_ms:6.1f}ms] 🚨 发现异物 | 坐标: ({x1},{y1})-({x2},{y2}) | 输出: {output_text}")
            
            # 画框并保存图片
            draw_frame = raw_frame.copy()
            cv2.rectangle(draw_frame, (x1, y1), (x2, y2), (0, 0, 255), 3) # 红色粗框
            cv2.putText(draw_frame, "Foreign Object", (x1, max(y1 - 10, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            save_path = os.path.join(save_dir, f"frame_{frame_idx:04d}_detected.jpg")
            cv2.imwrite(save_path, draw_frame)
        else:
            print(f"[Frame {frame_idx:04d}] [{latency_ms:6.1f}ms] ✅ 未检出异物框 | 输出: {output_text}")

        inferred += 1
        
        # 为了测试效率，推断前10000帧后也可以停止，或完整跑完
        # if frame_idx > 10000: break

    cap.release()

    print(f"\n{'=' * 55}")
    print(f"[Qwen3-VL-4B] ✅ 测试完成！共推断 {inferred} 帧")
    print(f"[Qwen3-VL-4B] 📊 成功画框检出异物: {detected_count} 帧")

if __name__ == "__main__":
    run_qwen3_vl_grounding()
