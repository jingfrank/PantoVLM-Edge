#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qwen3-VL-4B 悬挂异物检测专用测试脚本 (test_foreign_obj_roi.py)
使用 ROI (Region of Interest) 裁剪方法放大接触网区域，并引入 CoT 推理
"""

import os
import sys
import time
import json
import cv2
import torch
from PIL import Image


def run_qwen3_vl_inference_roi(
    video_path="./弓网典型视频/foreign_1.mp4",
    model_path="./models/Qwen3-VL-4B-Instruct",
):
    print(f"=======================================================")
    print(f"[Qwen3-VL-4B] 正在初始化悬挂异物检测(ROI裁剪)脚本...")
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
        print("[Qwen3-VL-4B] 正在向 GPU (TITAN RTX) 加载模型 (FP16)...")
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

    prompt = (
        "你是一个受电弓和接触网故障诊断专家。这张图像是受电弓上方接触网区域的局部放大图。\n"
        "请仔细观察接触网线路上，检测受电弓上是否有任何非金属的悬挂异物（如塑料薄膜、编织袋、风筝线、树枝、鸟窝等）,这是open vocabulary objecgt detection问题。\n"
        "请先给出一句话的推理分析(reasoning)，然后再输出标准 JSON 格式。\n"
        "仅按标准 JSON 格式输出，1表示有故障，0表示正常:\n"
        '{"reasoning": "...", "obj":0, "spk":0, "str":0, "dirt":0, "tilt":0, "flt":0}'
    )

    frame_idx = 0
    sample_interval = 25  # 每 25 帧（1 秒）采样 1 帧
    results = []
    fault_count = {"spk": 0, "str": 0, "dirt": 0, "tilt": 0, "obj": 0, "flt": 0}
    inferred = 0

    while cap.isOpened():
        ret, raw_frame = cap.read()
        if not ret or raw_frame is None:
            break

        frame_idx += 1
        if frame_idx % sample_interval != 0 and frame_idx != 1:
            continue

        # ==========================================
        # ROI 裁剪逻辑：保留上半部分 60%
        # ==========================================
        height, width, _ = raw_frame.shape
        roi_height = int(height * 0.6)
        roi_frame = raw_frame[0:roi_height, :]

        pil_img = Image.fromarray(cv2.cvtColor(roi_frame, cv2.COLOR_BGR2RGB))

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
        output_text = processor.batch_decode(new_ids, skip_special_tokens=True)[
            0
        ].strip()

        # 解析 JSON 输出
        try:
            res = json.loads(output_text)
        except Exception:
            import re

            m = re.search(r"\{[^}]+\}", output_text)
            res = (
                json.loads(m.group())
                if m
                else {
                    "reasoning": "Parse Error",
                    "obj": 0,
                    "spk": 0,
                    "str": 0,
                    "dirt": 0,
                    "tilt": 0,
                    "flt": 0,
                }
            )

        results.append({"frame": frame_idx, "latency_ms": round(latency_ms, 1), **res})

        is_obj = res.get("obj", 0)
        if is_obj:
            fault_count["obj"] += 1
            print(
                f"[Frame {frame_idx:04d}] [{latency_ms:6.1f}ms] 🚨 悬挂异物 | Reasoning: {res.get('reasoning', '')}"
            )
        else:
            print(
                f"[Frame {frame_idx:04d}] [{latency_ms:6.1f}ms] ✅ 正常 | Reasoning: {res.get('reasoning', '')}"
            )

        inferred += 1

    cap.release()

    print(f"\n{'=' * 55}")
    print(f"[Qwen3-VL-4B] ✅ 测试完成！共采样 {inferred} 帧")
    print(f"[Qwen3-VL-4B] 📊 异物检出统计: {fault_count['obj']} 帧")


if __name__ == "__main__":
    run_qwen3_vl_inference_roi()
