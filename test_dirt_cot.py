#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qwen3-VL-4B 脏污检测专用测试脚本 (test_dirt_cot.py)
引入 CoT (Chain of Thought) 机制解决全屏极度脏污被漏检的问题。
"""

import os
import sys
import time
import json
import cv2
import torch
from PIL import Image


def run_dirt_detection(
    video_path="./弓网典型视频/spark_1.mp4",
    model_path="./models/Qwen3-VL-4B-Instruct",
):
    print(f"=======================================================")
    print(f"[Qwen3-VL-4B] 正在初始化脏污检测 (CoT) 脚本...")
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

    # ==========================================
    # 核心：带有 CoT 和极度模糊处理的 Prompt
    # ==========================================
    prompt = (
        "你是一个受电弓故障诊断专家。请仔细观察这张受电弓和接触网图像，判断是否存在以下故障：\n"
        "1. spk: 大火花 (Spark)\n"
        "2. str: 受电弓结构异常 (Structural Anomaly)\n"
        "3. dirt: 画面脏污遮挡 (Dirty Camera)，画面脏污主要源于镜头表面粘附泥水、油污或飞虫残体，表现为图形局部区域的固定遮挡或持续模糊。以及整个画面被泥水、油污完全覆盖导致极度模糊、完全看不清受电弓和接触网的严重情况。如果画面模糊到无法辨认设备，必须判定为 dirt(1)\n"
        "4. tilt: 受电弓倾斜 (Tilt)\n"
        "5. obj: 悬挂异物 (Foreign Object)\n"
        "6. flt: 异常飘弓 (Abnormal Float)\n"
        "请先给出一句话的推理分析(reasoning)，描述画面的清晰度和你看到了什么，然后再输出标准 JSON 格式。\n"
        "示例 JSON 格式：\n"
        '{"reasoning": "画面被大面积污泥覆盖，极度模糊，无法看清受电弓结构。", "spk":0, "str":0, "dirt":1, "tilt":0, "obj":0, "flt":0}'
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

        pil_img = Image.fromarray(cv2.cvtColor(raw_frame, cv2.COLOR_BGR2RGB))

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
            # max_new_tokens 增加到 128，以容纳 reasoning 文本
            generated_ids = model.generate(**inputs, max_new_tokens=128)
        latency_ms = (time.time() - t0) * 1000

        input_len = inputs["input_ids"].shape[1]
        new_ids = generated_ids[:, input_len:]
        output_text = processor.batch_decode(new_ids, skip_special_tokens=True)[
            0
        ].strip()

        # 解析 JSON 输出 (兼容 reasoning)
        try:
            res = json.loads(output_text)
        except Exception:
            import re

            m = re.search(r"\{[^}]+\}", output_text)
            res = (
                json.loads(m.group())
                if m
                else {
                    "reasoning": "解析错误",
                    "spk": 0,
                    "str": 0,
                    "dirt": 0,
                    "tilt": 0,
                    "obj": 0,
                    "flt": 0,
                }
            )

        results.append({"frame": frame_idx, "latency_ms": round(latency_ms, 1), **res})

        is_dirt = res.get("dirt", 0)
        if is_dirt:
            fault_count["dirt"] += 1
            print(
                f"[Frame {frame_idx:04d}] [{latency_ms:6.1f}ms] 🚨 画面脏污 | Reasoning: {res.get('reasoning', '')}"
            )
        else:
            print(
                f"[Frame {frame_idx:04d}] [{latency_ms:6.1f}ms] ✅ 正常 | Reasoning: {res.get('reasoning', '')}"
            )

        inferred += 1

        # 脏污视频由于画面一直很糊，我们取前 2000 帧（推断约 80 次）即可验证效果
        if frame_idx > 2000:
            break

    cap.release()

    print(f"\n{'=' * 55}")
    print(f"[Qwen3-VL-4B] ✅ 测试完成！共采样 {inferred} 帧")
    print(
        f"[Qwen3-VL-4B] 📊 脏污检出统计: {fault_count['dirt']} 帧 (占比 {fault_count['dirt'] / inferred * 100:.1f}%)"
    )


if __name__ == "__main__":
    run_dirt_detection()
