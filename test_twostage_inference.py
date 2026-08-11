#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qwen3-VL-4B 两阶段综合检测脚本 (test_twostage_inference.py)
Stage 1: 火花/脏污分类 (短 Prompt)
Stage 2: 悬挂异物定位 (与 grounding 脚本一致的纯净 Prompt)
两阶段对同一帧图片分别提问，避免注意力丢失。
"""

import os
import time
import json
import re
import cv2
import torch
from PIL import Image


def infer_one(model, processor, pil_img, prompt, max_tokens=128):
    """对一张图片执行一次推断，返回原始文本输出。"""
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
        generated_ids = model.generate(**inputs, max_new_tokens=max_tokens)
    latency_ms = (time.time() - t0) * 1000

    input_len = inputs["input_ids"].shape[1]
    new_ids = generated_ids[:, input_len:]
    output_text = processor.batch_decode(new_ids, skip_special_tokens=True)[0].strip()
    return output_text, latency_ms


def run_twostage_inference():
    print("=======================================================")
    print("[Qwen3-VL-4B] 正在初始化【两阶段】综合检测脚本...")
    print("=======================================================")

    model_path = "./models/Qwen3-VL-4B-Instruct"
    videos_to_test = [
        {"name": "大火花测试",   "path": "./弓网典型视频/spark_1.mp4",   "skip_to": 1400},
        {"name": "极度脏污测试", "path": "./弓网典型视频/dirty_2.mp4",   "skip_to": 1000},
        {"name": "局部脏污测试", "path": "./弓网典型视频/dirty_1.mp4",   "skip_to": 1000},
        {"name": "悬挂异物测试", "path": "./弓网典型视频/foreign_1.mp4", "skip_to": 13400},
    ]

    if not torch.cuda.is_available():
        print("[ERROR] 未检测到 CUDA GPU！")
        return

    torch.cuda.set_per_process_memory_fraction(0.50, device=0)
    print("[Qwen3-VL-4B] 🛡️ 已启用 50% GPU 显存配额保护锁 (~12GB)")

    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

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
        print("✅ [Qwen3-VL-4B] 模型加载成功！\n")
    except Exception as e:
        print(f"❌ [Qwen3-VL-4B] 权重加载异常: {e}")
        return

    save_dir = "./local_cache/twostage_results"
    os.makedirs(save_dir, exist_ok=True)

    # =============================================
    # Stage 1 Prompt: 火花 & 脏污分类 (简短专注)
    # =============================================
    prompt_stage1 = (
        "你是受电弓故障诊断专家。请观察图像，判断：\n"
        "1. 火花(spk)：受电弓与接触网之间有无刺眼强光或电弧？注意：火花光晕不是脏污。\n"
        "2. 脏污(dirt)：画面有无泥水、油污或水渍遮挡？只要污渍明显干扰了对受电弓或接触网的观察，即算脏污。\n"
        "请先写一句reasoning，再输出JSON。\n"
        "{\"reasoning\": \"...\", \"spk\": 0, \"dirt\": 0}"
    )

    for video_info in videos_to_test:
        v_name = video_info["name"]
        v_path = video_info["path"]
        skip_to = video_info["skip_to"]
        print(f"\n{'-' * 60}")
        print(f"▶️ 开始测试: {v_name} ({v_path})")
        print(f"{'-' * 60}")

        cap = cv2.VideoCapture(v_path)
        if not cap.isOpened():
            print(f"[ERROR] 无法打开视频: {v_path}")
            continue

        frame_idx = 0
        inferred = 0
        sample_interval = 25
        max_inferences = 10  # 每个视频快速验证 10 帧

        while cap.isOpened() and inferred < max_inferences:
            ret, raw_frame = cap.read()
            if not ret or raw_frame is None:
                break

            frame_idx += 1
            if frame_idx < skip_to:
                continue
            if frame_idx % sample_interval != 0:
                continue

            height, width, _ = raw_frame.shape
            pil_img = Image.fromarray(cv2.cvtColor(raw_frame, cv2.COLOR_BGR2RGB))

            # ===== Stage 1: 火花 & 脏污 =====
            s1_text, s1_ms = infer_one(model, processor, pil_img, prompt_stage1, max_tokens=128)

            # 解析 Stage 1 JSON
            try:
                s1_res = json.loads(s1_text)
            except Exception:
                m = re.search(r"\{[^}]+\}", s1_text)
                s1_res = json.loads(m.group()) if m else {}

            spk = s1_res.get("spk", 0)
            dirt = s1_res.get("dirt", 0)
            s1_reasoning = s1_res.get("reasoning", "")

            # ===== Stage 2: 悬挂异物定位 =====
            # 保留 <box> 格式指令让模型输出可解析坐标，但不告知像素尺寸
            # 模型会用原生 [0,1000] 归一化坐标输出，代码中做反归一化
            prompt_stage2 = (
                "你是一个受电弓和接触网故障诊断专家。"
                "接触网线路上可能会挂有非金属的悬挂异物（如塑料薄膜、编织袋、风筝线、树枝、鸟窝等）。\n"
                "请仔细寻找图像中的悬挂异物。\n"
                "如果找到了异物，请输出一段描述，并在最后使用XML格式给出异物的坐标，"
                "格式为：<box>x_min, y_min, x_max, y_max</box>。\n"
                "如果没有找到异物，请直接回复 'Normal'。"
            )
            s2_text, s2_ms = infer_one(model, processor, pil_img, prompt_stage2, max_tokens=128)

            # 解析 Stage 2 box（兼容两种格式）
            # 格式1: <box>x_min, y_min, x_max, y_max</box>
            box_pattern1 = r'<box>\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*</box>'
            # 格式2: (y_min,x_min),(y_max,x_max) — Qwen原生格式，注意x/y顺序反转
            box_pattern2 = r'\((\d+)\s*,\s*(\d+)\)\s*,?\s*\((\d+)\s*,\s*(\d+)\)'
            box_match = re.search(box_pattern1, s2_text)
            box_format = 1
            if not box_match:
                box_match = re.search(box_pattern2, s2_text)
                box_format = 2
            total_ms = s1_ms + s2_ms

            # 构造日志
            flags = []
            if spk == 1:
                flags.append("🔥 大火花")
            if dirt == 1:
                flags.append("🌫️ 脏污")
            if box_match:
                flags.append("📦 悬挂异物")

            status_str = " | ".join(flags) if flags else "✅ 正常"
            print(f"[F:{frame_idx:05d}] [{total_ms:5.0f}ms] {status_str}")
            print(f"  ├─ S1({s1_ms:4.0f}ms): {s1_reasoning}")
            if box_match:
                print(f"  └─ S2({s2_ms:4.0f}ms): {s2_text}")
            else:
                s2_short = s2_text[:80] if len(s2_text) > 80 else s2_text
                print(f"  └─ S2({s2_ms:4.0f}ms): {s2_short}")

            # 如果检出异物，坐标反归一化后画框保存
            if box_match:
                raw_coords = list(map(int, box_match.groups()))
                if box_format == 1:
                    # <box>x_min, y_min, x_max, y_max</box>
                    nx1, ny1, nx2, ny2 = raw_coords
                else:
                    # (y_min,x_min),(y_max,x_max) — Qwen原生格式，x/y反转
                    ny1, nx1, ny2, nx2 = raw_coords

                # 从 [0, 1000] 归一化坐标映射到原图绝对像素坐标
                x1 = int(nx1 / 1000.0 * width)
                y1 = int(ny1 / 1000.0 * height)
                x2 = int(nx2 / 1000.0 * width)
                y2 = int(ny2 / 1000.0 * height)

                # 确保坐标在图像范围内
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(width, x2), min(height, y2)

                print(f"  📐 坐标映射: 归一化({nx1},{ny1},{nx2},{ny2}) → 像素({x1},{y1},{x2},{y2})")

                draw_frame = raw_frame.copy()
                cv2.rectangle(draw_frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                cv2.putText(
                    draw_frame, "Foreign Object",
                    (x1, max(y1 - 10, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2,
                )
                save_path = os.path.join(save_dir, f"{v_name}_f{frame_idx:05d}.jpg")
                cv2.imwrite(save_path, draw_frame)

            inferred += 1

        cap.release()

    print(f"\n{'=' * 55}")
    print(f"[Qwen3-VL-4B] ✅ 两阶段综合检测验证完成！")
    print(f"[Qwen3-VL-4B] 📁 异物检测图片保存至: {save_dir}")


if __name__ == "__main__":
    run_twostage_inference()
