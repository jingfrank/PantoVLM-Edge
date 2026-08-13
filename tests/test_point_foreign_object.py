#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qwen3-VL-4B 悬挂异物点定位 (Point Grounding) 检测脚本 (test_point_foreign_object.py)
输出异物中心点 <point>x, y</point> 坐标，避免传统画框在复杂场景下包揽大区/框选全弓的问题。
"""

import os
import time
import re
import cv2
import torch
from PIL import Image


def infer_one(model, processor, pil_img, prompt, max_tokens=128):
    """对一张图片执行一次推断，返回原始文本输出与延迟。"""
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
    inputs = processor(text=[text], images=[pil_img], padding=True, return_tensors="pt")
    inputs = {k: v.to("cuda") for k, v in inputs.items()}

    t0 = time.time()
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=max_tokens)
    latency_ms = (time.time() - t0) * 1000

    input_len = inputs["input_ids"].shape[1]
    new_ids = generated_ids[:, input_len:]
    output_text = processor.batch_decode(new_ids, skip_special_tokens=True)[0].strip()
    return output_text, latency_ms


def run_point_detection():
    print("=======================================================")
    print("[Qwen3-VL-4B] 正在初始化【悬挂异物点定位】检测脚本...")
    print("=======================================================")

    model_path = "./models/Qwen3-VL-4B-Instruct"
    videos_to_test = [
        {
            "name": "悬挂异物点定位测试(开头2分钟)",
            "path": "./弓网典型视频/foreign_1.mp4",
            "skip_to": 0,
        },
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

    save_dir = "./local_cache/point_foreign_results"
    os.makedirs(save_dir, exist_ok=True)

    # =============================================
    # 点定位 (Point Grounding) Prompt
    # =============================================
    prompt_point = (
        "你是一个视觉定位专家。\n"
        "请仔细观察受电弓碳滑板（最上方的弧形横梁）及其附近的接触网区域。\n"
        "如果发现任何不属于受电弓本体结构的附着物或悬挂物（如缠绕的塑料薄膜、编织袋碎片、棉纱线头等），"
        "请指出该异物最核心的中心点位置，并在最后使用XML格式给出点坐标，格式为：<point>x, y</point>（坐标归一化在 0 到 1000 之间）。\n\n"
        "【什么不是异物】受电弓本身的金属部件（弓角、线夹、螺栓、卡扣、铰接关节、弹簧座、集电靴接头等）绝不是异物！\n"
        "如果画面中只有受电弓本体结构而没有外来异物，请直接回复 'Normal'。"
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
        detected = 0
        sample_interval = 25
        max_inferences = 600  # 测试前 2 分钟，1秒1帧，共 120 帧

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

            # 推断
            text, latency_ms = infer_one(
                model, processor, pil_img, prompt_point, max_tokens=128
            )

            # 解析 point (<point>x, y</point> 或 <point>y, x</point> 或 (x, y) / (y, x))
            point_pattern1 = r"<point>\s*(\d+)\s*,\s*(\d+)\s*</point>"
            point_pattern2 = r"\((\d+)\s*,\s*(\d+)\)"

            point_match = re.search(point_pattern1, text)
            if not point_match:
                point_match = re.search(point_pattern2, text)

            if point_match:
                nx, ny = map(int, point_match.groups())

                # 映射到像素坐标
                px = int(nx / 1000.0 * width)
                py = int(ny / 1000.0 * height)

                # 限制坐标在有效图像范围内
                px = max(0, min(width - 1, px))
                py = max(0, min(height - 1, py))

                detected += 1
                print(f"[F:{frame_idx:05d}] [{latency_ms:5.0f}ms] 🎯 点定位悬挂异物")
                text_short = text[:100] if len(text) > 100 else text
                print(f"  └─ {text_short}")
                print(f"  📐 坐标映射: 归一化({nx},{ny}) → 像素({px},{py})")

                # 可视化：绘制红色靶心与圆圈
                draw_frame = raw_frame.copy()
                # 绘制同心圆
                cv2.circle(draw_frame, (px, py), 15, (0, 0, 255), 3)
                cv2.circle(draw_frame, (px, py), 4, (0, 0, 255), -1)
                # 绘制 Crosshair 准星十字线
                cv2.line(draw_frame, (px - 25, py), (px + 25, py), (0, 0, 255), 2)
                cv2.line(draw_frame, (px, py - 25), (px, py + 25), (0, 0, 255), 2)
                # 标注文本
                cv2.putText(
                    draw_frame,
                    f"Foreign Object Point ({px},{py})",
                    (max(px - 100, 10), max(py - 30, 30)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2,
                )
                save_path = os.path.join(save_dir, f"{v_name}_f{frame_idx:05d}.jpg")
                cv2.imwrite(save_path, draw_frame)
            else:
                text_short = text[:60] if len(text) > 60 else text
                print(
                    f"[F:{frame_idx:05d}] [{latency_ms:5.0f}ms] ✅ 正常 | {text_short}"
                )

            inferred += 1

        cap.release()

    print(f"\n{'=' * 55}")
    print(f"[Qwen3-VL-4B] ✅ 悬挂异物点定位检测完成！")
    print(f"[Qwen3-VL-4B] 📊 统计: 检测 {inferred} 帧, 检出 {detected} 帧")
    print(f"[Qwen3-VL-4B] 📁 检测结果图片保存至: {save_dir}")


if __name__ == "__main__":
    run_point_detection()
