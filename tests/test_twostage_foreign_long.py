#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qwen3-VL-4B 悬挂异物专项检测脚本 (test_twostage_foreign_long.py)
去掉动态路由，每帧直接执行异物检测。
使用精准、笃定的 Prompt 提升异物定位精度。
后续验证效果后再整合回综合检测脚本。
"""

import os
import time
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


def run_foreign_detection():
    print("=======================================================")
    print("[Qwen3-VL-4B] 正在初始化【悬挂异物专项检测】脚本...")
    print("=======================================================")

    model_path = "./models/Qwen3-VL-4B-Instruct"
    videos_to_test = [
        {"name": "悬挂异物测试(开头2分钟)", "path": "./弓网典型视频/foreign_1.mp4", "skip_to": 0},
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

    save_dir = "./local_cache/foreign_long_results"
    os.makedirs(save_dir, exist_ok=True)

    # =============================================
    # 异物检测 Prompt：笃定、收敛、精准
    # 借鉴火花 Prompt 成功经验：语气笃定 + 目标明确
    # =============================================
    prompt_foreign = (
        "你是一个视觉定位专家。\n"
        "请仔细观察受电弓碳滑板（最上方的弧形横梁）及其附近的接触网区域。\n"
        "如果发现任何不属于受电弓本体结构的附着物或悬挂物，请标注出来。\n\n"
        "【什么是异物】外来的非金属附着物，如：缠绕的塑料薄膜、编织袋碎片、棉纱线头等。"
        "异物通常表面蓬松、毛糙、边缘不规则，材质为布料/塑料/纤维。\n"
        "【什么不是异物】受电弓本身的金属部件绝不是异物！包括：弓角、线夹、螺栓、卡扣、铰接关节、"
        "弹簧座、集电靴接头、导线连接件等。这些部件表面光滑、形状规则、材质为金属。\n\n"
        "如果发现了真正的异物，请输出简短描述并用<box>x_min, y_min, x_max, y_max</box>给出最紧凑的坐标。\n"
        "如果画面中只有受电弓本体结构而没有外来异物，请直接回复 'Normal'。"
    )

    # 框面积过滤阈值：异物不可能占据超过 15% 的画面
    MAX_BOX_RATIO = 0.15

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
        filtered = 0
        sample_interval = 25
        max_inferences = 120  # 测试前 2 分钟，1秒1帧，共 120 帧

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

            # ===== 直接执行异物检测，无动态路由 =====
            text, latency_ms = infer_one(model, processor, pil_img, prompt_foreign, max_tokens=128)

            # 解析 box（兼容两种格式）
            box_pattern1 = r'<box>\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*</box>'
            box_pattern2 = r'\((\d+)\s*,\s*(\d+)\)\s*,?\s*\((\d+)\s*,\s*(\d+)\)'
            box_match = re.search(box_pattern1, text)
            box_format = 1
            if not box_match:
                box_match = re.search(box_pattern2, text)
                box_format = 2

            if box_match:
                raw_coords = list(map(int, box_match.groups()))
                if box_format == 1:
                    nx1, ny1, nx2, ny2 = raw_coords
                else:
                    ny1, nx1, ny2, nx2 = raw_coords

                # 从 [0, 1000] 归一化坐标映射到原图绝对像素坐标
                x1 = int(nx1 / 1000.0 * width)
                y1 = int(ny1 / 1000.0 * height)
                x2 = int(nx2 / 1000.0 * width)
                y2 = int(ny2 / 1000.0 * height)

                # 确保坐标在图像范围内
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(width, x2), min(height, y2)

                # 框面积过滤：异物很小，超大框是误检
                box_area = (x2 - x1) * (y2 - y1)
                image_area = width * height
                box_ratio = box_area / image_area

                if box_ratio > MAX_BOX_RATIO:
                    print(f"[F:{frame_idx:05d}] [{latency_ms:5.0f}ms] ⚠️ 框过大 ({box_ratio*100:.1f}%) 已丢弃")
                    text_short = text[:80] if len(text) > 80 else text
                    print(f"  └─ {text_short}")
                    filtered += 1
                else:
                    detected += 1
                    print(f"[F:{frame_idx:05d}] [{latency_ms:5.0f}ms] 📦 悬挂异物")
                    text_short = text[:100] if len(text) > 100 else text
                    print(f"  └─ {text_short}")
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
            else:
                text_short = text[:60] if len(text) > 60 else text
                print(f"[F:{frame_idx:05d}] [{latency_ms:5.0f}ms] ✅ 正常 | {text_short}")

            inferred += 1

        cap.release()

    print(f"\n{'=' * 55}")
    print(f"[Qwen3-VL-4B] ✅ 悬挂异物专项检测完成！")
    print(f"[Qwen3-VL-4B] 📊 统计: 检测 {inferred} 帧, 检出 {detected} 帧, 过滤大框 {filtered} 帧")
    print(f"[Qwen3-VL-4B] 📁 检测图片保存至: {save_dir}")


if __name__ == "__main__":
    run_foreign_detection()
