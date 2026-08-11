#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qwen3-VL-4B 综合异常检测终极版脚本 (test_comprehensive_inference.py)
融合了 CoT、视觉定位(Visual Grounding) 和防混淆(Anti-Confusion) 逻辑。
"""

import os
import sys
import time
import json
import re
import cv2
import torch
from PIL import Image


def run_comprehensive_inference():
    print(f"=======================================================")
    print(f"[Qwen3-VL-4B] 正在初始化【终极版】综合异常检测脚本...")
    print(f"=======================================================")

    model_path = "./models/Qwen3-VL-4B-Instruct"
    videos_to_test = [
        {"name": "悬挂异物测试", "path": "./弓网典型视频/foreign_1.mp4"},
    ]

    if not torch.cuda.is_available():
        print("[ERROR] 未检测到 CUDA GPU，请确认 NVIDIA 驱动正常！")
        return

    torch.cuda.set_per_process_memory_fraction(0.50, device=0)

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
        print(f"✅ [Qwen3-VL-4B] 模型加载成功！\n")
    except Exception as e:
        print(f"❌ [Qwen3-VL-4B] 权重加载异常: {e}")
        return

    save_dir = "./local_cache/comprehensive_results"
    os.makedirs(save_dir, exist_ok=True)

    # 终极版 Master CoT Prompt
    prompt_template = (
        "图像的宽度是 {width} 像素，高度是 {height} 像素。\n\n"
        "你是一个受电弓故障诊断专家。请仔细观察这张受电弓和接触网图像，判断是否存在以下故障：\n"
        "1. spk: 大火花 (Spark)\n"
        "2. str: 受电弓结构异常 (Structural Anomaly)\n"
        "3. dirt: 画面脏污遮挡 (Dirty Camera)，画面是否存在明显的泥水、油污或水渍遮挡。只要污渍明显干扰了对受电弓或接触网的观察（无论是局部遮挡还是全屏模糊），即视为脏污。\n"
        "4. tilt: 受电弓倾斜 (Tilt)\n"
        "5. obj: 悬挂异物 (Foreign Object),接触网线路上可能会挂有非金属的悬挂异物（如塑料薄膜、编织袋、风筝线、树枝、鸟窝等）\n"
        "6. flt: 异常飘弓 (Abnormal Float)\n"
        "请先输出一段推理过程(reasoning)，严格按照上述3个步骤描述你的观察。\n"
        "然后输出标准 JSON 格式。\n"
        "尝仔细寻找图像中的悬挂异物。请在 JSON 的 'obj_box' 字段中用 <box>x_min, y_min, x_max, y_max</box> 给出绝对像素坐标；如果没有发现异物，'obj_box' 填 \"Normal\"。\n"
        "示例 JSON 格式：\n"
        '{{"reasoning": "", "spk": , "dirt": , "obj_box": "Normal", "str": , "tilt": , "flt": }}'
    )

    for video_info in videos_to_test:
        v_name = video_info["name"]
        v_path = video_info["path"]
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
        # 每个视频只推断最多 10 帧，快速验证效果
        max_inferences = 100

        while cap.isOpened() and inferred < max_inferences:
            ret, raw_frame = cap.read()
            if not ret or raw_frame is None:
                break

            frame_idx += 1
            # 异物视频前段可能没有异物，快进到后面有异物的帧
            if v_name == "悬挂异物测试" and frame_idx < 1:
                continue

            # 脏污视频取稍后一点的帧确保都是脏的
            if v_name == "极度脏污测试" and frame_idx < 1000:
                continue

            # 火花视频快进到有火花的时刻
            if v_name == "大火花测试" and frame_idx < 1400:
                continue

            if frame_idx % sample_interval != 0:
                continue

            height, width, _ = raw_frame.shape
            # 对于异物检测，我们可以仍做一定的 ROI 裁剪，或者直接原图，这里统一原图测试模型的泛化能力
            pil_img = Image.fromarray(cv2.cvtColor(raw_frame, cv2.COLOR_BGR2RGB))

            prompt = prompt_template.format(width=width, height=height)

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
                generated_ids = model.generate(**inputs, max_new_tokens=256)
            latency_ms = (time.time() - t0) * 1000

            input_len = inputs["input_ids"].shape[1]
            new_ids = generated_ids[:, input_len:]
            output_text = processor.batch_decode(new_ids, skip_special_tokens=True)[
                0
            ].strip()

            # 解析 JSON
            try:
                res = json.loads(output_text)
            except Exception:
                import re

                m = re.search(r"\{[\s\S]*?\}", output_text)
                res = json.loads(m.group()) if m else {}

            spk = res.get("spk", 0)
            dirt = res.get("dirt", 0)
            obj_box = res.get("obj_box", "Normal")
            reasoning = res.get("reasoning", "")

            # 判断检出情况，构造短日志
            flags = []
            if spk == 1:
                flags.append("🔥 大火花")
            if dirt == 1:
                flags.append("🌫️ 脏污")
            if "box" in obj_box:
                flags.append("📦 悬挂异物")

            status_str = " | ".join(flags) if flags else "✅ 正常"
            print(
                f"[F:{frame_idx:04d}] [{latency_ms:4.0f}ms] {status_str}\n  └─ {reasoning}"
            )

            # 如果检出了异物，画框保存
            if "box" in obj_box:
                box_pattern = (
                    r"<box>\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*</box>"
                )
                match = re.search(box_pattern, obj_box)
                if match:
                    x1, y1, x2, y2 = map(int, match.groups())
                    draw_frame = raw_frame.copy()
                    cv2.rectangle(draw_frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                    cv2.putText(
                        draw_frame,
                        "Obj",
                        (x1, max(y1 - 10, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        (0, 0, 255),
                        2,
                    )
                    save_path = os.path.join(save_dir, f"{v_name}_f{frame_idx}.jpg")
                    cv2.imwrite(save_path, draw_frame)

            inferred += 1

        cap.release()

    print(f"\n{'=' * 55}")
    print(f"[Qwen3-VL-4B] ✅ 综合异常检测验证完成！")


if __name__ == "__main__":
    run_comprehensive_inference()
