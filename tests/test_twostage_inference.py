#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qwen3-VL-4B 统一两阶段综合检测脚本 (test_twostage_inference.py)
Stage 1: 火花/脏污全局分类 (短 Prompt + JSON 规范)
Stage 2 动态路由:
  - dirt == 1: 极度脏污，跳过 Stage 2 防止误报画框
  - spk == 1: 触发火花 <box> 框定位分支
  - dirt == 0 & spk == 0: 触发悬挂异物 <point> 靶心点定位分支 (含金属部件反向约束)
"""

import os
import time
import json
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
    print("[Qwen3-VL-4B] 正在初始化【统一两阶段动态路由】综合检测脚本...")
    print("=======================================================")

    model_path = "./models/Qwen3-VL-4B-Instruct"
    videos_to_test = [
        {"name": "悬挂异物测试", "path": "./弓网典型视频/foreign_1.mp4", "skip_to": 800},
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
    # Stage 1 Prompt: 火花 & 脏污全局分类
    # =============================================
    prompt_stage1 = (
        "你是受电弓故障诊断专家。请观察图像，判断：\n"
        "1. 火花(spk)：有无刺眼强光或电弧放电？\n"
        "2. 脏污(dirt)：画面是否极度模糊、失焦、或被水滴/污泥遮挡？注意：不要推测模糊的原因！只要画面模糊到难以辨认受电弓细节，就必须判定为脏污(dirt=1)。\n"
        "请先给出一句话的推理分析(reasoning)，描述画面的清晰度，然后再输出JSON。\n"
        "示例 JSON:\n"
        "{\"reasoning\": \"画面整体极度模糊，无法看清受电弓细节，判定为脏污。\", \"spk\": 0, \"dirt\": 1}"
    )

    # =============================================
    # Stage 2A Prompt: 火花 <box> 矩形框定位
    # =============================================
    prompt_stage2_spark = (
        "你是一个受电弓和接触网故障诊断专家。\n"
        "图像中发生了大火花或电弧放电。请仔细观察并找出火花的位置。\n"
        "如果找到了火花，请输出一段描述，并在最后必须使用XML格式给出火花的坐标，"
        "格式为：<box>x_min, y_min, x_max, y_max</box>。\n"
        "如果没有找到，请直接回复 'Normal'。"
    )

    # =============================================
    # Stage 2B Prompt: 悬挂异物 <point> 点定位 (含反向约束)
    # =============================================
    prompt_stage2_foreign = (
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

            # ===== Stage 1: 火花 & 脏污 全局分类 =====
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

            # ===== 动态路由 Stage 2 =====
            s2_text, s2_ms = "", 0
            box_match = None
            point_match = None

            if dirt == 1:
                # 分支 1：极度脏污短路保护，跳过 Stage 2
                s2_text = "Skipped (Dirt detected)"
            elif spk == 1:
                # 分支 2：大火花定位 (<box> 模式)
                s2_text, s2_ms = infer_one(model, processor, pil_img, prompt_stage2_spark, max_tokens=128)
                box_pattern1 = r'<box>\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*</box>'
                box_pattern2 = r'\((\d+)\s*,\s*(\d+)\)\s*,?\s*\((\d+)\s*,\s*(\d+)\)'
                box_match = re.search(box_pattern1, s2_text)
                if not box_match:
                    box_match = re.search(box_pattern2, s2_text)
            else:
                # 分支 3：悬挂异物点定位 (<point> 模式)
                s2_text, s2_ms = infer_one(model, processor, pil_img, prompt_stage2_foreign, max_tokens=128)
                point_pattern1 = r'<point>\s*(\d+)\s*,\s*(\d+)\s*</point>'
                point_pattern2 = r'\((\d+)\s*,\s*(\d+)\)'
                point_match = re.search(point_pattern1, s2_text)
                if not point_match:
                    point_match = re.search(point_pattern2, s2_text)

            total_ms = s1_ms + s2_ms

            # 构造控制台展示状态
            flags = []
            if spk == 1:
                flags.append("🔥 大火花")
            if dirt == 1:
                flags.append("🌫️ 脏污")
            if box_match and spk == 1:
                flags.append("🎯 火花定位(Box)")
            elif point_match and spk == 0 and dirt == 0:
                flags.append("🎯 悬挂异物(Point)")

            status_str = " | ".join(flags) if flags else "✅ 正常"
            print(f"[F:{frame_idx:05d}] [{total_ms:5.0f}ms] {status_str}")
            print(f"  ├─ S1({s1_ms:4.0f}ms): {s1_reasoning}")
            s2_short = s2_text[:80] if len(s2_text) > 80 else s2_text
            print(f"  └─ S2({s2_ms:4.0f}ms): {s2_short}")

            # ===== 图像结果绘制与保存 =====
            draw_frame = raw_frame.copy()
            should_save = False

            # Case A: 绘制火花 <box>
            if box_match and spk == 1:
                raw_coords = list(map(int, box_match.groups()))
                if len(raw_coords) == 4:
                    nx1, ny1, nx2, ny2 = raw_coords
                    x1 = max(0, min(width, int(nx1 / 1000.0 * width)))
                    y1 = max(0, min(height, int(ny1 / 1000.0 * height)))
                    x2 = max(0, min(width, int(nx2 / 1000.0 * width)))
                    y2 = max(0, min(height, int(ny2 / 1000.0 * height)))
                    print(f"  📐 火花 Box 坐标: 归一化({nx1},{ny1},{nx2},{ny2}) → 像素({x1},{y1},{x2},{y2})")

                    cv2.rectangle(draw_frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                    cv2.putText(
                        draw_frame, "Spark",
                        (x1, max(y1 - 10, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2,
                    )
                    should_save = True

            # Case B: 绘制悬挂异物 <point> 准星标尺
            elif point_match and spk == 0 and dirt == 0:
                nx, ny = map(int, point_match.groups())
                px = max(0, min(width - 1, int(nx / 1000.0 * width)))
                py = max(0, min(height - 1, int(ny / 1000.0 * height)))
                print(f"  📐 异物 Point 坐标: 归一化({nx},{ny}) → 像素({px},{py})")

                # 同心圆与十字准星
                cv2.circle(draw_frame, (px, py), 15, (0, 0, 255), 3)
                cv2.circle(draw_frame, (px, py), 4, (0, 0, 255), -1)
                cv2.line(draw_frame, (px - 25, py), (px + 25, py), (0, 0, 255), 2)
                cv2.line(draw_frame, (px, py - 25), (px, py + 25), (0, 0, 255), 2)
                cv2.putText(
                    draw_frame, f"Foreign Object Point ({px},{py})",
                    (max(px - 100, 10), max(py - 30, 30)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2,
                )
                should_save = True

            # Case C: 脏污画框标记
            elif dirt == 1:
                cv2.putText(
                    draw_frame, "WARNING: Dirt / Blur Detected",
                    (50, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 165, 255), 3,
                )
                should_save = True

            if should_save:
                save_path = os.path.join(save_dir, f"{v_name}_f{frame_idx:05d}.jpg")
                cv2.imwrite(save_path, draw_frame)

            inferred += 1

        cap.release()

    print(f"\n{'=' * 55}")
    print(f"[Qwen3-VL-4B] ✅ 统一两阶段动态路由检测完成！")
    print(f"[Qwen3-VL-4B] 📁 异常检测图片保存至: {save_dir}")


if __name__ == "__main__":
    run_twostage_inference()
