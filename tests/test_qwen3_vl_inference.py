#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qwen3-VL-4B 真实权重离线视频推断测试脚本 (test_qwen3_vl_inference.py)
使用 transformers 4.57.6 原生 Qwen3VLForConditionalGeneration 加载模型进行真实 GPU 推断
"""

import os
import sys
import time
import json
import cv2
import torch
from PIL import Image


def run_qwen3_vl_inference(
    video_path="./弓网典型视频/dirty_1.mp4",
    model_path="./models/Qwen3-VL-4B-Instruct",
):
    print(f"=======================================================")
    print(f"[Qwen3-VL-4B] 正在初始化真实大模型推断脚本...")
    print(f"  ├─ 模型权重路径: {model_path}")
    print(f"  └─ 测试视频路径: {video_path}")
    print(f"=======================================================")

    if not torch.cuda.is_available():
        print("[ERROR] 未检测到 CUDA GPU，请确认 NVIDIA 驱动正常！")
        return

    # 显存安全保护：限制为 35% GPU 显存，保障系统稳定不冻结
    torch.cuda.set_per_process_memory_fraction(0.50, device=0)
    print("[Qwen3-VL-4B] 🛡️ 已启用 50% GPU 显存配额保护锁 (~12GB)，保障系统稳定性")

    # =========================================================
    # 使用 transformers 4.57.6 原生 Qwen3VLForConditionalGeneration
    # =========================================================
    from transformers import (
        Qwen3VLForConditionalGeneration,
        Qwen3VLProcessor,
        AutoProcessor,
    )

    try:
        print("[Qwen3-VL-4B] 正在加载 Qwen3VLProcessor 处理器...")
        processor = AutoProcessor.from_pretrained(model_path)
        print("[Qwen3-VL-4B] Processor 加载成功！")

        print(
            "[Qwen3-VL-4B] 正在向 GPU (TITAN RTX) 加载 Qwen3-VL-4B 真实大模型权重 (FP16)..."
        )
        t_load_start = time.time()
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path,
            dtype=torch.float16,
            device_map="cuda:0",
            low_cpu_mem_usage=True,
        )
        model.eval()
        t_load = time.time() - t_load_start

        vram_mb = torch.cuda.memory_allocated(0) / 1024 / 1024
        print(f"✅ [Qwen3-VL-4B] 真实大模型权重成功加载至 GPU 显存！")
        print(f"   ├─ 加载耗时: {t_load:.1f} 秒")
        print(f"   └─ GPU 显存占用: {vram_mb:.0f} MB\n")

    except Exception as e:
        print(f"❌ [Qwen3-VL-4B] 权重加载异常: {e}")
        import traceback

        traceback.print_exc()
        return

    # =========================================================
    # 打开视频进行真实大模型推断
    # =========================================================
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] 无法打开视频: {video_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    print(f"[Qwen3-VL-4B] 视频读取成功 | 总帧数: {total_frames} | FPS: {fps:.1f}")
    print(f"[Qwen3-VL-4B] 每 30 帧采样 1 帧送入大模型检测\n")

    prompt = (
        "你是一个受电弓故障诊断专家。请仔细观察这张受电弓和接触网图像，判断是否存在以下故障：\n"
        "1. spk: 大火花 (Spark)\n"
        "2. str: 受电弓结构异常 (Structural Anomaly)\n"
        "3. dirt: 画面脏污遮挡 (Dirty Camera)，画面脏污主要源于镜头表面粘附泥水、油污或飞虫残体，表现为图形局部区域的固定遮挡或持续模糊。以及整个画面被泥水、油污完全覆盖导致极度模糊、完全看不清受电弓和接触网的严重情况。如果画面模糊到无法辨认设备，必须判定为 dirt\n"
        "4. tilt: 受电弓倾斜 (Tilt)\n"
        "5. obj: 悬挂异物 (Foreign Object),悬挂异物是指接触网或受电弓上挂有塑料薄膜、风筝线、鸟窝或树枝等细小物体，仔细观察弓网区域。\n"
        "6. flt: 异常飘弓 (Abnormal Float)\n"
        "仅按标准 JSON 格式输出，1表示有故障，0表示正常:\n"
        '{"spk":0, "str":0, "dirt":0, "tilt":0, "obj":0, "flt":0}'
    )

    frame_idx = 0
    sample_interval = 25  # 每 25 帧（1 秒）采样 1 帧送入大模型
    results = []  # 保存所有帧的检测结果
    fault_count = {"spk": 0, "str": 0, "dirt": 0, "tilt": 0, "obj": 0, "flt": 0}
    fault_labels = {
        "spk": "大火花",
        "str": "结构异常",
        "dirt": "画面脏污",
        "tilt": "受电弓倾斜",
        "obj": "悬挂异物",
        "flt": "异常飘弓",
    }
    inferred = 0
    while cap.isOpened():
        ret, raw_frame = cap.read()
        if not ret or raw_frame is None:
            break

        frame_idx += 1
        if frame_idx % sample_interval != 0 and frame_idx != 1:
            continue

        # 将 OpenCV BGR 转为 PIL Image
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
            generated_ids = model.generate(**inputs, max_new_tokens=64)
        latency_ms = (time.time() - t0) * 1000

        # 只截取新生成的 token
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
                else {"spk": 0, "str": 0, "dirt": 0, "tilt": 0, "obj": 0, "flt": 0}
            )

        results.append({"frame": frame_idx, "latency_ms": round(latency_ms, 1), **res})
        for k in fault_count:
            if res.get(k, 0):
                fault_count[k] += 1

        # 只在检测到故障时打印（否则太多输出）
        faults_found = [fault_labels[k] for k in fault_count if res.get(k, 0)]
        if faults_found:
            print(
                f"[Frame {frame_idx:04d}/{total_frames:04d}] [{latency_ms:6.1f}ms] 🚨 {', '.join(faults_found)} | {output_text}"
            )
        else:
            print(
                f"[Frame {frame_idx:04d}/{total_frames:04d}] [{latency_ms:6.1f}ms] ✅ 正常 | {output_text}"
            )
        inferred += 1

    cap.release()

    # 保存结果到 JSON
    result_path = (
        f"./local_cache/qwen3vl_inference_result_{time.strftime('%m%d_%H%M%S')}.json"
    )
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "total_frames_sampled": inferred,
                "fault_count": fault_count,
                "results": results,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"\n{'=' * 55}")
    print(f"[Qwen3-VL-4B] ✅ 完整视频推断完成！共采样 {inferred} 帧")
    print(f"[Qwen3-VL-4B] 📊 故障检出统计:")
    for k, v in fault_count.items():
        if v > 0:
            print(f"   ├─ {fault_labels[k]}: {v} 帧")
    print(f"[Qwen3-VL-4B] 📄 结果已保存至: {result_path}")


if __name__ == "__main__":
    run_qwen3_vl_inference()
