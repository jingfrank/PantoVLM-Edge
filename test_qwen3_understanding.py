#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试 Qwen3-VL-4B 对受电弓图像/整段视频的综合理解能力 (test_qwen3_understanding.py)

功能支持：
1. 【整段视频时序理解模式 (默认/推荐 --mode video)】:
   从整段视频中均匀抽取时序多帧（默认 8 帧），一次性作为整段视频输入 Qwen3-VL，
   测试大模型对整段视频时序变化、事件发展（如火花何时出现、受电弓运动状态）的宏观综合理解能力。

2. 【连续逐帧分析模式 (--mode stream / --max_frames 0)】:
   连续遍历读取视频流中的每一个抽样帧，逐帧独立进行深度诊断并保存结果。

3. 【支持纯视觉无 Prompt 模式 (--pure_image)】:
   不传入任何文字 Prompt 提示词，仅输入图像/视频，测试大模型的无引导自发理解能力。

使用示例：
1. 纯视频输入，一次性分析整段完整视频 (默认抽8帧):
   python test_qwen3_understanding.py --video ./弓网典型视频/spark_1.mp4 --mode video --pure_image

2. 连续遍历分析整部视频所有帧 (从头到尾持续运行):
   python test_qwen3_understanding.py --video ./弓网典型视频/spark_1.mp4 --mode stream --pure_image --max_frames 0

3. 整段视频带自定义提问:
   python test_qwen3_understanding.py --video ./弓网典型视频/spark_1.mp4 --mode video --prompt "请描述这整段视频中受电弓的工作状态，中途是否发生了火花或异常？"
"""

import os
import sys
import time
import argparse
import cv2
import torch
import numpy as np
from PIL import Image

try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

DEFAULT_MODEL_PATH = "./models/Qwen3-VL-4B-Instruct"
DEFAULT_FRAME_PROMPT = (
    "你是一个顶级的受电弓与接触网故障诊断专家。请仔细观察这张受电弓监控图像，给出详细的画面理解和诊断分析：\n"
    "1. 【画面质量与环境】：画面清晰度如何？镜头是否有水滴、污泥、油膜遮挡或模糊失焦？\n"
    "2. 【受电弓状态】：受电弓碳滑板、弓角、支架、底座结构是否完整正常？有无倾斜变形或结构破损？\n"
    "3. 【接触网与接触状态】：受电弓与上方接触网导线的接触情况如何？是否存在刺眼的电弧放电、火花或强烈反光？\n"
    "4. 【悬挂异物排查】：碳滑板、弓角及接触网上是否有缠绕或悬挂的塑料薄膜、编织袋、风筝线、树枝等外来异物？\n"
    "5. 【最终结论】：请给出综合诊断结论（正常 / 大火花 / 画面脏污 / 悬挂异物 / 结构异常 / 倾斜）。"
)

DEFAULT_VIDEO_PROMPT = (
    "你是一个顶级的受电弓与接触网故障诊断专家。请仔细观察这段受电弓运行监控视频的时序变化，给出完整的视频分析：\n"
    "1. 【受电弓运行动态】：受电弓在整个过程中运行状态如何？是否有剧烈晃动、异常飘弓或姿态倾斜？\n"
    "2. 【放电与火花事件】：视频过程中是否发生了火花或电弧放电？若是，请描述火花出现的大致阶段与强度。\n"
    "3. 【异物与结构异常】：是否有外来悬挂物卷入或部件脱落破损？\n"
    "4. 【镜头与环境变化】：画面整体质量及背景环境变化情况。\n"
    "5. 【综合诊断结论】：整段视频的最终安全评级与故障判定。"
)


def load_model(model_path=DEFAULT_MODEL_PATH, gpu_mem_fraction=0.60):
    print("=" * 70, flush=True)
    print("正在初始化 Qwen3-VL-4B 模型...", flush=True)
    print(f"模型路径: {model_path}", flush=True)

    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(gpu_mem_fraction, device=0)
        gpu_name = torch.cuda.get_device_name(0)
        print(f"GPU 设备: {gpu_name} (已锁定 {int(gpu_mem_fraction*100)}% 显存配额)", flush=True)

    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

    t0 = time.time()
    processor = AutoProcessor.from_pretrained(model_path)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path,
        dtype=torch.float16,
        device_map="cuda:0",
        low_cpu_mem_usage=True,
    )
    model.eval()
    print(f"模型加载完毕！耗时: {time.time() - t0:.2f}s", flush=True)
    print("=" * 70, flush=True)
    return model, processor


def infer_single_frame(model, processor, frame_bgr, prompt="", max_tokens=512):
    """使用 Qwen3-VL 对单帧进行理解推理"""
    pil_img = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))

    content = [{"type": "image", "image": pil_img}]
    if prompt and prompt.strip():
        content.append({"type": "text", "text": prompt.strip()})

    messages = [{"role": "user", "content": content}]

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


def infer_video_sequence(model, processor, frames_bgr_list, prompt="", max_tokens=512):
    """使用 Qwen3-VL 对多帧时序视频序列进行整段理解推理"""
    pil_imgs = [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in frames_bgr_list]

    content = [{"type": "video", "video": pil_imgs}]
    if prompt and prompt.strip():
        content.append({"type": "text", "text": prompt.strip()})

    messages = [{"role": "user", "content": content}]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(
        text=[text], videos=[pil_imgs], padding=True, return_tensors="pt"
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


def run_video_mode(model, processor, args):
    """整段视频多帧时序理解模式"""
    print(f"\n[整段视频时序理解] 正在载入视频: {args.video}", flush=True)
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"[错误] 无法打开视频源: {args.video}", file=sys.stderr, flush=True)
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration_sec = total_frames / fps if fps > 0 else 0

    num_samples = args.num_samples
    sample_indices = np.linspace(0, total_frames - 1, num_samples, dtype=int)

    print(f"视频时长: {duration_sec:.1f} 秒 ({total_frames} 帧), 分辨率: {w}x{h}, FPS: {fps:.1f}", flush=True)
    print(f"采样策略: 在全视频中均匀抽取 {num_samples} 帧关键时序帧进行综合理解: {sample_indices.tolist()}", flush=True)
    if args.pure_image:
        print("测试模式: 【纯视觉模式 - 无任何文字 Prompt，自发时序理解】", flush=True)
    else:
        print(f"测试模式: 【带 Prompt 模式】: {args.prompt[:60]}...", flush=True)

    frames = []
    curr_idx = 0
    sample_set = set(sample_indices)
    sample_frames_dict = {}

    while cap.isOpened() and len(sample_frames_dict) < len(sample_set):
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        if curr_idx in sample_set:
            sample_frames_dict[curr_idx] = frame
        curr_idx += 1

    cap.release()

    ordered_frames = [sample_frames_dict[i] for i in sample_indices if i in sample_frames_dict]
    if not ordered_frames:
        print("[错误] 未能成功抽取到视频帧", flush=True)
        return

    print(f"\n已成功抽取 {len(ordered_frames)} 帧时序数据，送入 Qwen3-VL 进行整段视频理解...", flush=True)
    print("⚡ 正在调用 GPU 进行整段视频多模态推理，请稍候...", flush=True)

    active_prompt = "" if args.pure_image else (args.prompt if args.prompt != DEFAULT_FRAME_PROMPT else DEFAULT_VIDEO_PROMPT)

    output_text, latency_ms = infer_video_sequence(
        model, processor, ordered_frames, active_prompt, max_tokens=args.max_tokens
    )

    print("=" * 70, flush=True)
    print(f"🎬 【Qwen3-VL 对整段视频 ({args.video}) 的完整分析报告】", flush=True)
    print(f"⏱️ 推理耗时: {latency_ms:.1f} ms ({latency_ms/1000:.2f} 秒)", flush=True)
    print("-" * 70, flush=True)
    print(f"{output_text}", flush=True)
    print("=" * 70, flush=True)

    # 保存报告与采样关键帧拼图
    base_name = os.path.splitext(os.path.basename(args.video))[0]
    txt_save_path = os.path.join(args.save_dir, f"{base_name}_video_understanding.txt")
    with open(txt_save_path, "w", encoding="utf-8") as f:
        f.write(f"视频: {args.video}\n")
        f.write(f"时长: {duration_sec:.1f}s, 帧数: {total_frames}, 采样帧数: {len(ordered_frames)}\n")
        f.write(f"推理耗时: {latency_ms:.1f} ms\n")
        f.write("=" * 50 + "\n")
        f.write(f"Qwen3-VL 完整输出:\n{output_text}\n")

    print(f"💾 分析报告已保存至: {txt_save_path}", flush=True)


def run_stream_mode(model, processor, args):
    """连续遍历抽帧分析模式"""
    print(f"\n[连续遍历抽帧分析] 正在打开视频源: {args.video}", flush=True)
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"[错误] 无法打开视频源: {args.video}", file=sys.stderr, flush=True)
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"视频属性: 分辨率={w}x{h}, 总帧数={total_frames}, FPS={fps:.1f}", flush=True)
    max_desc = "不限帧数 (全视频扫描直到结束)" if args.max_frames <= 0 else f"最多分析 {args.max_frames} 帧"
    print(f"抽帧策略: 每隔 {args.interval} 帧推理一次，分析上限: {max_desc}\n", flush=True)

    active_prompt = "" if args.pure_image else args.prompt
    frame_idx = 0
    analyzed_count = 0

    while cap.isOpened():
        if args.max_frames > 0 and analyzed_count >= args.max_frames:
            break

        ret, frame = cap.read()
        if not ret or frame is None:
            print("视频读取完毕或视频流结束。", flush=True)
            break

        frame_idx += 1
        if frame_idx < args.skip_to:
            continue

        if (frame_idx - args.skip_to) % args.interval != 0:
            continue

        analyzed_count += 1
        video_time_sec = frame_idx / fps if fps > 0 else 0
        print("=" * 70, flush=True)
        print(f"🔍 [第 {analyzed_count} 次分析] (视频当前进度: 第 {frame_idx}/{total_frames} 帧, 对应时间: {video_time_sec:.2f}s)", flush=True)
        print("⚡ 正在调用 GPU 进行图像理解推理，请稍候...", flush=True)

        output_text, latency_ms = infer_single_frame(
            model, processor, frame, active_prompt, max_tokens=args.max_tokens
        )

        print(f"⏱️ 推理耗时: {latency_ms:.1f} ms ({latency_ms/1000:.2f} 秒)", flush=True)
        print(f"🤖 Qwen3-VL 完整输出:\n{output_text}", flush=True)
        print("=" * 70 + "\n", flush=True)

        # 保存结果
        base_name = f"frame_{frame_idx:06d}"
        img_save_path = os.path.join(args.save_dir, f"{base_name}.jpg")
        txt_save_path = os.path.join(args.save_dir, f"{base_name}_output.txt")
        cv2.imwrite(img_save_path, frame)
        with open(txt_save_path, "w", encoding="utf-8") as f:
            f.write(f"帧号: {frame_idx} (时间: {video_time_sec:.2f}s)\n耗时: {latency_ms:.1f}ms\nQwen3-VL:\n{output_text}\n")

    cap.release()
    print(f"\n✅ 连续遍历分析完成！共完成 {analyzed_count} 帧深度分析。", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Qwen3-VL 受电弓图像/整段视频理解与测试工具")
    parser.add_argument("--video", type=str, default="./弓网典型视频/spark_1.mp4", help="视频文件路径或 RTSP 流地址")
    parser.add_argument("--mode", type=str, choices=["video", "stream"], default="video",
                        help="测试模式: 'video' (整段视频时序理解，推荐); 'stream' (连续逐帧抽检分析)")
    parser.add_argument("--prompt", type=str, default=DEFAULT_FRAME_PROMPT, help="自定义 Prompt 提示词")
    parser.add_argument("--pure_image", action="store_true", help="启用纯视觉模式 (无任何文字 Prompt)")
    parser.add_argument("--num_samples", type=int, default=8, help="video 模式下从全视频抽取的时序帧数 (默认8帧)")
    parser.add_argument("--interval", type=int, default=30, help="stream 模式下抽帧间隔 (默认每30帧抽取1帧)")
    parser.add_argument("--skip_to", type=int, default=0, help="stream 模式下跳过前 N 帧")
    parser.add_argument("--max_frames", type=int, default=0, help="stream 模式下最大帧数 (0 表示遍历全视频不限制)")
    parser.add_argument("--max_tokens", type=int, default=512, help="模型最大输出 token 数量 (默认512)")
    parser.add_argument("--save_dir", type=str, default="./local_cache/understanding_test", help="结果保存路径")
    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH, help="模型权重目录")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    # 加载模型 (单例)
    model, processor = load_model(args.model_path)

    if args.mode == "video":
        run_video_mode(model, processor, args)
    else:
        run_stream_mode(model, processor, args)


if __name__ == "__main__":
    main()
