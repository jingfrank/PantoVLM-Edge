#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qwen3-VL 全视频 4 帧滑动窗口强制异物检测测试脚本 (Point 准星标定修复版)
(tests/test_forced_foreign_object.py)

参考 test_twostage_inference.py 坐标映射与同心圆十字准星绘制机制：
1. 【坐标精准换算】：px = int(nx / 1000.0 * width), py = int(ny / 1000.0 * height)
2. 【同心圆 + 十字准星 + 黑色抗干扰文字底板】：确保在任何复杂接触网背景下均清晰可见
3. 【拼图与高清原图双存】：
   - 2x2 拼图在缩放后直接重绘高清准星，避免缩放模糊
   - 检出异物时同步保存 1080P 全分辨率标定原图，方便微观细节放大复核
"""

import os
import sys
import time
import json
import re
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

# ==================== 4 帧时序滑动窗口强制异物 Prompt (Point 点定位) ====================
PROMPT_WINDOW_FORCED_POINT = (
    "这是动车列车车顶摄像头拍摄的受电弓视频图像"
    "输入是由连续4秒（每秒1帧，共4帧）构成的时序视频片段。当前画面中高度怀疑或已知存在外来悬挂异物（如塑料袋碎片、编织薄膜、防尘网、风筝线、布条或树枝）。\n"
    "请仔细观察4帧画面中受电弓碳滑板（顶部横梁）、双侧弓角及接触网，进行强制深度排查：\n\n"
    "1. 描述你在这4秒时序过程中发现的异物外观特征与动态（颜色、材质、附着部位、是否随风飘动）。\n"
    "2. 找出异物最清晰可见的一帧，给出对应的帧序号（1、2、3 或 4）以及该异物在该帧中的归一化中心点定位坐标，格式必须为：frame <序号> <point>x, y</point>；若确实无外来异物则填 null。\n"
    "3. 给出该异物的置信度评分（0% - 100%）。\n\n"
    "if foreign object does not exist in a frame, skip that frame。\n"
    "请严格按以下 JSON 格式输出（请勿输出任何多余的前缀或Markdown格式）：\n"
    "{\n"
    '  "foreign_object": 1,\n'
    '  "foreign_point": "frame 2 <point>x, y</point>",\n'
    '  "confidence": "90%",\n'
    '  "description": "描述异物特征与定位位置"\n'
    "}"
)

PROMPT_WINDOW_COT_POINT = (
    "你是一个受电弓高精度视觉巡检专家。\n"
    "输入是由连续4秒（4帧）构成的时序视频片段。请执行三步渐进式排查：\n"
    "【第一步：结构定位】定位4帧画面中受电弓碳滑板工作面、左右弓角及上方接触线。\n"
    "【第二步：异常特征排查】对比排查碳滑板及接触线上是否有不属于正常机械金属结构的附着物（排除正常螺栓、线夹与支架）。\n"
    "【第三步：中心点定位】对于发现的非金属悬挂物/异物，请找出它最清晰的一帧（序号1-4），并输出中心点坐标。格式：frame <序号> <point>x, y</point>（0-1000 归一化）。\n\n"
    "请严格按 JSON 格式输出：\n"
    "{\n"
    '  "foreign_object": 1,\n'
    '  "foreign_point": "frame 2 <point>x, y</point>",\n'
    '  "confidence": "90%",\n'
    '  "description": "三步分析过程与异物描述"\n'
    "}"
)


# ==================== 模型加载 ====================
def load_vlm_model(model_path=DEFAULT_MODEL_PATH, gpu_mem_fraction=0.65):
    print("=" * 75, flush=True)
    print(f"正在加载 Qwen3-VL 视频理解与点定位模型...", flush=True)
    print(f"模型权重路径: {model_path}", flush=True)

    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(gpu_mem_fraction, device=0)
        gpu_name = torch.cuda.get_device_name(0)
        print(
            f"GPU 设备: {gpu_name} (分配显存上限: {int(gpu_mem_fraction * 100)}%)",
            flush=True,
        )

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
    print(f"✅ 模型加载成功！耗时: {time.time() - t0:.2f}s", flush=True)
    print("=" * 75, flush=True)
    return model, processor


# ==================== 4 帧时序推理 ====================
def infer_4frame_window(
    model, processor, frames_bgr_list, prompt=PROMPT_WINDOW_FORCED_POINT, max_tokens=384
):
    """
    将 4 帧画面组装成多模态时序序列送入 Qwen3-VL 进行推理与 Point 坐标定位
    """
    pil_frames = [
        Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in frames_bgr_list
    ]

    content = [
        {"type": "video", "video": pil_frames, "fps": 1.0},
        {"type": "text", "text": prompt},
    ]
    messages = [{"role": "user", "content": content}]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(
        text=[text], videos=[pil_frames], padding=True, return_tensors="pt"
    )
    inputs = {k: v.to("cuda") for k, v in inputs.items()}

    t0 = time.time()
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,  # 确定性贪婪解码
        )
    latency_ms = (time.time() - t0) * 1000

    input_len = inputs["input_ids"].shape[1]
    new_ids = generated_ids[:, input_len:]
    output_text = processor.batch_decode(new_ids, skip_special_tokens=True)[0].strip()

    # 尝试解析 JSON
    parsed_json = None
    try:
        clean_text = output_text
        if "```json" in clean_text:
            clean_text = clean_text.split("```json")[1].split("```")[0].strip()
        elif "```" in clean_text:
            clean_text = clean_text.split("```")[1].split("```")[0].strip()

        json_match = re.search(r"\{[\s\S]*\}", clean_text)
        if json_match:
            parsed_json = json.loads(json_match.group(0))
        else:
            parsed_json = json.loads(clean_text)
    except Exception:
        pass

    if parsed_json is None:
        parsed_json = {
            "foreign_object": 0,
            "foreign_point": None,
            "confidence": "0%",
            "description": output_text,
        }

    # ==================== 正则提取中心点坐标 <point>x, y</point> 与 帧序号 ====================
    search_target = str(parsed_json.get("foreign_point") or "") + " " + output_text
    point_match = re.search(r"<point>\s*(\d+)\s*,\s*(\d+)\s*</point>", search_target)

    # 提取帧序号 frame_num (如 frame 2, 默认为 4)
    frame_num = 4
    fm = re.search(r"(?i)frame\s*(\d)", search_target)
    if fm:
        try:
            fn = int(fm.group(1))
            if 1 <= fn <= 4:
                frame_num = fn
        except:
            pass

    if point_match:
        nx = int(point_match.group(1))
        ny = int(point_match.group(2))
        parsed_json["foreign_object"] = 1
        parsed_json["point_coords"] = (nx, ny)
        parsed_json["frame_num"] = frame_num
        parsed_json["foreign_point"] = f"frame {frame_num} <point>{nx}, {ny}</point>"
    else:
        parenthesis_match = re.search(r"\((\d+)\s*,\s*(\d+)\)", search_target)
        if parenthesis_match:
            nx = int(parenthesis_match.group(1))
            ny = int(parenthesis_match.group(2))
            parsed_json["foreign_object"] = 1
            parsed_json["point_coords"] = (nx, ny)
            parsed_json["frame_num"] = frame_num
            parsed_json["foreign_point"] = (
                f"frame {frame_num} <point>{nx}, {ny}</point>"
            )
        else:
            parsed_json["point_coords"] = None
            if parsed_json.get("foreign_object") == 1:
                parsed_json["foreign_object"] = 0

    return output_text, parsed_json, latency_ms


# ==================== 参考 test_twostage_inference.py 绘制准星 ====================
def draw_twostage_point_target(img, nx, ny, label="Foreign Point"):
    """
    按照 test_twostage_inference.py 绘制标准同心圆与十字准星
    nx, ny: 0-1000 归一化坐标
    """
    h, w = img.shape[:2]
    px = max(0, min(w - 1, int(nx / 1000.0 * w)))
    py = max(0, min(h - 1, int(ny / 1000.0 * h)))

    # 1. 外层黄色光圈 + 红色主同心圆
    cv2.circle(img, (px, py), 18, (0, 255, 255), 1)
    cv2.circle(img, (px, py), 15, (0, 0, 255), 3)
    # 2. 中心实心圆点
    cv2.circle(img, (px, py), 4, (0, 0, 255), -1)
    # 3. 红色十字准星线
    cv2.line(img, (px - 25, py), (px + 25, py), (0, 0, 255), 2)
    cv2.line(img, (px, py - 25), (px, py + 25), (0, 0, 255), 2)

    # 4. 靶心坐标文字 (带黑色底板以增强在复杂背景下的可读性)
    tag_text = f"{label} ({nx},{ny})"
    (tw, th), _ = cv2.getTextSize(tag_text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
    tag_x = max(10, min(px - int(tw / 2), w - tw - 10))
    tag_y = max(th + 10, py - 25)

    cv2.rectangle(
        img, (tag_x - 4, tag_y - th - 4), (tag_x + tw + 4, tag_y + 4), (0, 0, 0), -1
    )
    cv2.putText(
        img,
        tag_text,
        (tag_x, tag_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )
    return img, (px, py)


# ==================== 主测试流程：全视频滑动窗口遍历 ====================
def run_full_video_sliding_window_test(args):
    os.makedirs(args.save_dir, exist_ok=True)
    video_path = args.video

    if not os.path.exists(video_path):
        print(f"[错误] 视频文件不存在: {video_path}", file=sys.stderr)
        return

    # 1. 载入模型
    prompt = (
        PROMPT_WINDOW_COT_POINT
        if args.strategy == "cot_point"
        else PROMPT_WINDOW_FORCED_POINT
    )
    model, processor = load_vlm_model(args.model_path)

    # 2. 读取视频元数据
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration_sec = total_frames / fps

    print(f"\n🎬 【全视频滑动窗口异物测试启动 (Point 准星定位修复版)】")
    print(f"  ├─ 视频源: {video_path}")
    print(
        f"  ├─ 视频规格: {w}x{h} | FPS: {fps:.1f} | 总帧数: {total_frames} ({duration_sec:.1f} 秒)"
    )
    print(f"  ├─ 采样策略: 每隔 {args.sample_interval_sec} 秒抽取 1 帧")
    print(
        f"  ├─ 窗口规格: 窗口大小 = {args.window_size} 帧 (覆盖 4 秒) | 滑动步长 = {args.stride} 帧"
    )
    print(f"  └─ 结果目录: {args.save_dir}")
    print("=" * 75, flush=True)

    # 3. 按 1 秒抽 1 帧预先采样整部视频的帧索引
    sample_interval_frames = max(1, int(round(fps * args.sample_interval_sec)))
    sampled_frame_indices = list(range(0, total_frames, sample_interval_frames))
    print(
        f"📦 全视频共抽样得到 {len(sampled_frame_indices)} 帧基准画面 (总计 {len(sampled_frame_indices) // args.stride} 个窗口)"
    )

    # 4. 执行全视频滑动窗口遍历
    window_size = args.window_size
    stride = args.stride
    window_idx = 0
    detected_count = 0
    total_latency_ms = 0.0
    detection_log = []

    t_start_all = time.time()

    for start_pos in range(0, len(sampled_frame_indices) - window_size + 1, stride):
        window_idx += 1
        current_indices = sampled_frame_indices[start_pos : start_pos + window_size]

        # 计算时间跨度 (秒)
        t_start_sec = current_indices[0] / fps
        t_end_sec = current_indices[-1] / fps
        time_span_str = f"{t_start_sec:.1f}s - {t_end_sec:.1f}s (Frames: {current_indices[0]}-{current_indices[-1]})"

        print(f"\n🔄 ===== [窗口 #{window_idx:03d} | 时间区间: {time_span_str}] =====")

        # 读取该窗口对应的 4 帧
        window_frames = []
        for f_idx in current_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(f_idx))
            ret, frame = cap.read()
            if ret and frame is not None:
                window_frames.append(frame)

        if len(window_frames) != window_size:
            print(f"[跳过] 窗口帧数不足 ({len(window_frames)}/{window_size})")
            continue

        # 执行 Qwen3-VL 4 帧时序强制异物推理 (Point 模式)
        output_text, parsed_json, latency_ms = infer_4frame_window(
            model, processor, window_frames, prompt=prompt, max_tokens=args.max_tokens
        )
        total_latency_ms += latency_ms

        is_obj = parsed_json.get("foreign_object", 0)
        point_coords = parsed_json.get("point_coords")
        conf = parsed_json.get("confidence", "未知")
        desc = parsed_json.get("description", "")

        # 打印窗口诊断
        print(f"⏱️ 推理耗时: {latency_ms:.1f} ms")
        if is_obj and point_coords:
            detected_count += 1
            nx, ny = point_coords
            px_val = int(nx / 1000.0 * w)
            py_val = int(ny / 1000.0 * h)
            print(
                f"🔴 🚨 【发现悬挂异物】 置信度: {conf} | 归一化坐标: ({nx}, {ny}) → 原图像素: ({px_val}, {py_val})"
            )
            print(f"   📝 异物描述: {desc}")
        else:
            print(f"🟢 正常无异物 | 输出描述: {desc[:60]}...")

        # 仅保存全分辨率高清原图，不使用拼图
        if (is_obj and point_coords) or args.save_all:
            tag = "DETECTED" if is_obj else "NORMAL"

            # 使用大模型选择的帧序号进行标定和保存
            best_f_idx = parsed_json.get("frame_num", 4) if is_obj else 4
            f_idx = best_f_idx - 1  # 转换为 0-index

            full_frame = window_frames[f_idx].copy()
            if is_obj and point_coords:
                draw_twostage_point_target(
                    full_frame, point_coords[0], point_coords[1], label="Foreign Object"
                )

            # 绘制文字水印
            cv2.putText(
                full_frame,
                f"TIME: {time_span_str} | Target Frame: {best_f_idx}/4 | {tag}",
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0, 0, 255) if is_obj else (0, 255, 0),
                3,
            )

            full_save_name = f"win_{window_idx:03d}_{tag}_F{best_f_idx}_hd.jpg"
            full_save_path = os.path.join(args.save_dir, full_save_name)
            cv2.imwrite(full_save_path, full_frame)

            if is_obj:
                print(
                    f"   💾 大模型锁定第 {best_f_idx} 帧, 高清原图已保存: {full_save_name}"
                )
            else:
                print(f"   💾 1080P 高清原图已保存至: {full_save_name}")

        detection_log.append(
            {
                "window_idx": window_idx,
                "time_span": time_span_str,
                "t_start_sec": t_start_sec,
                "t_end_sec": t_end_sec,
                "foreign_object": is_obj,
                "point_coords": point_coords,
                "confidence": conf,
                "description": desc,
                "latency_ms": latency_ms,
            }
        )

    cap.release()
    total_test_time = time.time() - t_start_all

    # ==================== 5. 输出全视频检测汇总报表 ====================
    print("\n" + "=" * 75)
    print("📊 🎬 【全视频 4 帧滑动窗口异物检测 (Point 定位修复版) - 综合评测报告】")
    print("=" * 75)
    print(f"📁 视频文件: {video_path}")
    print(f"⏱️ 视频总时长: {duration_sec:.1f} 秒 | 窗口规格: 每窗口 4 帧 (覆盖 4 秒)")
    print(f"🔢 总处理窗口数: {window_idx} 个")
    print(
        f"🚨 检出异物窗口数: {detected_count} 个 (检出率: {detected_count / max(1, window_idx) * 100:.1f}%)"
    )
    print(
        f"⚡ 平均单窗口耗时: {total_latency_ms / max(1, window_idx):.1f} ms | 总运行时间: {total_test_time:.1f} 秒"
    )
    print("-" * 75)

    if detected_count > 0:
        print("🚩 【异物检出时间段清单】:")
        for item in detection_log:
            if item["foreign_object"] == 1:
                print(
                    f"   • 窗口 #{item['window_idx']:03d} [{item['time_span']}] 定位点: {item['point_coords']} | 置信度: {item['confidence']}"
                )
                print(f"     详情: {item['description']}")
    else:
        print("🟢 全视频所有滑动窗口中未检出异常悬挂异物。")

    log_json_path = os.path.join(args.save_dir, "full_video_foreign_point_report.json")
    with open(log_json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "video_path": video_path,
                "duration_sec": duration_sec,
                "total_windows": window_idx,
                "detected_windows": detected_count,
                "detection_rate": f"{detected_count / max(1, window_idx) * 100:.1f}%",
                "avg_latency_ms": total_latency_ms / max(1, window_idx),
                "log": detection_log,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"\n💾 完整测试报告已保存至: {log_json_path}")
    print("=" * 75 + "\n")


# ==================== 命令行参数 ====================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Qwen3-VL 全视频4帧滑动窗口强制异物检测测试 (Point 准星标定修复版)"
    )
    parser.add_argument(
        "--video",
        type=str,
        default="./弓网典型视频/foreign_1.mp4",
        help="测试视频文件路径",
    )
    parser.add_argument(
        "--window_size", type=int, default=4, help="滑动窗口包含的帧数 (默认: 4 帧)"
    )
    parser.add_argument(
        "--sample_interval_sec",
        type=float,
        default=1.0,
        help="采样间隔 (秒，默认 1.0 秒抽取 1 帧)",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=4,
        help="滑动步长帧数 (默认: 4 帧，即窗口大小，无重叠步进)",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="forced_point",
        choices=["forced_point", "cot_point"],
        help="Prompt策略: forced_point (强制点定位), cot_point (三步思维链点定位)",
    )
    parser.add_argument("--max_tokens", type=int, default=384, help="生成 Token 上限")
    parser.add_argument(
        "--save_all",
        action="store_true",
        help="是否保存所有窗口快照（默认只保存检出异物的窗口快照）",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=DEFAULT_MODEL_PATH,
        help="Qwen3-VL 模型权重路径",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./local_cache/forced_foreign_tests",
        help="结果快照与评测报告保存目录",
    )

    args = parser.parse_args()
    run_full_video_sliding_window_test(args)
