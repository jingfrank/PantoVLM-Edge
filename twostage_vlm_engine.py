#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qwen3-VL-4B 统一两阶段综合检测引擎 (twostage_vlm_engine.py)
用于 Titan RTX 和 Jetson Orin，提取动态路由推理逻辑为复用模块。
"""

import time
import json
import re
import cv2
import torch
from PIL import Image

class TwoStageVLMEngine:
    def __init__(self, algo_cfg):
        self.algo_cfg = algo_cfg
        model_path = algo_cfg.get("downloaded_model_path", "./models/Qwen3-VL-4B-Instruct")
        
        # 运行时平台检测与显存配额保护
        try:
            from platform_utils import get_gpu_memory_fraction
            mem_frac = get_gpu_memory_fraction()
        except ImportError:
            mem_frac = 0.50

        if torch.cuda.is_available():
            torch.cuda.set_per_process_memory_fraction(mem_frac, device=0)
            print(f"[VLM-Engine] 🛡️ 已启用 {mem_frac*100}% GPU 显存配额保护")

        print("[VLM-Engine] 正在向 GPU 加载模型 (FP16)...")
        from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
        
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path,
            dtype=torch.float16,
            device_map="cuda:0",
            low_cpu_mem_usage=True,
        )
        self.model.eval()
        print("[VLM-Engine] ✅ 模型加载成功！")

        self.prompt_stage1 = (
            "你是受电弓故障诊断专家。请观察图像，判断：\n"
            "1. 火花(spk)：有无刺眼强光或电弧放电？\n"
            "2. 脏污(dirt)：画面是否极度模糊、失焦、或被水滴/污泥遮挡？注意：不要推测模糊的原因！只要画面模糊到难以辨认受电弓细节，就必须判定为脏污(dirt=1)。\n"
            "请先给出一句话的推理分析(reasoning)，描述画面的清晰度，然后再输出JSON。\n"
            "示例 JSON:\n"
            "{\"reasoning\": \"画面整体极度模糊，无法看清受电弓细节，判定为脏污。\", \"spk\": 0, \"dirt\": 1}"
        )

        self.prompt_stage2_spark = (
            "你是一个受电弓和接触网故障诊断专家。\n"
            "图像中发生了大火花或电弧放电。请仔细观察并找出火花的位置。\n"
            "如果找到了火花，请输出一段描述，并在最后必须使用XML格式给出火花的坐标，"
            "格式为：<box>x_min, y_min, x_max, y_max</box>。\n"
            "如果没有找到，请直接回复 'Normal'。"
        )

        self.prompt_stage2_foreign = (
            "你是一个视觉定位专家。\n"
            "请仔细观察受电弓碳滑板（最上方的弧形横梁）及其附近的接触网区域。\n"
            "如果发现任何不属于受电弓本体结构的附着物或悬挂物（如缠绕的塑料薄膜、编织袋碎片、棉纱线头等），"
            "请指出该异物最核心的中心点位置，并在最后使用XML格式给出点坐标，格式为：<point>x, y</point>（坐标归一化在 0 到 1000 之间）。\n\n"
            "【什么不是异物】受电弓本身的金属部件（弓角、线夹、螺栓、卡扣、铰接关节、弹簧座、集电靴接头等）绝不是异物！\n"
            "如果画面中只有受电弓本体结构而没有外来异物，请直接回复 'Normal'。"
        )

    def _infer_one(self, pil_img, prompt, max_tokens=128):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_img},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            text=[text], images=[pil_img], padding=True, return_tensors="pt"
        )
        inputs = {k: v.to("cuda") for k, v in inputs.items()}

        t0 = time.time()
        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=max_tokens)
        
        input_len = inputs["input_ids"].shape[1]
        new_ids = generated_ids[:, input_len:]
        output_text = self.processor.batch_decode(new_ids, skip_special_tokens=True)[0].strip()
        return output_text

    def infer_frame(self, frame_bgr):
        """
        对单帧图像进行推理，返回结构化状态字典。
        返回格式:
        {
            "spk": 0/1,
            "dirt": 0/1,
            "obj": 0/1,
            "str": 0/1,
            "tilt": 0/1,
            "flt": 0/1,
            "box": (nx1, ny1, nx2, ny2) or None,
            "point": (nx, ny) or None,
            "reasoning": str
        }
        """
        pil_img = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        
        # Stage 1
        s1_text = self._infer_one(pil_img, self.prompt_stage1)
        
        try:
            s1_res = json.loads(s1_text)
        except Exception:
            m = re.search(r"\{[^}]+\}", s1_text)
            s1_res = json.loads(m.group()) if m else {}

        spk = s1_res.get("spk", 0)
        dirt = s1_res.get("dirt", 0)
        reasoning = s1_res.get("reasoning", "")
        
        box_coords = None
        point_coords = None
        obj = 0

        # Stage 2 Dynamic Routing
        if dirt == 1:
            pass # Skip stage 2
        elif spk == 1:
            s2_text = self._infer_one(pil_img, self.prompt_stage2_spark)
            box_pattern1 = r'<box>\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*</box>'
            box_pattern2 = r'\((\d+)\s*,\s*(\d+)\)\s*,?\s*\((\d+)\s*,\s*(\d+)\)'
            box_match = re.search(box_pattern1, s2_text)
            if not box_match:
                box_match = re.search(box_pattern2, s2_text)
            if box_match:
                raw_coords = list(map(int, box_match.groups()))
                if len(raw_coords) == 4:
                    box_coords = tuple(raw_coords)
        else:
            s2_text = self._infer_one(pil_img, self.prompt_stage2_foreign)
            point_pattern1 = r'<point>\s*(\d+)\s*,\s*(\d+)\s*</point>'
            point_pattern2 = r'\((\d+)\s*,\s*(\d+)\)'
            point_match = re.search(point_pattern1, s2_text)
            if not point_match:
                point_match = re.search(point_pattern2, s2_text)
            if point_match:
                point_coords = tuple(map(int, point_match.groups()))
                obj = 1  # Detect foreign object

        return {
            "spk": spk,
            "dirt": dirt,
            "obj": obj,
            "str": 0,    # Placeholder for structure anomaly
            "tilt": 0,   # Placeholder for tilt
            "flt": 0,    # Placeholder for abnormal float
            "box": box_coords,
            "point": point_coords,
            "reasoning": reasoning
        }
