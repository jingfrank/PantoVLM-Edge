#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qwen3-VL-4B 统一动静分离 + 方案1时序一致性滤波综合检测引擎 (twostage_vlm_engine.py)
--------------------------------------------------------------------------------
- Stage 1 (动态时序全局诊断): 采用 4 帧 720P 视频流时序理解，快速完成火花(spk)与脏污(dirt)分类（~500ms）
- Stage 2 (细粒度静态高精定位): 针对悬挂异物(foreign_object)，切换为最后一帧高清原图单图推理，消除微小目标漏检
- 时序一致性滤波追踪器 (TemporalConsistencyTracker): 跨时序窗口空间距离与连续性校验，彻底过滤单帧偶发误报
- 兼容 infer_frame(单帧调用) 与 infer_window(滑动窗口视频流调用)
"""

import time
import json
import re
import math
import cv2
import torch
import numpy as np
from PIL import Image


class ForeignObjectCandidate:
    """单个异物候选点跟踪实体"""
    def __init__(self, pt, timestamp, window_id):
        self.point = pt  # (x, y) 归一化坐标 (0-1000)
        self.first_seen_ts = timestamp
        self.last_seen_ts = timestamp
        self.hit_count = 1
        self.window_ids = [window_id]
        self.is_confirmed = False

    def distance_to(self, other_pt):
        dx = self.point[0] - other_pt[0]
        dy = self.point[1] - other_pt[1]
        return math.sqrt(dx * dx + dy * dy)

    def update(self, pt, timestamp, window_id):
        # 移动平均平滑坐标
        self.point = (
            int(round(self.point[0] * 0.4 + pt[0] * 0.6)),
            int(round(self.point[1] * 0.4 + pt[1] * 0.6)),
        )
        self.last_seen_ts = timestamp
        self.hit_count += 1
        self.window_ids.append(window_id)


class TemporalConsistencyTracker:
    """
    【时序一致性追踪器】
    管理异物候选点生命周期，过滤单帧偶发误报，确认持续性真阳性故障。
    """
    def __init__(self, match_dist=80.0, confirm_thresh=2, ttl=6.0):
        self.match_dist = match_dist
        self.confirm_thresh = confirm_thresh
        self.ttl = ttl
        self.candidates = []

    def update(self, raw_point, current_ts, window_id):
        """
        输入单帧检出的原始点坐标 raw_point: (x, y) or None
        返回: (status, confirmed_point, candidate_info, expired_log)
        status:
          - "CONFIRMED": 经过时序连续确认的真实故障
          - "SUSPECT": 首次或正在观察中的疑似候选点 (暂不误报)
          - "CLEARED": 本次无异物或误报已自动平息
        """
        # 1. 先清理超时的候选点 (未持续命中的单帧瞬间误报)
        active_candidates = []
        expired_candidates = []
        for cand in self.candidates:
            if (current_ts - cand.last_seen_ts) <= self.ttl:
                active_candidates.append(cand)
            else:
                expired_candidates.append(cand)
        self.candidates = active_candidates

        expired_log = []
        for exp in expired_candidates:
            if not exp.is_confirmed:
                expired_log.append(f"点 {exp.point} (仅命中 {exp.hit_count} 次)")

        # 2. 如果本窗口无检出
        if raw_point is None:
            return "CLEARED", None, None, expired_log

        # 3. 尝试与现有候选点进行空间邻近匹配
        best_match = None
        min_d = float("inf")
        for cand in self.candidates:
            d = cand.distance_to(raw_point)
            if d < self.match_dist and d < min_d:
                min_d = d
                best_match = cand

        if best_match is not None:
            # 命中已有候选点，累计时序一致性
            best_match.update(raw_point, current_ts, window_id)
            if best_match.hit_count >= self.confirm_thresh:
                best_match.is_confirmed = True
                return "CONFIRMED", best_match.point, best_match, expired_log
            else:
                return "SUSPECT", best_match.point, best_match, expired_log
        else:
            # 首次检出新候选点，建立追踪观察
            new_cand = ForeignObjectCandidate(raw_point, current_ts, window_id)
            self.candidates.append(new_cand)
            if self.confirm_thresh <= 1:
                new_cand.is_confirmed = True
                return "CONFIRMED", new_cand.point, new_cand, expired_log
            else:
                return "SUSPECT", new_cand.point, new_cand, expired_log


class TwoStageVLMEngine:
    def __init__(self, algo_cfg):
        self.algo_cfg = algo_cfg
        model_path = algo_cfg.get(
            "downloaded_model_path", "./models/Qwen3-VL-4B-Instruct"
        )

        # 运行时平台检测与显存配额保护
        try:
            from platform_utils import get_gpu_memory_fraction

            mem_frac = get_gpu_memory_fraction()
        except ImportError:
            mem_frac = 0.50

        if torch.cuda.is_available():
            torch.cuda.set_per_process_memory_fraction(mem_frac, device=0)
            print(f"[VLM-Engine] 🛡️ 已启用 {mem_frac * 100}% GPU 显存配额保护")

        print("[VLM-Engine] 正在向 GPU 加载模型 (FP16)...")
        from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

        self.processor = AutoProcessor.from_pretrained(model_path)

        # 🎯 视频流分辨率优化（720P 均衡档，供 Stage 1 视频推理使用）
        if (
            hasattr(self.processor, "video_processor")
            and self.processor.video_processor is not None
        ):
            self.processor.video_processor.size = {
                "shortest_edge": 64 * 32 * 32,
                "longest_edge": int(960 * 720 * 2),
            }

        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path,
            dtype=torch.float16,
            device_map="cuda:0",
            low_cpu_mem_usage=True,
        )
        self.model.eval()
        print("[VLM-Engine] ✅ 动静分离 + 时序一致性滤波模型加载成功！")

        # 🎯 方案 1: 时序一致性滤波追踪器初始化
        confirm_hits = algo_cfg.get("foreign_confirm_hits", 2)
        match_dist = algo_cfg.get("foreign_match_dist", 80.0)
        ttl = algo_cfg.get("foreign_candidate_ttl", 6.0)
        self.tracker = TemporalConsistencyTracker(
            match_dist=match_dist, confirm_thresh=confirm_hits, ttl=ttl
        )
        self.window_id = 0

        # Stage 1: 火花 & 脏污全局分类 (动态时序视频/单图)
        self.prompt_stage1 = (
            "你是受电弓故障诊断专家。请观察图像，判断：\n"
            "1. 火花(spk)：有无刺眼强光或电弧放电？\n"
            "2. 脏污(dirt)：画面是否影响了视野，极度模糊、失焦、或被水珠/雾气/污泥遮挡？\n"
            "请先给出一句话的推理分析(reasoning)，描述画面的清晰度，然后再输出JSON。\n"
            "示例 JSON:\n"
            '{"reasoning": "画面整体极度模糊，无法看清受电弓细节，判定为脏污。", "spk": 0, "dirt": 1}'
        )

        # Stage 2A: 火花 <box> 矩形框定位
        self.prompt_stage2_spark = (
            "你是一个受电弓和接触网故障诊断专家。\n"
            "图像中发生了大火花或电弧放电。请仔细观察并找出火花的位置。\n"
            "如果找到了火花，请输出一段描述，并在最后必须使用XML格式给出火花的坐标，"
            "格式为：<box>x_min, y_min, x_max, y_max</box>。\n"
            "如果没有找到，请直接回复 'Normal'。"
        )

        # Stage 2B: 悬挂异物 <point> 定位 (高清单图)
        self.prompt_stage2_foreign = (
            "你是一个受电弓与接触网视觉定位专家。\n"
            "请仔细观察受电弓碳滑板（最上方的弧形横梁）及其正上方的接触网区域。\n"
            "如果发现任何不属于受电弓本体结构的附着物或悬挂物（如附着在滑板上的白色羽毛/絮状物、塑料薄膜、编织袋碎片、棉纱线头等），"
            "请使用XML格式给出该异物的中心点坐标，格式为：<point>x, y</point>（0到1000归一化，原点在左上角）。\n\n"
            "【负样本约束】受电弓本体金属部件（弓角、线夹、螺栓、卡扣、接头等）绝不是异物！列车经过站台、灯光密集区、高压线密集区时，因高速运动产生的光源拖影也不是异物。若画面正常无外来异物，请直接回复 'Normal'。"
        )

    def _infer_single_image(self, pil_img, prompt, max_tokens=128):
        """【静态高清单图推理】单图送入，保留 100% 原始像素细节"""
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
            generated_ids = self.model.generate(
                **inputs, max_new_tokens=max_tokens, do_sample=False
            )
        latency_ms = (time.time() - t0) * 1000

        input_len = inputs["input_ids"].shape[1]
        new_ids = generated_ids[:, input_len:]
        output_text = self.processor.batch_decode(
            new_ids, skip_special_tokens=True
        )[0].strip()
        return output_text, latency_ms

    def _infer_video(self, pil_frames, prompt, max_tokens=128):
        """【动态时序视频推理】4 帧 720P 视频送入，具备动态时序分析能力"""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": pil_frames, "fps": 1.0},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            text=[text], videos=[pil_frames], padding=True, return_tensors="pt"
        )
        inputs = {k: v.to("cuda") for k, v in inputs.items()}

        t0 = time.time()
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs, max_new_tokens=max_tokens, do_sample=False
            )
        latency_ms = (time.time() - t0) * 1000

        input_len = inputs["input_ids"].shape[1]
        new_ids = generated_ids[:, input_len:]
        output_text = self.processor.batch_decode(
            new_ids, skip_special_tokens=True
        )[0].strip()
        return output_text, latency_ms

    def infer_window(self, frames_bgr_list):
        """
        【动静分离两阶段 + 时序一致性滤波滑动窗口推理】
        - Stage 1: 4 帧 720P 视频流时序动态诊断 (火花/脏污)
        - Stage 2: 最后一帧高清单图定位 (悬挂异物)
        - 时序滤波: 跨窗口校验异物位置连续性，过滤单帧误报
        """
        if not frames_bgr_list:
            return {}

        self.window_id += 1
        pil_frames = [
            Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
            for f in frames_bgr_list
        ]
        last_pil_frame = pil_frames[-1]

        # ================= Stage 1 (动态时序) =================
        s1_text, s1_ms = self._infer_video(
            pil_frames, self.prompt_stage1, max_tokens=128
        )

        try:
            s1_res = json.loads(s1_text)
        except Exception:
            m = re.search(r"\{[^}]+\}", s1_text)
            s1_res = json.loads(m.group()) if m else {}

        spk = s1_res.get("spk", 0)
        dirt = s1_res.get("dirt", 0)
        reasoning = s1_res.get("reasoning", "")

        box_coords = None
        raw_point = None
        point_coords = None
        obj = 0
        s2_text = ""
        s2_ms = 0

        # ================= Stage 2 (细粒度定位) =================
        if dirt == 1:
            s2_text = "Skipped (极度脏污短路保护)"
        elif spk == 1:
            s2_text, s2_ms = self._infer_video(
                pil_frames, self.prompt_stage2_spark, max_tokens=128
            )
            box_pattern1 = (
                r"<box>\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*</box>"
            )
            box_pattern2 = r"\((\d+)\s*,\s*(\d+)\)\s*,?\s*\((\d+)\s*,\s*(\d+)\)"
            box_match = re.search(box_pattern1, s2_text)
            if not box_match:
                box_match = re.search(box_pattern2, s2_text)
            if box_match:
                raw_coords = list(map(int, box_match.groups()))
                if len(raw_coords) == 4:
                    box_coords = tuple(raw_coords)
        else:
            # 🎯 最后一帧高清单图定位
            s2_text, s2_ms = self._infer_single_image(
                last_pil_frame, self.prompt_stage2_foreign, max_tokens=128
            )
            point_pattern1 = r"<point>\s*(\d+)\s*,\s*(\d+)\s*</point>"
            point_pattern2 = r"\((\d+)\s*,\s*(\d+)\)"
            point_match = re.search(point_pattern1, s2_text)
            if not point_match:
                point_match = re.search(point_pattern2, s2_text)
            if point_match and "Normal" not in s2_text:
                coords = tuple(map(int, point_match.groups()))
                if not (coords[0] == 0 and coords[1] == 0):
                    raw_point = coords

        # ================= 时序一致性滤波校验 =================
        now = time.time()
        status, confirmed_pt, cand, expired_log = self.tracker.update(
            raw_point, now, self.window_id
        )

        if status == "CONFIRMED":
            obj = 1
            point_coords = confirmed_pt
        elif status == "SUSPECT":
            # 疑似观察中，点坐标供调试，但不触发正式故障告警
            point_coords = cand.point if cand else raw_point
            obj = 0
        else:
            obj = 0

        return {
            "spk": spk,
            "dirt": dirt,
            "obj": obj,
            "str": 0,
            "tilt": 0,
            "flt": 0,
            "box": box_coords,
            "point": point_coords,
            "raw_point": raw_point,
            "reasoning": reasoning,
            "s1_text": s1_text,
            "s2_text": s2_text,
            "s1_ms": s1_ms,
            "s2_ms": s2_ms,
            "temporal_status": status,
            "temporal_cand": cand,
            "temporal_expired_log": expired_log,
        }

    def infer_frame(self, frame_input):
        """
        兼容单帧调用接口：
        如果传入的是单张 BGR 图像 (ndarray)，封装为单帧推理并经由时序追踪器校验；
        如果传入的是图像列表 (list of ndarrays)，则调用 infer_window。
        """
        if isinstance(frame_input, (list, tuple)):
            return self.infer_window(frame_input)

        frame_bgr = frame_input
        self.window_id += 1
        pil_img = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))

        # Stage 1 单图
        s1_text, s1_ms = self._infer_single_image(
            pil_img, self.prompt_stage1, max_tokens=128
        )

        try:
            s1_res = json.loads(s1_text)
        except Exception:
            m = re.search(r"\{[^}]+\}", s1_text)
            s1_res = json.loads(m.group()) if m else {}

        spk = s1_res.get("spk", 0)
        dirt = s1_res.get("dirt", 0)
        reasoning = s1_res.get("reasoning", "")

        box_coords = None
        raw_point = None
        point_coords = None
        obj = 0
        s2_text = ""
        s2_ms = 0

        # Stage 2 细粒度定位
        if dirt == 1:
            s2_text = "Skipped"
        elif spk == 1:
            s2_text, s2_ms = self._infer_single_image(
                pil_img, self.prompt_stage2_spark, max_tokens=128
            )
            box_pattern1 = (
                r"<box>\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*</box>"
            )
            box_pattern2 = r"\((\d+)\s*,\s*(\d+)\)\s*,?\s*\((\d+)\s*,\s*(\d+)\)"
            box_match = re.search(box_pattern1, s2_text)
            if not box_match:
                box_match = re.search(box_pattern2, s2_text)
            if box_match:
                raw_coords = list(map(int, box_match.groups()))
                if len(raw_coords) == 4:
                    box_coords = tuple(raw_coords)
        else:
            s2_text, s2_ms = self._infer_single_image(
                pil_img, self.prompt_stage2_foreign, max_tokens=128
            )
            point_pattern1 = r"<point>\s*(\d+)\s*,\s*(\d+)\s*</point>"
            point_pattern2 = r"\((\d+)\s*,\s*(\d+)\)"
            point_match = re.search(point_pattern1, s2_text)
            if not point_match:
                point_match = re.search(point_pattern2, s2_text)
            if point_match and "Normal" not in s2_text:
                coords = tuple(map(int, point_match.groups()))
                if not (coords[0] == 0 and coords[1] == 0):
                    raw_point = coords

        now = time.time()
        status, confirmed_pt, cand, expired_log = self.tracker.update(
            raw_point, now, self.window_id
        )

        if status == "CONFIRMED":
            obj = 1
            point_coords = confirmed_pt
        elif status == "SUSPECT":
            point_coords = cand.point if cand else raw_point
            obj = 0
        else:
            obj = 0

        return {
            "spk": spk,
            "dirt": dirt,
            "obj": obj,
            "str": 0,
            "tilt": 0,
            "flt": 0,
            "box": box_coords,
            "point": point_coords,
            "raw_point": raw_point,
            "reasoning": reasoning,
            "s1_text": s1_text,
            "s2_text": s2_text,
            "s1_ms": s1_ms,
            "s2_ms": s2_ms,
            "temporal_status": status,
            "temporal_cand": cand,
            "temporal_expired_log": expired_log,
        }
