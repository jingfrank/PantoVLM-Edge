#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vLLM 高性能客户端推理引擎 (vllm_client_engine.py)
--------------------------------------------------------------------------------
严格对齐 twostage_vlm_engine.py 的全部逻辑：
- Stage 1: 4 帧图像送入 Qwen3-VL 进行火花(spk)与脏污(dirt)全局分类
- 级联分支控制：
  1. dirt == 1: 极度脏污短路保护，跳过 Stage 2
  2. spk == 1: Stage 2A 火花 <box> 矩形框定位
  3. spk == 0 且 dirt == 0: Stage 2B 悬挂异物 <point> 点定位
- TemporalConsistencyTracker: 跨时序窗口空间距离与连续性校验，过滤单帧偶发误报
- 返回字段与 twostage_vlm_engine.py 100% 对齐，兼容 ai_detector.py
"""

import time
import json
import re
import math
import base64
import cv2
import requests


# ==============================================================================
# 时序一致性追踪器（与 twostage_vlm_engine.py 完全相同）
# ==============================================================================

class ForeignObjectCandidate:
    """单个异物候选点跟踪实体"""
    def __init__(self, pt, timestamp, window_id):
        self.point = pt
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
    def __init__(self, match_dist=80.0, confirm_thresh=2, ttl=30.0):
        self.match_dist = match_dist
        self.confirm_thresh = confirm_thresh
        self.ttl = ttl
        self.candidates = []

    def update(self, raw_point, current_ts, window_id):
        # 1. 清理超时候选点
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

        # 2. 本窗口无检出，不再一刀切清空候选，让候选靠 TTL 自然过期
        #    这样真实异物即使偶尔漏检一个窗口，也不会前功尽弃
        if raw_point is None:
            return "CLEARED", None, None, expired_log

        # 3. 空间邻近匹配
        best_match = None
        min_d = float("inf")
        for cand in self.candidates:
            d = cand.distance_to(raw_point)
            if d < self.match_dist and d < min_d:
                min_d = d
                best_match = cand

        if best_match is not None:
            best_match.update(raw_point, current_ts, window_id)
            if best_match.hit_count >= self.confirm_thresh:
                best_match.is_confirmed = True
                return "CONFIRMED", best_match.point, best_match, expired_log
            else:
                return "SUSPECT", best_match.point, best_match, expired_log
        else:
            new_cand = ForeignObjectCandidate(raw_point, current_ts, window_id)
            self.candidates.append(new_cand)
            if self.confirm_thresh <= 1:
                new_cand.is_confirmed = True
                return "CONFIRMED", new_cand.point, new_cand, expired_log
            else:
                return "SUSPECT", new_cand.point, new_cand, expired_log


# ==============================================================================
# Prompts（与 twostage_vlm_engine.py 对齐，使用更详细的负样本约束版）
# ==============================================================================

PROMPT_STAGE1 = (
    "你是受电弓故障诊断专家。请观察图像，判断：\n"
    "1. 火花(spk)：有无刺眼强光或电弧放电？\n"
    "2. 脏污(dirt)：画面是否影响了视野，极度模糊、失焦、或被水珠/雾气/污泥遮挡？\n"
    "请先给出一句话的推理分析(reasoning)，描述画面的清晰度，然后再输出JSON。\n"
    "示例 JSON:\n"
    '{"reasoning": "画面整体极度模糊，无法看清受电弓细节，判定为脏污。", "spk": 0, "dirt": 1}'
)

PROMPT_STAGE2_SPARK = (
    "你是一个受电弓和接触网故障诊断专家。\n"
    "图像中发生了大火花或电弧放电。请仔细观察并找出火花的位置。\n"
    "如果找到了火花，请输出一段描述，并在最后必须使用XML格式给出火花的坐标，"
    "格式为：<box>x_min, y_min, x_max, y_max</box>。\n"
    "如果没有找到，请直接回复 'Normal'。"
)

PROMPT_STAGE2_FOREIGN = (
    "你是一个受电弓与接触网视觉定位专家。\n"
    "请仔细观察受电弓碳滑板（最上方的弧形横梁）及其正上方的接触网区域。\n"
    "如果发现任何不属于受电弓本体结构的附着物或悬挂物（如附着在滑板上的白色羽毛/絮状物、塑料薄膜、编织袋碎片、棉纱线头等），"
    "请使用XML格式给出该异物的中心点坐标，格式为：<point>x, y</point>（0到1000归一化，原点在左上角）。\n\n"
    "【负样本约束】受电弓本体金属部件（弓角、线夹、螺栓、卡扣、接头等）绝不是异物！"
    "列车经过站台、灯光密集区、高压线密集区时，因高速运动产生的光源拖影也不是异物。"
    "若画面正常无外来异物，请直接回复 'Normal'。"
)


# ==============================================================================
# TwoStageVLLMEngine
# ==============================================================================

class TwoStageVLLMEngine:
    """
    基于 vLLM OpenAI API 服务端的受电弓两阶段级联检测引擎
    逻辑与 twostage_vlm_engine.py 100% 严格对齐，含时序一致性追踪器
    """

    def __init__(self, algo_cfg=None):
        if algo_cfg is None:
            algo_cfg = {}
        self.algo_cfg = algo_cfg
        self.api_base = algo_cfg.get("vllm_api_base", "http://127.0.0.1:8000/v1")
        self.model_name = algo_cfg.get("vllm_model_name", "Qwen3-VL-4B-Instruct")
        self.session = requests.Session()
        self.window_id = 0

        # 时序一致性追踪器（与 twostage_vlm_engine.py 完全相同的参数）
        confirm_hits = algo_cfg.get("foreign_confirm_hits", 2)
        match_dist = algo_cfg.get("foreign_match_dist", 80.0)
        ttl = algo_cfg.get("foreign_candidate_ttl", 6.0)
        self.tracker = TemporalConsistencyTracker(
            match_dist=match_dist, confirm_thresh=confirm_hits, ttl=ttl
        )

        self._wait_for_server_ready()

    def _wait_for_server_ready(self, timeout=120):
        """等待 vLLM 服务端启动与预热完毕"""
        models_url = f"{self.api_base}/models"
        t0 = time.time()
        while time.time() - t0 < timeout:
            try:
                resp = self.session.get(models_url, timeout=2.0)
                if resp.status_code == 200:
                    print("[vLLM客户端] ✅ vLLM 服务端就绪！", flush=True)
                    return
            except Exception:
                pass
            print("⏳ [vLLM客户端] 等待 vLLM 服务端预热...", flush=True)
            time.sleep(3.0)
        print(f"⚠️ [警告] 超过 {timeout}s 未检测到 vLLM 服务端 ({models_url})", flush=True)

    def _frame_to_base64_url(self, frame_bgr, quality=85):
        """将 BGR 图像压缩为 Base64 Data URL"""
        ret, buf = cv2.imencode(
            ".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        )
        if not ret:
            raise ValueError("图像 JPEG 编码失败")
        b64_str = base64.b64encode(buf).decode("utf-8")
        return f"data:image/jpeg;base64,{b64_str}"

    def _call_vllm_chat(self, messages, max_tokens=128, temperature=0.01):
        """调用 vLLM OpenAI Chat Completions API"""
        url = f"{self.api_base}/chat/completions"
        payload = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        resp = self.session.post(url, json=payload, timeout=30.0)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"].strip()

    def _build_image_messages(self, frames_bgr, prompt, max_side=640, quality=75):
        """将多帧 BGR 图像 + prompt 构建为 OpenAI messages 格式"""
        content_items = []
        for f in frames_bgr:
            h, w = f.shape[:2]
            if max(h, w) > max_side:
                scale = max_side / max(h, w)
                f = cv2.resize(f, (int(w * scale), int(h * scale)))
            b64_url = self._frame_to_base64_url(f, quality=quality)
            content_items.append({"type": "image_url", "image_url": {"url": b64_url}})
        content_items.append({"type": "text", "text": prompt})
        return [{"role": "user", "content": content_items}]

    def infer_window(self, frames_bgr_list, timestamp=None):
        """
        【两阶段级联 + 时序一致性滤波】主推理接口
        返回字段与 twostage_vlm_engine.py 100% 对齐，ai_detector.py 直接兼容。
        """
        if not frames_bgr_list:
            return {}

        self.window_id += 1

        # ================= Stage 1: 火花 & 脏污全局分类 =================
        t0 = time.time()
        messages_s1 = self._build_image_messages(
            frames_bgr_list, PROMPT_STAGE1, max_side=640, quality=75
        )
        s1_text = self._call_vllm_chat(messages_s1, max_tokens=128)
        s1_ms = int((time.time() - t0) * 1000)

        spk, dirt, reasoning = 0, 0, ""
        try:
            parsed = json.loads(s1_text)
            spk = parsed.get("spk", 0)
            dirt = parsed.get("dirt", 0)
            reasoning = parsed.get("reasoning", "")
        except Exception:
            m = re.search(r"\{[^}]+\}", s1_text)
            if m:
                try:
                    parsed = json.loads(m.group(0))
                    spk = parsed.get("spk", 0)
                    dirt = parsed.get("dirt", 0)
                    reasoning = parsed.get("reasoning", "")
                except Exception:
                    pass

        # ================= Stage 2: 细粒度定位（严格级联） =================
        box_coords = None
        raw_point = None
        s2_text = ""
        s2_ms = 0

        if dirt == 1:
            # 分支 1: 极度脏污短路保护
            s2_text = "Skipped (极度脏污短路保护)"

        elif spk == 1:
            # 分支 2: 火花 <box> 定位（送所有帧，640p）
            t0 = time.time()
            messages_s2 = self._build_image_messages(
                frames_bgr_list, PROMPT_STAGE2_SPARK, max_side=640, quality=80
            )
            s2_text = self._call_vllm_chat(messages_s2, max_tokens=128)
            s2_ms = int((time.time() - t0) * 1000)

            box_match = re.search(
                r"<box>\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*</box>", s2_text
            )
            if not box_match:
                box_match = re.search(
                    r"\((\d+)\s*,\s*(\d+)\)\s*,?\s*\((\d+)\s*,\s*(\d+)\)", s2_text
                )
            if box_match:
                raw_coords = list(map(int, box_match.groups()))
                if len(raw_coords) == 4:
                    box_coords = tuple(raw_coords)

        else:
            # 分支 3: 悬挂异物 <point> 定位（仅最后一帧，960p 高清）
            last_frame = frames_bgr_list[-1]
            t0 = time.time()
            messages_s2 = self._build_image_messages(
                [last_frame], PROMPT_STAGE2_FOREIGN, max_side=960, quality=85
            )
            s2_text = self._call_vllm_chat(messages_s2, max_tokens=256)
            s2_ms = int((time.time() - t0) * 1000)

            # 与 twostage_vlm_engine.py 完全相同的防误报过滤
            point_match = re.search(r"<point>\s*(\d+)\s*,\s*(\d+)\s*</point>", s2_text)
            if not point_match:
                point_match = re.search(r"\((\d+)\s*,\s*(\d+)\)", s2_text)
            if point_match and "Normal" not in s2_text:
                coords = tuple(map(int, point_match.groups()))
                if not (coords[0] == 0 and coords[1] == 0):  # 过滤 (0,0) 误报
                    raw_point = coords

        # ================= 时序一致性滤波校验 =================
        now = time.time() if timestamp is None else float(timestamp)
        status, confirmed_pt, cand, expired_log = self.tracker.update(
            raw_point, now, self.window_id
        )

        obj = 0
        point_coords = None
        if status == "CONFIRMED":
            obj = 1
            point_coords = confirmed_pt
        elif status == "SUSPECT":
            point_coords = cand.point if cand else raw_point
            obj = 0
        else:
            obj = 0

        return {
            # 故障标志（ai_detector.py 直接读取）
            "spk": spk,
            "dirt": dirt,
            "obj": obj,
            "str": 0,
            "tilt": 0,
            "flt": 0,
            # 坐标结果
            "box": box_coords,
            "point": point_coords,
            "raw_point": raw_point,
            # 推理文本与耗时（ai_detector.py 打印日志用）
            "reasoning": reasoning,
            "s1_text": s1_text,
            "s2_text": s2_text,
            "s1_ms": s1_ms,
            "s2_ms": s2_ms,
            # 时序追踪器状态（ai_detector.py 读取 temporal_cand.hit_count）
            "temporal_status": status,
            "temporal_cand": cand,
            "temporal_expired_log": expired_log,
        }

    def infer_frame(self, frame, timestamp=None):
        """兼容单帧调用接口"""
        if isinstance(frame, (list, tuple)):
            return self.infer_window(frame, timestamp=timestamp)
        return self.infer_window([frame], timestamp=timestamp)
