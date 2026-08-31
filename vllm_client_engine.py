#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vLLM 高性能客户端推理引擎 (vllm_client_engine.py)
--------------------------------------------------------------------------------
100% 严格对齐 rtsp_twostage_vlm.py 的级联检测逻辑：
- Stage 1: 4 帧送入 Qwen3-VL 进行火花(spk)与脏污(dirt)全局分类 (PROMPT_STAGE1)
- 级联分支控制：
  1. 若 dirt == 1: 触发极度脏污短路保护，跳过 Stage 2
  2. 若 spk == 1: 触发 Stage 2A 火花 <box> 矩形框定位 (PROMPT_STAGE2_SPARK)
  3. 若 spk == 0 且 dirt == 0: 触发 Stage 2B 悬挂异物 <point> 点定位 (PROMPT_STAGE2_FOREIGN)
- 正则提取 <box> 与 <point>，无误报时输出 Normal
"""

import time
import json
import re
import base64
import cv2
import requests


# ==========================================================
# Stage 1 Prompt: 火花 & 脏污全局分类 (100% 保持 rtsp_twostage_vlm.py 原版)
# ==========================================================
PROMPT_STAGE1 = (
    "你是受电弓故障诊断专家。请观察图像，判断：\n"
    "1. 火花(spk)：有无刺眼强光或电弧放电？\n"
    "2. 脏污(dirt)：画面是否影响了视野，极度模糊、失焦、或被水珠/雾气/污泥遮挡？\n"
    "请先给出一句话的推理分析(reasoning)，描述画面的清晰度，然后再输出JSON。\n"
    "示例 JSON:\n"
    '{"reasoning": "画面整体极度模糊，无法看清受电弓细节，判定为脏污。", "spk": 0, "dirt": 1}'
)

# ==========================================================
# Stage 2A Prompt: 火花 <box> 矩形框定位 (100% 保持 rtsp_twostage_vlm.py 原版)
# ==========================================================
PROMPT_STAGE2_SPARK = (
    "你是一个受电弓和接触网故障诊断专家。\n"
    "图像中发生了大火花或电弧放电。请仔细观察并找出火花的位置。\n"
    "如果找到了火花，请输出一段描述，并在最后必须使用XML格式给出火花的坐标，"
    "格式为：<box>x_min, y_min, x_max, y_max</box>。\n"
    "如果没有找到，请直接回复 'Normal'。"
)

# ==========================================================
# Stage 2B Prompt: 悬挂异物 <point> 点定位 (100% 保持 rtsp_twostage_vlm.py 原版)
# ==========================================================
PROMPT_STAGE2_FOREIGN = (
    "你是一个视觉定位专家"
    "观察受电弓碳滑板（最上方的弧形横梁）及其附近的接触网区域。\n"
    "如果发现不属于受电弓结构的悬挂异物（如编织袋碎片、棉纱线头、鸟窝、羽毛等），"
    "若有异物，输出中心点坐标：<point>x, y</point>（0-1000归一化，原点左上角）。\n"
    "金属部件（弓角、线夹、螺栓、卡扣、铰接关节等）不是异物\n"
    "若无异物，只回复 'Normal'。"
)


class TwoStageVLLMEngine:
    """
    基于 vLLM OpenAI API 服务端的受电弓两阶段级联检测引擎
    逻辑与 rtsp_twostage_vlm.py 100% 严格一致
    """

    def __init__(self, algo_cfg=None):
        if algo_cfg is None:
            algo_cfg = {}
        self.algo_cfg = algo_cfg
        self.api_base = algo_cfg.get("vllm_api_base", "http://127.0.0.1:8000/v1")
        self.model_name = algo_cfg.get("vllm_model_name", "Qwen3-VL-4B-Instruct")
        self.session = requests.Session()
        self.window_id = 0
        self._wait_for_server_ready()

    def _wait_for_server_ready(self, timeout=60):
        """等待 vLLM 服务端启动与预热完毕"""
        models_url = f"{self.api_base}/models"
        t0 = time.time()
        while time.time() - t0 < timeout:
            try:
                resp = self.session.get(models_url, timeout=2.0)
                if resp.status_code == 200:
                    return
            except Exception:
                pass
            print("⏳ [vLLM客户端] 服务端正在进行显存加载与图编译预热，等待就绪中...", flush=True)
            time.sleep(3.0)
        print(f"⚠️ [警告] 超过 {timeout}s 未检测到 vLLM 服务端 ({models_url})，请检查 docker logs vllm-qwen3", flush=True)

    def _frame_to_base64_url(self, frame_bgr, quality=85):
        """将 BGR 图像快速压缩并编码为 Base64 Data URL"""
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

    def infer_stage1_window(self, frames):
        """Stage 1: 4 帧送入进行火花与脏污全局分类"""
        content_items = []
        for f in frames:
            h, w = f.shape[:2]
            if max(h, w) > 640:
                scale = 640.0 / max(h, w)
                f_resized = cv2.resize(f, (int(w * scale), int(h * scale)))
            else:
                f_resized = f
            b64_url = self._frame_to_base64_url(f_resized, quality=75)
            content_items.append({"type": "image_url", "image_url": {"url": b64_url}})
        content_items.append({"type": "text", "text": PROMPT_STAGE1})

        messages = [{"role": "user", "content": content_items}]
        t0 = time.time()
        raw_text = self._call_vllm_chat(messages, max_tokens=128)
        latency_ms = int((time.time() - t0) * 1000)

        # 解析 JSON (严格对齐 rtsp_twostage_vlm.py)
        spk = 0
        dirt = 0
        reasoning = ""
        try:
            parsed = json.loads(raw_text)
            spk = parsed.get("spk", 0)
            dirt = parsed.get("dirt", 0)
            reasoning = parsed.get("reasoning", "")
        except Exception:
            m = re.search(r"\{[^}]+\}", raw_text)
            if m:
                try:
                    parsed = json.loads(m.group(0))
                    spk = parsed.get("spk", 0)
                    dirt = parsed.get("dirt", 0)
                    reasoning = parsed.get("reasoning", "")
                except Exception:
                    pass

        return {
            "spk": spk,
            "dirt": dirt,
            "reasoning": reasoning,
            "raw_output": raw_text,
            "latency_ms": latency_ms,
        }

    def infer_stage2_spark(self, frames):
        """Stage 2A: 火花 <box> 矩形框定位"""
        content_items = []
        for f in frames:
            h, w = f.shape[:2]
            if max(h, w) > 640:
                scale = 640.0 / max(h, w)
                f_resized = cv2.resize(f, (int(w * scale), int(h * scale)))
            else:
                f_resized = f
            b64_url = self._frame_to_base64_url(f_resized, quality=80)
            content_items.append({"type": "image_url", "image_url": {"url": b64_url}})
        content_items.append({"type": "text", "text": PROMPT_STAGE2_SPARK})

        messages = [{"role": "user", "content": content_items}]
        t0 = time.time()
        raw_text = self._call_vllm_chat(messages, max_tokens=128)
        latency_ms = int((time.time() - t0) * 1000)

        box_coords = None
        box_match = re.search(
            r"<box>\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*</box>", raw_text
        )
        if not box_match:
            box_match = re.search(
                r"\((\d+)\s*,\s*(\d+)\)\s*,?\s*\((\d+)\s*,\s*(\d+)\)", raw_text
            )
        if box_match:
            box_coords = list(map(int, box_match.groups()))

        return {
            "box_coords": box_coords,
            "raw_output": raw_text,
            "latency_ms": latency_ms,
        }

    def infer_stage2_foreign(self, frame_or_frames):
        """Stage 2B: 悬挂异物 <point> 点定位"""
        content_items = []
        if isinstance(frame_or_frames, list):
            target_frame = frame_or_frames[-1]
        else:
            target_frame = frame_or_frames

        h, w = target_frame.shape[:2]
        if max(h, w) > 960:
            scale = 960.0 / max(h, w)
            f_resized = cv2.resize(target_frame, (int(w * scale), int(h * scale)))
        else:
            f_resized = target_frame

        b64_url = self._frame_to_base64_url(f_resized, quality=85)
        content_items.append({"type": "image_url", "image_url": {"url": b64_url}})
        content_items.append({"type": "text", "text": PROMPT_STAGE2_FOREIGN})

        messages = [{"role": "user", "content": content_items}]
        t0 = time.time()
        raw_text = self._call_vllm_chat(messages, max_tokens=256)
        latency_ms = int((time.time() - t0) * 1000)

        point_coords = None
        obj_detected = 0
        point_match = re.search(r"<point>\s*(\d+)\s*,\s*(\d+)\s*</point>", raw_text)
        if not point_match:
            point_match = re.search(r"\((\d+)\s*,\s*(\d+)\)", raw_text)
        if point_match:
            point_coords = list(map(int, point_match.groups()))
            obj_detected = 1

        return {
            "foreign_object": obj_detected,
            "point_coords": point_coords,
            "raw_output": raw_text,
            "latency_ms": latency_ms,
        }

    def infer_window(self, frames, timestamp=None):
        """
        100% 严格对齐 rtsp_twostage_vlm.py 级联检测主循环：
        先执行 Stage 1，根据 Stage 1 结果决定 Stage 2 的走向。
        """
        self.window_id += 1
        t_start = time.time()

        # ================= Stage 1 =================
        s1_res = self.infer_stage1_window(frames)
        spk = s1_res["spk"]
        dirt = s1_res["dirt"]
        s1_reasoning = s1_res["reasoning"]

        # ================= Stage 2 (严格级联) =================
        s2_text = ""
        s2_ms = 0
        box_coords = None
        point_coords = None
        obj_detected = 0

        if dirt == 1:
            # 分支 1: 极度脏污短路保护，跳过 Stage 2
            s2_text = "Skipped (极度脏污短路保护)"
            s2_ms = 0
        elif spk == 1:
            # 分支 2: 火花 <box> 定位
            s2_spark_res = self.infer_stage2_spark(frames)
            box_coords = s2_spark_res["box_coords"]
            s2_text = s2_spark_res["raw_output"]
            s2_ms = s2_spark_res["latency_ms"]
        else:
            # 分支 3: 正常画面，触发悬挂异物 <point> 定位
            s2_foreign_res = self.infer_stage2_foreign(frames)
            obj_detected = s2_foreign_res["foreign_object"]
            point_coords = s2_foreign_res["point_coords"]
            s2_text = s2_foreign_res["raw_output"]
            s2_ms = s2_foreign_res["latency_ms"]

        total_latency_ms = int((time.time() - t_start) * 1000)

        # 状态判定 (对齐 rtsp_twostage_vlm.py)
        flags = []
        if spk == 1:
            flags.append("🔥 大火花")
        if dirt == 1:
            flags.append("🌫️ 脏污")
        if box_coords and spk == 1:
            flags.append("🎯 火花定位(Box)")
        elif point_coords and obj_detected == 1:
            flags.append("🎯 悬挂异物(Point)")

        status_str = " | ".join(flags) if flags else "✅ 正常"

        return {
            "spk": spk,
            "dirt": dirt,
            "obj": obj_detected,
            "box_coords": box_coords,
            "foreign_point": point_coords,
            "status_str": status_str,
            "temporal_status": "CONFIRMED" if obj_detected == 1 else "CLEARED",
            "reasoning": s1_reasoning,
            "stage1_raw": s1_res["raw_output"],
            "stage2_raw": s2_text,
            "stage1_latency_ms": s1_res["latency_ms"],
            "stage2_latency_ms": s2_ms,
            "total_latency_ms": total_latency_ms,
        }

    def infer_frame(self, frame, timestamp=None):
        return self.infer_window([frame], timestamp=timestamp)
