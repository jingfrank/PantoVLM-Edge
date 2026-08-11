#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
弓网AI监测算法结果定义与编码 (usrcode.py)
输出字段与 AppConfig.xlsx 输出序号 0~9 严格对齐。
"""

import struct

# 与 AppConfig.xlsx / ai_detector 故障状态机对齐
OUTPUT_FIELD_NAMES = [
    "大火花故障",           # 0
    "受电弓结构异常故障",   # 1
    "画面脏污故障",         # 2
    "受电弓倾斜故障",       # 3
    "悬挂异物故障",         # 4
    "异常飘弓故障",         # 5
    "综合故障标志",         # 6
    "车厢号",               # 7
    "相机号",               # 8
]

OUTPUT_PACK_FORMAT = "<9I"
OUTPUT_FIELD_COUNT = 9

# 字段索引 -> 检测器内部标志属性 / 配置中的故障级别名
FAULT_OUTPUT_SPEC = [
    # (output_index, detector_attr, event_name_in_fault_levels)
    (0, "spark_alarmed", "大火花"),
    (1, "structure_alarmed", "受电弓结构异常"),
    (2, "dirty_alarmed", "画面脏污"),
    (3, "tilt_alarmed", "受电弓倾斜"),
    (4, "foreign_alarmed", "悬挂异物"),
    (5, "float_alarmed", "异常飘弓"),
]


def build_monitor_result_values(detectors, fault_levels=None):
    """汇总各检测器状态，无报警时返回 None，不对外输出。"""
    fault_levels = fault_levels or {}
    flags = [0] * 6
    car_number = 0
    camera_id = 0

    for detector in detectors:
        if car_number == 0:
            try:
                car_number = int(getattr(detector, "car_number", 0) or 0)
            except (TypeError, ValueError):
                car_number = 0
        if camera_id == 0:
            try:
                camera_id = int(getattr(detector, "camera_id", 0) or 0)
            except (TypeError, ValueError):
                camera_id = 0

        for out_idx, attr, event_name in FAULT_OUTPUT_SPEC:
            active = int(bool(getattr(detector, attr, False)))
            flags[out_idx] = max(flags[out_idx], active)

    summary = 1 if any(flags) else 0
    if summary == 0:
        return None

    return flags + [summary, car_number, camera_id]


def encode_monitor_outputs(values):
    """编码为裸 UINT32 字节流（无报警返回 0 长度）。"""
    if values is None:
        return 0, b""
    if len(values) != OUTPUT_FIELD_COUNT:
        raise ValueError(f"期望 {OUTPUT_FIELD_COUNT} 个输出字段, 实际 {len(values)}")
    clamped = [max(0, int(v)) & 0xFFFFFFFF for v in values]
    payload = struct.pack(OUTPUT_PACK_FORMAT, *clamped)
    return len(payload), payload


def get_monitor_output_snapshot(detectors, fault_levels=None):
    values = build_monitor_result_values(detectors, fault_levels)
    return encode_monitor_outputs(values)
