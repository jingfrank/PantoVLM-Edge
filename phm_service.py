#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""弓网监测算法 PHM 心跳与故障上报封装。"""

import os

from common import PhmBackgroundReporter, HOST_SOCKET_ADDR_DEFAULT
from usrcode import OUTPUT_FIELD_NAMES, get_monitor_output_snapshot


def _parse_host_addrs(phm_cfg):
    host_addrs = phm_cfg.get("host_addrs")
    if not host_addrs:
        return HOST_SOCKET_ADDR_DEFAULT
    return [tuple(addr) for addr in host_addrs]


def build_monitor_output_provider(detectors, global_cfg):
    fault_levels = (global_cfg.get("alarm_push_config") or {}).get("fault_levels", {})

    def provider():
        return get_monitor_output_snapshot(detectors, fault_levels)

    return provider


def create_monitor_phm_reporter(global_cfg, detectors, package_dir=None):
    phm_cfg = global_cfg.get("phm_config", {})
    monitor_cfg = phm_cfg.get("monitor", {})
    if not phm_cfg.get("enabled", True):
        return None

    app_dir = package_dir or os.path.dirname(os.path.abspath(__file__))
    app_config_path = monitor_cfg.get("app_config", "AppConfig.xlsx")
    if not os.path.isabs(app_config_path):
        candidates = [
            os.path.join(app_dir, app_config_path),
            os.path.join(app_dir, "config", app_config_path),
        ]
        app_config_path = next((p for p in candidates if os.path.exists(p)), candidates[0])

    print(
        f"[PHM] AppConfig: {app_config_path}\n"
        f"[PHM] 输出字段: {OUTPUT_FIELD_NAMES}"
    )

    return PhmBackgroundReporter(
        algo_id=int(monitor_cfg.get("algo_id", 10008)),
        algo_port=int(monitor_cfg.get("algo_port", 39008)),
        app_config_path=app_config_path,
        output_provider=build_monitor_output_provider(detectors, global_cfg),
        host_socket_addr=_parse_host_addrs(phm_cfg),
        output_interval_ms=int(monitor_cfg.get("output_interval_ms", 1000)),
        log_status=bool(monitor_cfg.get("log_status", True)),
        log_output=bool(monitor_cfg.get("log_output", True)),
        output_field_names=OUTPUT_FIELD_NAMES,
    )
