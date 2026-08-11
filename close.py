#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
算法关闭程序 (close.py)

向本算法监听端口发送 PROG_ALGO_STOP(0x24)，触发算法进程执行 _self_stop() 干净退出。
行为对齐 signalProcess/shiwen：应用管理程序下发停止指令 → 算法上报离线 → os._exit(0)。

用法:
  python3 close.py
  python3 close.py 39008
"""

import os
import socket
import sys

import yaml

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from common.protocol import PROG_ALGO_STOP, pack_message_header


def resolve_algo_port(default_port=39008):
    if len(sys.argv) > 1:
        try:
            return int(sys.argv[1])
        except ValueError:
            pass

    cfg_path = os.path.join(BASE_DIR, "config_prod.yaml")
    if os.path.exists(cfg_path):
        with open(cfg_path, "r", encoding="utf-8") as fp:
            cfg = yaml.safe_load(fp) or {}
        monitor_cfg = ((cfg.get("phm_config") or {}).get("monitor") or {})
        if monitor_cfg.get("algo_port"):
            return int(monitor_cfg["algo_port"])
    return int(default_port)


def build_stop_packet():
    return pack_message_header(PROG_ALGO_STOP, 0)


def main():
    algo_port = resolve_algo_port(39008)
    packet = build_stop_packet()
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.sendto(packet, ("127.0.0.1", algo_port))
        print(
            f"[close] 已向 127.0.0.1:{algo_port} 发送 PROG_ALGO_STOP(0x24) "
            f"HEX={packet.hex().upper()}"
        )
    except OSError as e:
        print(f"[close] 发送停止指令失败: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        sock.close()


if __name__ == "__main__":
    main()
