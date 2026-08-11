#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""弓网AI监测算法包入口 (algo_id=10008)，结构参考 signalProcess/shiwen。"""

import os
import sys
import time

os.environ['TZ'] = 'Asia/Shanghai'
if hasattr(time, 'tzset'):
    time.tzset()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

import monitor_main as monitor_app


def main():
    cfg = os.path.join(BASE_DIR, "config_prod.yaml")
    if len(sys.argv) < 2 and os.path.exists(cfg):
        sys.argv.append(cfg)
    os.chdir(BASE_DIR)
    monitor_app.app_dir = BASE_DIR
    monitor_app.src_dir = BASE_DIR
    monitor_app.main()


if __name__ == "__main__":
    main()
