#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI 实时故障检测与告警上报引擎 (ai_detector.py)
功能：
1. 加载 YOLOv11 模型对多路受电弓 RTSP 视频流进行实时推理；
2. 运行 6 种故障检测状态机（大火花、受电弓倾斜、画面脏污、悬挂异物、结构异常、异常飘弓）；
3. 遵循《受电弓故障报警报文》规范（图片1.png），发射 0xFF 0x20 标准二进制 UDP 告警报文；
4. 算法结果按 `%m-%d` 月-日建目录落盘：`起始时间-终止时间-CR400BF-PCM Algo result-故障代码.jpg/.mp4`。
"""

import os
import sys
import time
import socket
import struct
import cv2
import yaml
import numpy as np
import datetime
import threading
import queue
from collections import deque
import datetime

FAULT_TYPE_MAP = {
    "大火花": 3,
    "受电弓结构异常": 4,
    "画面脏污": 10,
    "受电弓倾斜": 12,
    "高压故障": 13,
    "悬挂异物": 14,
    "异常飘弓": 15,
}

from PIL import Image, ImageDraw, ImageFont


def draw_osd_watermark(
    frame, train_id="CR400BF", car_number="05", speed=0, camera_id="0", ts=None
):
    """在 OpenCV 图像帧上叠加 3+2 双侧 OSD 水印"""
    try:
        img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)

        candidate_fonts = [
            "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
            "C:/Windows/Fonts/msyh.ttc",
            "C:/Windows/Fonts/simhei.ttf",
        ]
        font_path = next((f for f in candidate_fonts if os.path.exists(f)), None)
        font_size = 26

        if font_path:
            font = ImageFont.truetype(font_path, font_size)
        else:
            font = ImageFont.load_default()

        h, w = frame.shape[:2]
        dt_str = datetime.datetime.fromtimestamp(ts or time.time()).strftime(
            "%Y-%m-%d %H:%M:%S"
        )

        car_str = str(car_number).zfill(2)
        cam_str = str(camera_id)

        left_lines = [f"车次: {train_id}", f"车厢号: {car_str}", f"速度: {speed}km/h"]
        right_lines = [f"{dt_str}", f"位置号: {cam_str}"]

        # 绘制左下角 (3行)
        y_start_left = h - 110
        for i, line in enumerate(left_lines):
            y = y_start_left + i * 32
            draw.text((30, y), line, font=font, fill=(255, 255, 255))

        # 绘制右下角 (2行)
        y_start_right = h - 78
        for i, line in enumerate(right_lines):
            y = y_start_right + i * 32
            if hasattr(font, "getbbox"):
                bbox = font.getbbox(line)
                text_w = bbox[2] - bbox[0]
            else:
                text_w = len(line) * 14
            x = max(10, w - text_w - 30)
            draw.text((x, y), line, font=font, fill=(255, 255, 255))

        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    except Exception:
        return frame


def get_result_dir_for_timestamp(ts, base_dir):
    """根据故障触发时间戳自动创建并返回 base_dir/月-日 文件夹"""
    dt = datetime.datetime.fromtimestamp(ts)
    month_day = dt.strftime("%m-%d")
    daily_dir = os.path.join(base_dir, month_day)
    os.makedirs(daily_dir, exist_ok=True)
    return daily_dir


def report_event_binary(
    udp_sock,
    alarm_cfg,
    hw_cfg,
    frame,
    event_name,
    camera_id=None,
    car_number=None,
    save_dir="./events",
    pic_path=None,
):
    if pic_path and os.path.exists(pic_path):
        full_path_str = os.path.abspath(pic_path)
    else:
        os.makedirs(save_dir, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        filename = f"{event_name}_{ts}.jpg"
        full_path_str = os.path.abspath(os.path.join(save_dir, filename))
        cv2.imwrite(full_path_str, frame)

    type_code = FAULT_TYPE_MAP.get(event_name, 10)

    try:
        cam_id = camera_id if camera_id is not None else hw_cfg.get("camera_id", 0)
        car_num = car_number if car_number is not None else hw_cfg.get("car_number", 0)

        cfg_level = alarm_cfg["fault_levels"].get(event_name, 2)
        fault_level = 0x01 if cfg_level == 1 else 0x03

        # 遵循图片1.png规范:
        # Data数据域: 1B故障级别 + 2B故障类型(1Byte) + 3B~4B故障代码(2Byte) + 5B~10B车体号(6Byte) + 11B~NB路径名
        fault_type_byte = struct.pack("B", type_code)
        fault_code_bytes = struct.pack(">H", 0x0000)
        dev_info_bytes = struct.pack(">HI", car_num, cam_id)
        path_bytes = full_path_str.encode("utf-8")

        data_field = (
            struct.pack("B", fault_level)
            + fault_type_byte
            + fault_code_bytes
            + dev_info_bytes
            + path_bytes
        )
        data_len = len(data_field)

        packet_header = struct.pack(">B B H", 0xFF, 0x20, data_len)
        packet_base = packet_header + data_field
        checksum = sum(packet_base) & 0xFF
        final_binary_packet = packet_base + struct.pack("B", checksum)

        udp_sock.sendto(
            final_binary_packet, (alarm_cfg["udp_ip"], alarm_cfg["udp_port"])
        )

        print(
            f"\n[AI-告警推送] 成功上报故障 -> 事件:【{event_name}】 CAR{car_num}-CAM{cam_id}"
        )
        print(
            f"  └─ 协议头: 0xFF 0x20 | 数据长度: {data_len} 字节 | 校验和: {hex(checksum)}"
        )
        print(f"  └─ Hex 码流: {final_binary_packet.hex().upper()}")
        print(f"  └─ 图片路径: {full_path_str}")

    except Exception as e:
        print(f"[ERROR] 发射二进制故障报文失败: {e}", file=sys.stderr)


class RTSPStreamSampler:
    """异步多线程 RTSP / 本地视频拉流采样器"""
    def __init__(self, source_url, sample_interval=1.0, is_file=False):
        self.source_url = source_url
        self.sample_interval = sample_interval
        self.is_file = is_file
        self.running = False
        self.cap = None
        self.frame_queue = queue.Queue(maxsize=30)
        self.last_sample_time = 0.0

    def start(self):
        self.running = True
        self.cap = cv2.VideoCapture(self.source_url)
        if not self.cap.isOpened():
            raise RuntimeError(f"无法打开视频源: {self.source_url}")
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def _worker(self):
        fps = self.cap.get(cv2.CAP_PROP_FPS) or 25.0
        frame_delay = 1.0 / fps if self.is_file else 0.001
        frame_count = 0

        while self.running:
            ret, frame = self.cap.read()
            if not ret or frame is None:
                if self.is_file:
                    self.frame_queue.put(None)
                    break
                else:
                    time.sleep(1.0)
                    continue

            now = time.time()
            if self.is_file:
                sample_stride = max(1, int(round(fps * self.sample_interval)))
                if frame_count % sample_stride == 0:
                    self._push_frame(frame, now)
                frame_count += 1
                time.sleep(frame_delay)
            else:
                if (now - self.last_sample_time) >= self.sample_interval:
                    self._push_frame(frame, now)
                    self.last_sample_time = now
                time.sleep(frame_delay)

    def _push_frame(self, frame, timestamp):
        if self.frame_queue.full():
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                pass
        self.frame_queue.put((frame, timestamp))

    def get_frame(self, timeout=5.0):
        try:
            return self.frame_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def stop(self):
        self.running = False
        if hasattr(self, 'thread') and self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.5)
        if self.cap:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None


class PantographFaultDetector:
    def __init__(self, global_cfg, stream_cfg, udp_sock, stop_event, vlm_engine):
        self.global_cfg = global_cfg
        self.vs_cfg = global_cfg["video_stream_config"]
        self.algo_cfg = global_cfg["base_algorithm_config"]
        self.alarm_cfg = global_cfg["alarm_push_config"]
        self.hw_cfg = global_cfg["hardware_config"]
        self.udp_sock = udp_sock
        self.stop_event = stop_event

        self.rtsp_url = stream_cfg.get("rtsp_url")
        self.camera_id = stream_cfg.get("camera_id", self.hw_cfg.get("camera_id", 0))
        self.car_number = stream_cfg.get("car_number", self.hw_cfg.get("car_number", 0))
        self.result_base_dir = stream_cfg.get(
            "result_base_dir", "/hirain/hi_opt/nvme1/Algo-result"
        )

        # [Qwen3-VL 4B 共享引擎]
        self.vlm_engine = vlm_engine
        print(
            f"[AI-CAR{self.car_number}-CAM{self.camera_id}] Qwen3-VL 4B 统一两阶段推理引擎接入完成"
        )

        self.frame_count = 0
        fps = self.vs_cfg.get("fps", 30)

        self.dirty_window = int(self.alarm_cfg["dirty_check_window_min"] * 60 * fps)
        self.dirty_history = deque(maxlen=self.dirty_window)

        self.spark_duration_thresh = int(
            (self.alarm_cfg["spark_duration_ms"] / 1000.0) * fps
        )
        self.common_response_frames = int(
            self.alarm_cfg["common_response_time_s"] * fps
        )

        self.standard_template = None
        self.dirty_alarmed = False
        self.spark_duration = 0
        self.spark_alarmed = False
        self.foreign_frames = 0
        self.foreign_alarmed = False
        self.structure_abnormal_cnt = 0
        self.structure_alarmed = False
        self.tilt_abnormal_cnt = 0
        self.tilt_alarmed = False
        self.float_abnormal_cnt = 0
        self.float_alarmed = False

        self.fps = fps
        self.pre_record_seconds = 60
        self.post_record_seconds = 60
        buffer_size = int(
            (self.pre_record_seconds + self.post_record_seconds + 30) * self.fps
        )
        self.frame_buffer = deque(maxlen=buffer_size)
        self.pending_recordings = []

        # 4 帧时序窗口缓冲队列（1 FPS 采样，窗口步长为 4 帧）
        self.window_size = 4
        self.sample_interval = 1.0
        self.window_buffer = deque(maxlen=self.window_size)
        self.window_count = 0
        self.last_sample_ts = 0.0
        self._current_vlm_res = {}

        self.ftp_remote_dir = stream_cfg.get(
            "ftp_remote_dir",
            global_cfg.get("ftp_config", {}).get(
                "monitor_remote_dir", "/hirain/hi_opt/nvme1/Algo-result"
            ),
        )

    def _ftp_upload_local_file(self, local_path):
        from ftp_uploader import get_ftp_uploader

        uploader = get_ftp_uploader()
        if (
            not uploader
            or not uploader.enabled
            or not local_path
            or not os.path.exists(local_path)
        ):
            return
        try:
            rel = os.path.relpath(local_path, self.result_base_dir).replace("\\", "/")
        except ValueError:
            rel = os.path.basename(local_path)
        remote_path = f"{self.ftp_remote_dir.rstrip('/')}/{rel}"
        print(f"[AI-FTP] 上传算法结果: {local_path} -> {remote_path}")
        uploader.upload_and_delete(local_path, remote_path)

    def get_active_fault_flags(self):
        """返回与 AppConfig 输出 0~5 对齐的故障激活标志。"""
        return [
            int(self.spark_alarmed),  # 0 大火花
            int(self.structure_alarmed),  # 1 受电弓结构异常
            int(self.dirty_alarmed),  # 2 画面脏污
            int(self.tilt_alarmed),  # 3 受电弓倾斜
            int(self.foreign_alarmed),  # 4 悬挂异物
            int(self.float_alarmed),  # 5 异常飘弓
        ]

    def _generate_base_name_and_dir(self, fault_ts, fault_code):
        date_dir = get_result_dir_for_timestamp(fault_ts, self.result_base_dir)
        start_ts = fault_ts - self.pre_record_seconds
        end_ts = fault_ts + self.post_record_seconds
        start_str = time.strftime("%H%M%S", time.localtime(start_ts))
        end_str = time.strftime("%H%M%S", time.localtime(end_ts))
        base_name = f"{start_str}-{end_str}-CR400BF-PCM_Algo_result-{fault_code}"
        event_save_dir = os.path.join(date_dir, base_name)
        os.makedirs(event_save_dir, exist_ok=True)
        return base_name, event_save_dir, start_ts, end_ts

    def save_fault_picture(self, frame, event_name, fault_code, current_ts):
        base_name, save_dir, start_ts, end_ts = self._generate_base_name_and_dir(
            current_ts, fault_code
        )
        pic_filename = f"{base_name}.jpg"
        pic_path = os.path.join(save_dir, pic_filename)
        cv2.imwrite(pic_path, frame)
        print(
            f"[Algo-result保存][CAR{self.car_number}-CAM{self.camera_id}] 快照已落盘: {pic_path}"
        )
        self._ftp_upload_local_file(pic_path)
        return pic_path

    def start_post_fault_recording(self, event_name, fault_code, fault_ts):
        base_name, save_dir, start_ts, end_ts = self._generate_base_name_and_dir(
            fault_ts, fault_code
        )
        pre_frames = [f for ts, f in self.frame_buffer if ts >= start_ts]
        pending = {
            "base_name": base_name,
            "pre_frames": pre_frames,
            "post_frames": [],
            "fault_time": fault_ts,
            "end_time": end_ts,
            "start_time": start_ts,
            "event_name": event_name,
            "save_dir": save_dir,
        }
        self.pending_recordings.append(pending)
        print(
            f"[AI-录像][CAR{self.car_number}-CAM{self.camera_id}] 启动故障【{event_name}】关联视频录制"
        )

    def collect_post_frames(self, current_ts, jpeg_buf):
        for rec in self.pending_recordings:
            if rec["fault_time"] < current_ts <= rec["end_time"]:
                rec["post_frames"].append(jpeg_buf)

    def finalize_completed_recordings(self):
        still_pending = []
        current_time = time.time()
        for rec in self.pending_recordings:
            if current_time >= rec["end_time"]:
                all_frames = rec["pre_frames"] + rec["post_frames"]
                if all_frames:
                    try:
                        first_frame = cv2.imdecode(all_frames[0], cv2.IMREAD_COLOR)
                        if first_frame is None:
                            continue
                        h, w = first_frame.shape[:2]
                        video_filename = f"{rec['base_name']}.mp4"
                        video_path = os.path.join(rec["save_dir"], video_filename)

                        real_covered_seconds = rec["end_time"] - rec["start_time"]
                        calc_fps = (
                            len(all_frames) / real_covered_seconds
                            if real_covered_seconds > 0
                            else self.fps
                        )
                        target_fps = float(max(1.0, min(60.0, calc_fps)))

                        # 使用标准的 H.264 编码 (avc1) 提升播放流畅度与浏览器/播放器兼容性
                        fourcc = cv2.VideoWriter_fourcc(*"avc1")
                        out = cv2.VideoWriter(video_path, fourcc, target_fps, (w, h))
                        if not out.isOpened():
                            # 回退至 mp4v 编码
                            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                            out = cv2.VideoWriter(
                                video_path, fourcc, target_fps, (w, h)
                            )

                        for jpeg_buf in all_frames:
                            frame = cv2.imdecode(jpeg_buf, cv2.IMREAD_COLOR)
                            if frame is not None:
                                out.write(frame)
                        out.release()
                        print(
                            f"[Algo-result保存][CAR{self.car_number}-CAM{self.camera_id}] 前后关联视频已生成: {video_path}"
                        )
                        self._ftp_upload_local_file(video_path)
                    except Exception as e:
                        print(f"[ERROR] 关联视频保存失败: {e}", file=sys.stderr)
            else:
                still_pending.append(rec)
        self.pending_recordings = still_pending

    def detect_dirty(self, result):
        if result.boxes is None or len(result.boxes) == 0:
            self.dirty_history.append(0)
        else:
            panto_confs = [
                box.conf.item() for box in result.boxes if int(box.cls.item()) == 1
            ]
            if panto_confs and max(panto_confs) > self.algo_cfg["dirty_conf_thresh"]:
                self.dirty_history.append(1)
            else:
                self.dirty_history.append(0)

        if len(self.dirty_history) >= self.dirty_window:
            dirty_ratio = 1 - (sum(self.dirty_history) / len(self.dirty_history))
            if dirty_ratio > self.algo_cfg["dirty_ratio_thresh"]:
                if not self.dirty_alarmed:
                    self.dirty_alarmed = True
                    return True
            else:
                self.dirty_alarmed = False
        return False

    def detect_big_spark(self, result):
        has_big_spark = False
        if result.boxes is not None and len(result.boxes) > 0:
            total_spark_area = 0
            max_panto_area = 0
            for box in result.boxes:
                cls_id = int(box.cls.item())
                xyxy = box.xyxy[0].tolist()
                area = (xyxy[2] - xyxy[0]) * (xyxy[3] - xyxy[1])
                if cls_id == 2:
                    total_spark_area += area
                elif cls_id == 1:
                    if area > max_panto_area:
                        max_panto_area = area
            if (
                max_panto_area > 0
                and (total_spark_area / max_panto_area)
                > self.algo_cfg["spark_panto_area_ratio"]
            ):
                has_big_spark = True

        if has_big_spark:
            self.spark_duration += 1
        else:
            self.spark_duration = max(0, self.spark_duration - 1)

        if self.spark_duration >= self.spark_duration_thresh:
            if not self.spark_alarmed:
                self.spark_alarmed = True
                return True
        else:
            self.spark_alarmed = False
        return False

    def detect_foreign_object(self, result):
        has_large_foreign = False
        min_w, min_h = self.algo_cfg["foreign_object_min_size"]
        if result.boxes is not None:
            for box in result.boxes:
                if int(box.cls.item()) == 0:
                    xyxy = box.xyxy[0].tolist()
                    w, h = xyxy[2] - xyxy[0], xyxy[3] - xyxy[1]
                    if w >= min_w and h >= min_h:
                        has_large_foreign = True
                        break
        if has_large_foreign:
            self.foreign_frames += 1
        else:
            self.foreign_frames = max(0, self.foreign_frames - 2)

        if self.foreign_frames >= self.common_response_frames:
            if not self.foreign_alarmed:
                self.foreign_alarmed = True
                return True
        else:
            self.foreign_alarmed = False
        return False

    def detect_structure_anomaly(self, result, frame):
        if result.boxes is None:
            return False
        panto_boxes = [box for box in result.boxes if int(box.cls.item()) == 1]
        if not panto_boxes:
            return False

        box = panto_boxes[0].xyxy[0].tolist()
        x1, y1, x2, y2 = map(int, box)
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return False
        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        if self.standard_template is None:
            self.standard_template = roi_gray.copy()
            return False

        try:
            resized_roi = cv2.resize(
                roi_gray,
                (self.standard_template.shape[1], self.standard_template.shape[0]),
            )
            ncc_res = cv2.matchTemplate(
                resized_roi, self.standard_template, cv2.TM_CCOEFF_NORMED
            )
            ncc_score = np.max(ncc_res)
            if ncc_score < self.algo_cfg["structure_ncc_thresh"]:
                self.structure_abnormal_cnt += 1
            else:
                self.structure_abnormal_cnt = 0
        except Exception:
            self.structure_abnormal_cnt = 0

        if self.structure_abnormal_cnt >= self.common_response_frames:
            if not self.structure_alarmed:
                self.structure_alarmed = True
                return True
        else:
            self.structure_alarmed = False
        return False

    def detect_tilt(self, result, frame):
        if result.boxes is None:
            return False
        panto_boxes = [box for box in result.boxes if int(box.cls.item()) == 1]
        if not panto_boxes:
            return False

        box = panto_boxes[0].xyxy[0].tolist()
        x1, y1, x2, y2 = map(int, box)
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return False

        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(roi_gray, 50, 150)
        lines = cv2.HoughLinesP(
            edges, 1, np.pi / 180, threshold=30, minLineLength=30, maxLineGap=10
        )

        if lines is not None:
            angles = []
            for line in lines:
                coords = line.flatten()
                if len(coords) == 4:
                    lx1, ly1, lx2, ly2 = coords
                    if abs(lx2 - lx1) > 1e-5:
                        angle = np.degrees(np.arctan2(ly2 - ly1, lx2 - lx1))
                        angles.append(angle)
            if angles:
                current_angle_avg = np.mean(angles)
                angle_deviation = abs(current_angle_avg - 0.0)
                if angle_deviation > self.algo_cfg["tilt_angle_bias_thresh"]:
                    self.tilt_abnormal_cnt += 1
                else:
                    self.tilt_abnormal_cnt = 0

        if self.tilt_abnormal_cnt >= self.common_response_frames:
            if not self.tilt_alarmed:
                self.tilt_alarmed = True
                return True
        else:
            self.tilt_alarmed = False
        return False

    def detect_arc_height(self, result, is_panto_raised=True):
        if is_panto_raised:
            self.float_abnormal_cnt = 0
            self.float_alarmed = False
            return False

        if result.boxes is not None:
            panto_boxes = [box for box in result.boxes if int(box.cls.item()) == 1]
            if panto_boxes:
                box = panto_boxes[0].xyxy[0].tolist()
                current_height = box[3] - box[1]
                if current_height > self.algo_cfg["panto_down_height_limit"]:
                    self.float_abnormal_cnt += 1
                else:
                    self.float_abnormal_cnt = 0
            else:
                self.float_abnormal_cnt = 0
        else:
            self.float_abnormal_cnt = 0

        if self.float_abnormal_cnt >= self.common_response_frames:
            if not self.float_alarmed:
                self.float_alarmed = True
                return True
        else:
            self.float_alarmed = False
        return False

    def run(self):
        while not self.stop_event.is_set():
            sampler = None
            try:
                is_file = os.path.exists(self.rtsp_url) or (
                    not self.rtsp_url.startswith("rtsp://") and not self.rtsp_url.startswith("http://")
                )
                print(
                    f"[AI-CAR{self.car_number}-CAM{self.camera_id}] 启动 Qwen3-VL 异步流采样器 -> {self.rtsp_url} (is_file={is_file})"
                )
                sampler = RTSPStreamSampler(
                    self.rtsp_url, sample_interval=self.sample_interval, is_file=is_file
                )
                sampler.start()

                while not self.stop_event.is_set():
                    item = sampler.get_frame(timeout=5.0)
                    if item is None:
                        if is_file:
                            print(
                                f"[AI-CAR{self.car_number}-CAM{self.camera_id}] 测试视频已读取完毕。"
                            )
                            break
                        continue

                    raw_frame, current_ts = item
                    self.frame_count += 1
                    self.window_buffer.append(raw_frame.copy())

                    # 1. 收集满 4 帧后执行一次推理，再开始收集下一个窗口。
                    #    Stage2 异物需在两个相邻的 4 秒窗口连续命中且点位接近才告警。
                    if len(self.window_buffer) == self.window_size:
                        self.window_count += 1
                        t_infer_start = time.time()
                        vlm_res = self.vlm_engine.infer_window(
                            list(self.window_buffer), timestamp=current_ts
                        )
                        t_infer_end = time.time()

                        # 更新内部状态
                        self.spark_alarmed = bool(vlm_res.get("spk", 0))
                        self.structure_alarmed = bool(vlm_res.get("str", 0))
                        self.dirty_alarmed = bool(vlm_res.get("dirt", 0))
                        self.tilt_alarmed = bool(vlm_res.get("tilt", 0))
                        self.foreign_alarmed = bool(vlm_res.get("obj", 0))
                        self.float_alarmed = bool(vlm_res.get("flt", 0))
                        self._current_vlm_res = vlm_res

                        flags = []
                        if self.spark_alarmed:
                            flags.append("🔥 大火花")
                        if self.dirty_alarmed:
                            flags.append("🌫️ 画面脏污")
                        if self.foreign_alarmed:
                            cand = vlm_res.get("temporal_cand")
                            hit_cnt = cand.hit_count if cand else 2
                            flags.append(f"🎯 悬挂异物 (时序已确认! 命中{hit_cnt}次)")
                        elif vlm_res.get("temporal_status") == "SUSPECT":
                            cand = vlm_res.get("temporal_cand")
                            hit_cnt = cand.hit_count if cand else 1
                            flags.append(f"⏳ 疑似异物 (时序观察中 命中{hit_cnt}/2次, 暂不虚警)")
                        if self.structure_alarmed:
                            flags.append("⚠️ 结构异常")
                        if self.tilt_alarmed:
                            flags.append("📐 倾斜异常")
                        if self.float_alarmed:
                            flags.append("〰️ 异常飘弓")
                        status_str = " | ".join(flags) if flags else "✅ 正常"

                        print(
                            f"\n⚡ ===== [AI-CAR{self.car_number}-CAM{self.camera_id}] [时序窗口 #{self.window_count}] Stage 1 & Stage 2 动静分离诊断 ====="
                        )
                        # 实时延迟监控计算
                        lag_sec = t_infer_end - current_ts
                        queue_backlog = sampler.frame_queue.qsize() if sampler else 0
                        lag_warn = " ⚠️ [高滞后警告: 大模型处理速度低于推流速度，正在处理历史积压画面!]" if lag_sec > 3.0 else " ✅ [实时同步中]"
                        print(
                            f"  ⏱️ [流时延监控] 帧采集时标: {datetime.datetime.fromtimestamp(current_ts).strftime('%H:%M:%S.%f')[:-3]} | 当前现实时钟: {datetime.datetime.fromtimestamp(t_infer_end).strftime('%H:%M:%S.%f')[:-3]} | 真实时延差: {lag_sec:.2f}s (队列积压: {queue_backlog} 帧){lag_warn}"
                        )
                        print(
                            f"  ├─ S1 (视频流) 耗时: {vlm_res.get('s1_ms', 0):.0f} ms | 推理: {vlm_res.get('reasoning', '')}"
                        )
                        if vlm_res.get("s2_text"):
                            s2_out = vlm_res.get("s2_text", "")
                            s2_short = s2_out[:90] + "..." if len(s2_out) > 90 else s2_out
                            print(f"  ├─ S2 (定位) 耗时: {vlm_res.get('s2_ms', 0):.0f} ms | 输出: {s2_short}")

                        for exp_str in vlm_res.get("temporal_expired_log", []):
                            print(f"  🛡️ [时序滤波] 成功过滤瞬时偶发误报: {exp_str} (未在后续窗口持续出现)")

                        print(f"  └─ 总体结论: {status_str}")
                        print("=" * 75, "\n", flush=True)

                        # 步长为 4：本窗口消费完毕后再收集下一组 4 帧。
                        self.window_buffer.clear()
                    else:
                        vlm_res = getattr(self, "_current_vlm_res", {})

                    # 2. 绘制检测框和准星
                    draw_frame = raw_frame.copy()
                    h, w = draw_frame.shape[:2]

                    if vlm_res.get("box") and self.spark_alarmed:
                        nx1, ny1, nx2, ny2 = vlm_res["box"]
                        x1 = max(0, min(w, int(nx1 / 1000.0 * w)))
                        y1 = max(0, min(h, int(ny1 / 1000.0 * h)))
                        x2 = max(0, min(w, int(nx2 / 1000.0 * w)))
                        y2 = max(0, min(h, int(ny2 / 1000.0 * h)))
                        cv2.rectangle(draw_frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                        cv2.putText(
                            draw_frame,
                            "Spark",
                            (x1, max(y1 - 10, 0)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.9,
                            (0, 0, 255),
                            2,
                        )

                    if vlm_res.get("point") and self.foreign_alarmed:
                        nx, ny = vlm_res["point"]
                        px = max(0, min(w - 1, int(nx / 1000.0 * w)))
                        py = max(0, min(h - 1, int(ny / 1000.0 * h)))
                        cv2.circle(draw_frame, (px, py), 15, (0, 0, 255), 3)
                        cv2.circle(draw_frame, (px, py), 4, (0, 0, 255), -1)
                        cv2.line(
                            draw_frame, (px - 25, py), (px + 25, py), (0, 0, 255), 2
                        )
                        cv2.line(
                            draw_frame, (px, py - 25), (px, py + 25), (0, 0, 255), 2
                        )
                        cv2.putText(
                            draw_frame,
                            "Foreign Object",
                            (max(px - 50, 10), max(py - 30, 30)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (0, 0, 255),
                            2,
                        )

                    if self.dirty_alarmed:
                        cv2.putText(
                            draw_frame,
                            "WARNING: Dirt / Blur Detected",
                            (50, 80),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1.2,
                            (0, 165, 255),
                            3,
                        )

                    # 3. 绘制 3+2 双侧 OSD 水印
                    static_speed = self.global_cfg.get("udp_speed_config", {}).get(
                        "static_speed", 0
                    )
                    train_id = self.global_cfg.get("train_id", "CR400BF")

                    final_frame = draw_osd_watermark(
                        draw_frame,
                        train_id=train_id,
                        car_number=self.car_number,
                        speed=static_speed,
                        camera_id=self.camera_id,
                        ts=current_ts,
                    )

                    # JPEG 压缩并存入 frame_buffer 避免 OOM
                    _, jpeg_buf = cv2.imencode(
                        ".jpg", final_frame, [cv2.IMWRITE_JPEG_QUALITY, 85]
                    )
                    self.frame_buffer.append((current_ts, jpeg_buf))
                    triggered = []

                    if self.dirty_alarmed and not getattr(
                        self, "_last_dirty_alarmed", False
                    ):
                        triggered.append(
                            ("画面脏污", FAULT_TYPE_MAP.get("画面脏污", 10))
                        )
                    self._last_dirty_alarmed = self.dirty_alarmed

                    if self.spark_alarmed and not getattr(
                        self, "_last_spark_alarmed", False
                    ):
                        triggered.append(("大火花", FAULT_TYPE_MAP.get("大火花", 3)))
                    self._last_spark_alarmed = self.spark_alarmed

                    if self.foreign_alarmed and not getattr(
                        self, "_last_foreign_alarmed", False
                    ):
                        triggered.append(
                            ("悬挂异物", FAULT_TYPE_MAP.get("悬挂异物", 14))
                        )
                    self._last_foreign_alarmed = self.foreign_alarmed

                    if self.structure_alarmed and not getattr(
                        self, "_last_structure_alarmed", False
                    ):
                        triggered.append(
                            ("受电弓结构异常", FAULT_TYPE_MAP.get("受电弓结构异常", 4))
                        )
                    self._last_structure_alarmed = self.structure_alarmed

                    if self.tilt_alarmed and not getattr(
                        self, "_last_tilt_alarmed", False
                    ):
                        triggered.append(
                            ("受电弓倾斜", FAULT_TYPE_MAP.get("受电弓倾斜", 12))
                        )
                    self._last_tilt_alarmed = self.tilt_alarmed

                    if self.float_alarmed and not getattr(
                        self, "_last_float_alarmed", False
                    ):
                        triggered.append(
                            ("异常飘弓", FAULT_TYPE_MAP.get("异常飘弓", 15))
                        )
                    self._last_float_alarmed = self.float_alarmed

                    for event_name, fault_code in triggered:
                        pic_path = self.save_fault_picture(
                            final_frame, event_name, fault_code, current_ts
                        )
                        report_event_binary(
                            self.udp_sock,
                            self.alarm_cfg,
                            self.hw_cfg,
                            final_frame,
                            event_name,
                            camera_id=self.camera_id,
                            car_number=self.car_number,
                            pic_path=pic_path,
                        )
                        self.start_post_fault_recording(
                            event_name, fault_code, current_ts
                        )

                    self.collect_post_frames(current_ts, jpeg_buf)
                    self.finalize_completed_recordings()

            except Exception as e:
                if self.stop_event.is_set():
                    break
                print(
                    f"[AI-CAR{self.car_number}-CAM{self.camera_id}] 视频流处理异常: {e}，5秒后自动重连..."
                )
                time.sleep(5)
            finally:
                if sampler:
                    sampler.stop()
