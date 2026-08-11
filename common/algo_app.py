"""UDP 算法应用基类：心跳上报与结果上报。"""

import os
import socket
import struct
import threading
import time

from .config import load_app_config
from .protocol import (
    HEADER_FORMAT,
    HEADER_SIZE,
    HOST_SOCKET_ADDR_DEFAULT,
    PROG_ALGO_DATA,
    PROG_ALGO_STOP,
    PROG_ALGO_STAT,
    RECV_BUFFER_SIZE,
    SEND_INTERVAL_SECONDS,
    STAT_DATA_LEN,
    STAT_NORMAL,
    STAT_OFFLINE,
    STAT_TIMEOUT,
    STATUS_FORMAT,
    build_result_packet,
    format_status_packet,
    format_result_packet,
    pack_message_header,
)


class AlgoUdpAppBase:
    """UDP 算法服务公共基类。"""

    def __init__(
        self,
        algo_id,
        algo_port,
        app_config_path,
        host_socket_addr=HOST_SOCKET_ADDR_DEFAULT,
        *,
        recv_timeout=SEND_INTERVAL_SECONDS,
        log_status=False,
    ):
        algo_id, algo_version, _ = load_app_config(app_config_path, algo_id)
        self.log_status = log_status

        self.stat_list = [algo_id, algo_version, os.getpid(), 0, STAT_NORMAL]
        self.running = True
        self.result_sequence = 0

        self.local_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.local_socket.bind(("0.0.0.0", algo_port))
        self.local_socket.settimeout(recv_timeout)
        self.stat_list[3] = self.local_socket.getsockname()[1]

        self.host_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        if host_socket_addr and isinstance(host_socket_addr[0], (list, tuple)):
            self.host_socket_addrs = [tuple(addr) for addr in host_socket_addr]
        else:
            self.host_socket_addrs = [tuple(host_socket_addr)]
        self.host_socket_addr = self.host_socket_addrs[0]

    def _send_to_hosts(self, packet):
        for addr in self.host_socket_addrs:
            try:
                self.host_socket.sendto(packet, addr)
            except socket.error as e:
                print(f"[PHM] 发送到 {addr[0]}:{addr[1]} 失败: {e}")

    def report_status_loop(self):
        packet_header = pack_message_header(PROG_ALGO_STAT, STAT_DATA_LEN)

        while self.running:
            try:
                status_payload = struct.pack(STATUS_FORMAT, *self.stat_list)
                packet = packet_header + status_payload
                self._send_to_hosts(packet)

                if self.log_status:
                    targets = ", ".join(f"{addr[0]}:{addr[1]}" for addr in self.host_socket_addrs)
                    print(f"{format_status_packet(packet)}\n  目标: {targets}", flush=True)
            except socket.error as e:
                print(f"[PHM] 发送心跳失败: {e}")

            time.sleep(SEND_INTERVAL_SECONDS)

    def send_offline_status(self):
        self.stat_list[4] = STAT_OFFLINE
        packet_header = pack_message_header(PROG_ALGO_STAT, STAT_DATA_LEN)
        status_payload = struct.pack(STATUS_FORMAT, *self.stat_list)
        packet = packet_header + status_payload
        self._send_to_hosts(packet)
        if self.log_status:
            print(format_status_packet(packet), flush=True)

    def close(self):
        self.running = False
        try:
            self.local_socket.close()
        except OSError:
            pass
        try:
            self.host_socket.close()
        except OSError:
            pass

    def _self_stop(self):
        """收到应用管理程序的停止指令后退出进程（与 shiwen 一致）。

        先上报离线状态，关闭套接字，再 os._exit(0) 干净退出。
        下次由应用管理程序重新拉起 start.sh，算法以全新进程运行。
        """
        print("[停止] 收到停止指令，算法进程即将退出...", flush=True)
        try:
            self.send_offline_status()
        except Exception:
            pass
        self.running = False
        for sock in (self.local_socket, self.host_socket):
            try:
                sock.close()
            except OSError:
                pass
        os._exit(0)

    def recv_command_loop(self):
        while self.running:
            try:
                recv_data, _ = self.local_socket.recvfrom(RECV_BUFFER_SIZE)
            except socket.timeout:
                continue

            if len(recv_data) < HEADER_SIZE:
                continue

            header_data = struct.unpack(HEADER_FORMAT, recv_data[:HEADER_SIZE])
            command = header_data[2]

            if command == PROG_ALGO_DATA:
                continue
            if command == PROG_ALGO_STOP:
                self._self_stop()


class PhmBackgroundReporter(AlgoUdpAppBase):
    """后台 PHM 心跳 + 周期性故障/结果上报（不阻塞主业务）。"""

    def __init__(
        self,
        algo_id,
        algo_port,
        app_config_path,
        output_provider,
        host_socket_addr=HOST_SOCKET_ADDR_DEFAULT,
        *,
        output_interval_ms=1000,
        log_status=True,
        log_output=False,
        output_field_names=None,
    ):
        super().__init__(
            algo_id,
            algo_port,
            app_config_path,
            host_socket_addr,
            log_status=log_status,
        )
        self.output_provider = output_provider
        self.output_interval_ms = output_interval_ms
        self.log_output = log_output
        self.output_field_names = output_field_names or []
        self._threads = []

    def send_algo_result_loop(self):
        while self.running:
            time.sleep(self.output_interval_ms / 1000.0)

            try:
                result_len, result_payload = self.output_provider()
            except Exception as e:
                print(f"[PHM] 构建结果包失败: {e}")
                continue

            if result_len <= 0:
                continue

            self.result_sequence += 1
            try:
                packet = build_result_packet(
                    self.stat_list, self.result_sequence, result_len, result_payload
                )
                self._send_to_hosts(packet)
                if self.log_output:
                    print(format_result_packet(packet, self.output_field_names), flush=True)
            except socket.error as e:
                print(f"[PHM] 发送故障/结果包失败: {e}")

    def start(self):
        self._threads = [
            threading.Thread(target=self.report_status_loop, daemon=True, name="phm-heartbeat"),
            threading.Thread(target=self.send_algo_result_loop, daemon=True, name="phm-result"),
            threading.Thread(target=self.recv_command_loop, daemon=True, name="phm-recv"),
        ]
        for thread in self._threads:
            thread.start()
        print(
            f"[PHM] 已启动心跳与故障上报 (算法ID={self.stat_list[0]}, "
            f"监听端口={self.stat_list[3]})"
        )

    def stop(self):
        try:
            self.send_offline_status()
        except Exception:
            pass
        self.close()
